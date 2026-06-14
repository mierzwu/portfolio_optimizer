"""
database.py – Warstwa dostępu do bazy SQLite dla Optymalizatora Portfela.

Schemat:
  emisje_obligacji  – Parametry bieżących serii EDO/COI
  analizy_historia           – Historia wykonanych analiz (zawiera wszystkie parametry analizy)
  wyniki_wagi_portfela       – Docelowy skład portfela dla każdej analizy (1:N)
"""
import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# Domyślna ścieżka do pliku bazy danych (obok modułu)
DB_PATH = Path(__file__).parent / "database.db"

# Flaga idempotentności – initialize_db() loguje tylko raz na sesję
_DB_INITIALIZED: set = set()

# ─── DDL ─────────────────────────────────────────────────────────────────────

_DDL_STATEMENTS = [
    # Tabela 1 – Parametry aktualnych list emisyjnych obligacji
    """
    CREATE TABLE IF NOT EXISTS emisje_obligacji (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        typ_obligacji        TEXT NOT NULL,
        symbol_emisji        TEXT NOT NULL UNIQUE,
        data_poczatkowa      TEXT NOT NULL,
        data_zakonczenia     TEXT NOT NULL,
        dlugosc_lat          INTEGER NOT NULL,
        oprocentowanie_rok_1 REAL NOT NULL,
        marza_odsetkowa      REAL NOT NULL,
        kara_wykup           REAL NOT NULL,
        data_aktualizacji    TEXT NOT NULL
    )
    """,
    # Tabela 2 – Historia analiz (parametry analizy + parametry konfiguracji wbudowane)
    """
    CREATE TABLE IF NOT EXISTS analizy_historia (
        id                        INTEGER PRIMARY KEY AUTOINCREMENT,
        data_analizy              TEXT    NOT NULL,
        nazwa_strategii           TEXT    NOT NULL DEFAULT 'optymalizacja',
        cel_optymalizacji         TEXT    NOT NULL DEFAULT 'min_risk',
        horyzont_inwestycyjny_lat INTEGER NOT NULL DEFAULT 4,
        prowizja_maklerska        REAL,
        cvar_alpha                REAL    NOT NULL,
        oczekiwana_stopa_zwrotu   REAL    NOT NULL,
        wartosc_ryzyka_cvar       REAL    NOT NULL,
        koszt_rebalancingu_netto  REAL
    )
    """,
    # Tabela 3 – Docelowy skład i wagi portfela po analizie (1:N → analizy_historia)
    """
    CREATE TABLE IF NOT EXISTS wyniki_wagi_portfela (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        analiza_id    INTEGER NOT NULL,
        ticker_aktywa TEXT    NOT NULL,
        klasa_aktywa  TEXT    NOT NULL CHECK(klasa_aktywa IN ('AKCJA', 'OBLIGACJA')),
        waga_docelowa REAL    NOT NULL,
        FOREIGN KEY (analiza_id) REFERENCES analizy_historia(id) ON DELETE CASCADE
    )
    """,
]

# ─── Connection management ────────────────────────────────────────────────────

@contextmanager
def get_connection(db_path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """
    Kontekst zwracający połączenie z włączonymi kluczami obcymi.
    Przy wyjściu bez wyjątku wykonuje COMMIT, w razie błędu ROLLBACK.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_db(db_path: Path = DB_PATH) -> None:
    """Tworzy wszystkie tabele (jeśli nie istnieją).

    Idempotentna: kolejne wywołania z tą samą ścieżką są no-op.
    """
    db_key = str(db_path.resolve())
    if db_key in _DB_INITIALIZED:
        return
    with get_connection(db_path) as conn:
        for stmt in _DDL_STATEMENTS:
            conn.execute(stmt)
    _DB_INITIALIZED.add(db_key)
    logger.info(f"[DB] Baza danych gotowa: {db_path}")


# ─── emisje_obligacji ───────────────────────────────────────────────────────

def upsert_bond_emission(
    typ: str,
    symbol: str,
    data_poczatkowa: str,
    data_zakonczenia: str,
    dlugosc_lat: int,
    oprocentowanie_rok_1: float,
    marza: float,
    kara: float,
    db_path: Path = DB_PATH,
) -> None:
    """
    Wstawia nową emisję lub aktualizuje istniejącą (na podstawie UNIQUE symbol_emisji).
    """
    sql = """
        INSERT INTO emisje_obligacji
            (typ_obligacji, symbol_emisji, data_poczatkowa, data_zakonczenia,
             dlugosc_lat, oprocentowanie_rok_1, marza_odsetkowa, kara_wykup, data_aktualizacji)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol_emisji) DO UPDATE SET
            data_poczatkowa      = excluded.data_poczatkowa,
            data_zakonczenia     = excluded.data_zakonczenia,
            dlugosc_lat          = excluded.dlugosc_lat,
            oprocentowanie_rok_1 = excluded.oprocentowanie_rok_1,
            marza_odsetkowa      = excluded.marza_odsetkowa,
            kara_wykup           = excluded.kara_wykup,
            data_aktualizacji    = excluded.data_aktualizacji
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as conn:
        conn.execute(sql, (typ, symbol, data_poczatkowa, data_zakonczenia,
                           dlugosc_lat, oprocentowanie_rok_1, marza, kara, now))
    logger.debug(f"[DB] Upsert emisji: {symbol}")


def get_bond_emissions(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Zwraca wszystkie emisje posortowane wg typu i symbolu."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM emisje_obligacji ORDER BY typ_obligacji, symbol_emisji"
        ).fetchall()
    return [dict(r) for r in rows]


def get_bond_penalties_by_type(db_path: Path = DB_PATH) -> Dict[str, float]:
    """
    Zwraca słownik {typ_obligacji: kara_wykup} z najnowszej emisji każdego typu.

    Używany przez data_fetcher do wzbogacenia parametrów obligacji o dokładną karę
    za przedterminowy wykup pobraną z bazy danych (np. 3.00 PLN / obligację).
    Jeśli baza jest pusta, zwraca wartości domyślne.
    """
    defaults: Dict[str, float] = {"EDO": 3.0, "COI": 1.5}
    try:
        with get_connection(db_path) as conn:
            rows = conn.execute(
                """
                SELECT typ_obligacji, kara_wykup
                FROM emisje_obligacji
                WHERE id IN (
                    SELECT MAX(id) FROM emisje_obligacji
                    GROUP BY typ_obligacji
                )
                """
            ).fetchall()
        if rows:
            result = {r["typ_obligacji"]: r["kara_wykup"] for r in rows}
            # Uzupełnienie brakujących typów wartościami domyślnymi
            for typ, default_kara in defaults.items():
                result.setdefault(typ, default_kara)
            logger.debug(f"[DB] Kary za wykup z bazy: {result}")
            return result
    except Exception as e:
        logger.warning(f"[DB] Nie udało się pobrać kar za wykup z bazy: {e}. Używam wartości domyślnych.")
    return defaults


def delete_bond_emission(symbol: str, db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "DELETE FROM emisje_obligacji WHERE symbol_emisji = ?", (symbol,)
        )


# ─── analizy_historia ────────────────────────────────────────────────────────

def save_analysis(
    nazwa_strategii: str,
    cel_optymalizacji: str,
    horyzont_inwestycyjny_lat: int,
    prowizja_maklerska: Optional[float],
    cvar_alpha: float,
    oczekiwana_stopa: float,
    wartosc_cvar: float,
    koszt_rebalancingu: Optional[float],
    db_path: Path = DB_PATH,
) -> int:
    """Zapisuje jeden rekord analizy ze wszystkimi parametrami. Zwraca jej id."""
    sql = """
        INSERT INTO analizy_historia
            (data_analizy, nazwa_strategii, cel_optymalizacji,
             horyzont_inwestycyjny_lat, prowizja_maklerska,
             cvar_alpha, oczekiwana_stopa_zwrotu, wartosc_ryzyka_cvar, koszt_rebalancingu_netto)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as conn:
        cur = conn.execute(sql, (
            now, nazwa_strategii, cel_optymalizacji,
            horyzont_inwestycyjny_lat, prowizja_maklerska,
            cvar_alpha, oczekiwana_stopa, wartosc_cvar, koszt_rebalancingu,
        ))
        new_id = cur.lastrowid
    logger.debug(f"[DB] Zapisano analizę id={new_id}: {nazwa_strategii}")
    return new_id


def get_analysis_history(
    limit: int = 100,
    db_path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """Zwraca historię analiz posortowaną od najnowszej."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM analizy_historia ORDER BY data_analizy DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_analysis(analiza_id: int, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    """Zwraca jeden rekord analizy lub None."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM analizy_historia WHERE id = ?", (analiza_id,)
        ).fetchone()
    return dict(row) if row else None


# ─── wyniki_wagi_portfela ─────────────────────────────────────────────────────

def save_portfolio_weights(
    analiza_id: int,
    weights: Dict[str, float],
    bond_tickers: set,
    db_path: Path = DB_PATH,
) -> None:
    """Zapisuje wagi portfela powiązane z daną analizą."""
    sql = """
        INSERT INTO wyniki_wagi_portfela
            (analiza_id, ticker_aktywa, klasa_aktywa, waga_docelowa)
        VALUES (?, ?, ?, ?)
    """
    rows = [
        (analiza_id, ticker, "OBLIGACJA" if ticker in bond_tickers else "AKCJA", float(w))
        for ticker, w in weights.items()
    ]
    with get_connection(db_path) as conn:
        conn.executemany(sql, rows)
    logger.debug(f"[DB] Zapisano {len(rows)} wag dla analizy id={analiza_id}")


def get_portfolio_weights(
    analiza_id: int, db_path: Path = DB_PATH
) -> List[Dict[str, Any]]:
    """Zwraca wagi portfela dla danej analizy, malejąco wg wagi."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM wyniki_wagi_portfela WHERE analiza_id = ? ORDER BY waga_docelowa DESC",
            (analiza_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ─── Convenience: zapis pełnego wyniku optymalizacji ─────────────────────────

def save_full_result(
    optimization_result: Dict[str, Any],
    bond_tickers: set,
    nazwa_strategii: str = "optymalizacja",
    cel_optymalizacji: str = "CVaR",
    horyzont_inwestycyjny_lat: int = 4,
    prowizja_maklerska: Optional[float] = None,
    cvar_alpha: float = 0.99,
    db_path: Path = DB_PATH,
) -> int:
    """
    Zapisuje analizę + wagi portfela w jednym wywołaniu.
    Zwraca id nowej analizy.
    """
    import pandas as pd

    metrics = optimization_result.get("metrics", {})
    analiza_id = save_analysis(
        nazwa_strategii=nazwa_strategii,
        cel_optymalizacji=cel_optymalizacji,
        horyzont_inwestycyjny_lat=horyzont_inwestycyjny_lat,
        prowizja_maklerska=prowizja_maklerska,
        cvar_alpha=cvar_alpha,
        oczekiwana_stopa=metrics.get("expected_return", 0.0),
        wartosc_cvar=metrics.get("cvar", metrics.get("volatility", 0.0)),
        koszt_rebalancingu=metrics.get(
            "total_rebalancing_cost",
            metrics.get("estimated_execution_cost", 0.0),
        ),
        db_path=db_path,
    )

    weights = optimization_result.get("weights")
    if weights is not None:
        weights_dict = (
            weights.to_dict() if isinstance(weights, pd.Series) else dict(weights)
        )
        save_portfolio_weights(analiza_id, weights_dict, bond_tickers, db_path)

    logger.info(f"[DB] Wynik analizy id={analiza_id} zapisany: {nazwa_strategii}")
    return analiza_id


# ─── CLI self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    initialize_db()

    # Emisje testowe
    upsert_bond_emission("EDO", "EDO0536", "2025-03-01", "2035-03-01", 10, 0.0535, 0.0200, 3.0)
    upsert_bond_emission("COI", "COI0530", "2025-03-01", "2029-03-01",  4, 0.0520, 0.0150, 2.0)
    print("Emisje:", get_bond_emissions())

    # Analiza testowa
    aid = save_analysis(
        nazwa_strategii="Konserwatywna CVaR 99%",
        cel_optymalizacji="CVaR",
        horyzont_inwestycyjny_lat=4,
        prowizja_maklerska=None,
        cvar_alpha=0.99,
        oczekiwana_stopa=0.12,
        wartosc_cvar=0.08,
        koszt_rebalancingu=None,
    )
    save_portfolio_weights(
        analiza_id=aid,
        weights={"PKN": 0.4, "PKO": 0.35, "EDO0536": 0.25},
        bond_tickers={"EDO0536"},
    )
    print("Historia:", get_analysis_history())
    print("Wagi:", get_portfolio_weights(aid))
    print("OK")
