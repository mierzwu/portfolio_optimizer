# Dokumentacja Techniczna — Portfolio Optimizer

## Spis treści

1. [Przegląd systemu](#1-przegląd-systemu)
2. [Architektura i przepływ danych](#2-architektura-i-przepływ-danych)
3. [Opis plików źródłowych](#3-opis-plików-źródłowych)
   - [models.py](#31-modelspy)
   - [logger_setup.py](#32-logger_setuppy)
   - [data_fetcher.py](#33-data_fetcherpy)
   - [data_processor.py](#34-data_processorpy)
   - [database.py](#35-databasepy)
   - [parameter_estimator.py](#36-parameter_estimatorpy)
   - [optimizer.py](#37-optimizerpy)
   - [backtester.py](#38-backtesterpy)
   - [gui.py](#39-guipy)
4. [Testy jednostkowe](#4-testy-jednostkowe)
5. [Zależności i wymagania](#5-zależności-i-wymagania)

---

## 1. Przegląd systemu

**Portfolio Optimizer** to aplikacja desktopowa do optymalizacji portfela inwestycyjnego, wspierająca zarówno akcje notowane na GPW (Giełdzie Papierów Wartościowych w Warszawie), jak i polskie detaliczne obligacje skarbowe (EDO – 10-letnie emerytalne, COI – 4-letnie oszczędnościowe).

### Główne funkcjonalności:
- Pobieranie danych rynkowych z Yahoo Finance (GPW) oraz parametrów obligacji z bazy SQLite
- Wycena obligacji detalicznych na podstawie mechanizmu CPI + marża (bez DCF)
- Estymacja parametrów modelu (James-Stein shrinkage, Ledoit-Wolf)
- Optymalizacja portfela (minimalizacja CVaR/wariancji lub maksymalizacja zwrotu)
- Symulacja Monte Carlo (GBM) i stress-testy
- Interfejs graficzny PySide6 z wieloekranową nawigacją
- Persystencja wyników w bazie SQLite

---

## 2. Architektura i przepływ danych

System realizuje potok przetwarzania danych w 5 etapach:

```
┌─────────────┐     ┌──────────────┐     ┌────────────────────┐
│ InputData   │────▶│ data_fetcher │────▶│  D1_raw            │
│ (models.py) │     │              │     │  (prices, CPI,     │
└─────────────┘     └──────────────┘     │   bond_params)     │
                                         └────────┬───────────┘
                                                  │
                                                  ▼
┌─────────────────────┐     ┌──────────────────────────────────┐
│ D2_processed        │◀────│ data_processor                   │
│ (prices, returns,   │     │ (obliczanie narosłej wartości     │
│  bond_metadata)     │     │  obligacji, synchronizacja dat)  │
└────────┬────────────┘     └──────────────────────────────────┘
         │
         ▼
┌─────────────────────┐     ┌──────────────────────────────────┐
│ model_params        │◀────│ parameter_estimator              │
│ (mu, sigma,         │     │ (James-Stein, Ledoit-Wolf,       │
│  sigma_shrink,      │     │  prospektywne μ obligacji)       │
│  avg_cpi)           │     └──────────────────────────────────┘
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌──────────────────────────────────┐
│ optimization_result │◀────│ optimizer (CVXPY)                │
│ (weights, metrics,  │     │ (CVaR / Mean-Variance,           │
│  transactions)      │     │  ograniczenia klas aktywów)      │
└────────┬────────────┘     └──────────────────────────────────┘
         │
         ▼
┌─────────────────────┐     ┌──────────────────────────────────┐
│ simulation_result   │◀────│ backtester                       │
│ (paths, VaR, CVaR,  │     │ (Monte Carlo GBM, stress-testy)  │
│  stress_tests)      │     └──────────────────────────────────┘
└─────────────────────┘
```

---

## 3. Opis plików źródłowych

---

### 3.1 `models.py`

**Cel:** Definicja modeli danych wejściowych przy użyciu Pydantic. Zapewnia walidację i typowanie wszystkich parametrów aplikacji.

#### Enumy

| Enum | Wartości | Opis |
|------|----------|------|
| `InstrumentType` | `STOCK`, `BOND` | Typ instrumentu finansowego |
| `BondType` | `EDO`, `COI` | Typ obligacji detalicznej |
| `GoalType` | `MIN_RISK`, `MAX_RETURN` | Cel optymalizacji |

#### Klasy

##### `PortfolioItem(BaseModel)`
Reprezentuje pojedynczą pozycję w portfelu.

| Pole | Typ | Opis |
|------|-----|------|
| `ticker` | `str` | Symbol instrumentu |
| `instrument_type` | `InstrumentType` | Typ: akcja lub obligacja |
| `quantity` | `float` | Liczba posiadanych jednostek |
| `symbol_emisji` | `Optional[str]` | Symbol emisji obligacji (np. `'EDO0135'`) |
| `bond_type` | `Optional[BondType]` | Typ obligacji |
| `margin` | `Optional[float]` | Marża ponad CPI (np. 0.015 = 1.5%) |
| `first_year_rate` | `Optional[float]` | Stałe oprocentowanie w 1. roku |
| `issue_date` | `Optional[date]` | Data emisji obligacji |
| `date_acquired` | `Optional[date]` | Data nabycia |
| `price_acquired` | `Optional[float]` | Cena nabycia |

##### `OptimizationParameters(BaseModel)`
Parametry celu optymalizacji.

| Pole | Typ | Opis |
|------|-----|------|
| `goal_type` | `GoalType` | Cel: minimalizacja ryzyka lub maksymalizacja zwrotu |
| `goal_value` | `Optional[float]` | Wartość docelowa (min. zwrot lub maks. ryzyko) |
| `cvar_alpha` | `Optional[float]` | Poziom ufności CVaR (0–1) |

##### `ConstraintsSettings(BaseModel)`
Ograniczenia portfela.

| Pole | Typ | Domyślnie | Opis |
|------|-----|-----------|------|
| `max_weight` | `float` | — | Maks. waga pojedynczej akcji (0–1) |
| `min_trade_unit` | `float` | — | Minimalna jednostka transakcyjna (PLN) |
| `transaction_cost_pct` | `float` | — | Prowizja maklerska (%) |
| `max_bond_weight` | `float` | 0.60 | Maks. łączny udział obligacji |
| `min_target_return` | `Optional[float]` | `None` | Min. roczna stopa zwrotu portfela |

##### `ExecutionConfig(BaseModel)`
Parametry realizacji zleceń.

| Pole | Typ | Domyślnie | Opis |
|------|-----|-----------|------|
| `spread_pct` | `float` | 0.002 | Bid-Ask spread (0.2%) |
| `slippage_pct` | `float` | 0.001 | Slippage (0.1%) |
| `market_impact_factor` | `float` | 1e-7 | Współczynnik wpływu na rynek |

##### `DataPolicy(BaseModel)`
Polityka przetwarzania danych.

| Pole | Typ | Domyślnie | Opis |
|------|-----|-----------|------|
| `min_observations` | `int` | 100 | Min. liczba obserwacji |
| `use_cache` | `bool` | `True` | Używaj cachowania |
| `cache_dir` | `str` | `".cache"` | Katalog cache |
| `data_version` | `str` | `"v1"` | Wersja danych |

##### `InputData(BaseModel)`
Główny model wejściowy – korzenny obiekt konfiguracji.

| Pole | Typ | Domyślnie | Opis |
|------|-----|-----------|------|
| `portfolio` | `List[PortfolioItem]` | — | Aktualne pozycje portfela |
| `parametry_opt` | `OptimizationParameters` | — | Parametry optymalizacji |
| `estimation_window` | `str` | — | Okno estymacji (np. `"3Y"`) |
| `investment_horizon_years` | `int` | 5 | Horyzont inwestycyjny (1–40 lat) |
| `data_sources` | `List[str]` | `["GPW"]` | Źródła danych |
| `data_policy` | `DataPolicy` | domyślna | Polityka danych |
| `execution_config` | `ExecutionConfig` | domyślna | Konfiguracja realizacji |
| `start_date` | `Optional[date]` | `None` (dziś) | Data analizy |
| `is_planning_phase` | `bool` | `False` | Faza planowania (bez kosztów) |
| `ustawienia_ograniczen` | `ConstraintsSettings` | — | Ograniczenia portfela |
| `additional_cash` | `Optional[float]` | `None` | Dodatkowe środki PLN |

#### Przykład użycia

```python
from models import InputData, PortfolioItem, InstrumentType, GoalType
from models import OptimizationParameters, ConstraintsSettings
from datetime import date

input_data = InputData(
    portfolio=[
        PortfolioItem(ticker="PKN", instrument_type=InstrumentType.STOCK, quantity=50),
        PortfolioItem(ticker="PKO", instrument_type=InstrumentType.STOCK, quantity=100),
        PortfolioItem(
            ticker="EDO0135",
            instrument_type=InstrumentType.BOND,
            quantity=20,
            symbol_emisji="EDO0135",
            bond_type="EDO",
            margin=0.015,
            first_year_rate=0.0535,
            issue_date=date(2025, 1, 1),
        ),
    ],
    parametry_opt=OptimizationParameters(
        goal_type=GoalType.MIN_RISK,
        goal_value=0.05,
        cvar_alpha=0.95,
    ),
    estimation_window="3Y",
    investment_horizon_years=5,
    start_date=date(2025, 6, 1),
    ustawienia_ograniczen=ConstraintsSettings(
        max_weight=0.40,
        min_trade_unit=500.0,
        transaction_cost_pct=0.0039,
        max_bond_weight=0.60,
    ),
)
```

---

### 3.2 `logger_setup.py`

**Cel:** Konfiguracja globalnego loggera z obsługą UTF-8 na Windows.

#### Funkcje

##### `setup_logger() → logging.Logger`

Konfiguruje root logger:
- Poziom: `INFO`
- Format: czysty tekst (`%(message)s`)
- Wymusza kodowanie UTF-8 na Windows (PowerShell domyślnie używa cp1250)
- Czyści istniejące handlery aby uniknąć duplikacji

```python
from logger_setup import setup_logger

logger = setup_logger()
logger.info("Aplikacja uruchomiona")
```

---

### 3.3 `data_fetcher.py`

**Cel:** Warstwa pobierania danych rynkowych z różnych źródeł (Yahoo Finance, SQLite, hardcoded CPI).

#### Funkcje

##### `fetch_gpw_data(tickers: List[str], start_date: date, end_date: date) → pd.DataFrame`

Pobiera dzienne ceny zamknięcia akcji z GPW via Yahoo Finance.

- Automatycznie dodaje sufiks `.WA` dla tickerów GPW
- Obsługuje pojedynczy i wiele tickerów
- Forward-fill brakujących wartości
- Zwraca DataFrame z indeksem DatetimeIndex i kolumnami = tickery

```python
from data_fetcher import fetch_gpw_data
from datetime import date

prices = fetch_gpw_data(
    tickers=["PKN", "PKO", "CDR"],
    start_date=date(2022, 1, 1),
    end_date=date(2025, 1, 1)
)
# prices.shape → (≈750, 3)
```

##### `fetch_bonds_data(tickers: List[str]) → pd.DataFrame`

Pobiera parametry obligacji detalicznych z bazy SQLite (`emisje_obligacji`).

- Dopasowanie po `symbol_emisji`
- Zwraca DataFrame z kolumnami: `bond_type`, `margin`, `first_year_rate`, `issue_date`, `kara_wykup`
- Fallback: pusty DataFrame jeśli brak danych

```python
from data_fetcher import fetch_bonds_data

bond_info = fetch_bonds_data(["EDO0135", "COI0530"])
# bond_info.loc["EDO0135", "margin"] → 0.02
```

##### `fetch_cpi_data(start_year: int, end_year: int) → pd.Series`

Zwraca roczne odczyty CPI dla Polski (źródło: GUS).

- Indeks: rok (int)
- Wartości: CPI jako ułamki (0.035 = 3.5%)
- Dane hardcoded 1996–2025

```python
from data_fetcher import fetch_cpi_data

cpi = fetch_cpi_data(2015, 2025)
# cpi[2022] → 0.144 (14.4% inflacja)
```

##### `parse_estimation_window(window_str: str) → timedelta`

Konwertuje string okna estymacji na `timedelta`.

| Wejście | Wynik |
|---------|-------|
| `"3Y"` | 1095 dni |
| `"6M"` | 180 dni |
| `"30D"` | 30 dni |

##### `get_cache_key(validated_input, history_start_date, history_end_date) → str`

Generuje klucz MD5 cache na podstawie: wersji danych, dat, tickerów, źródeł.

##### `fetch_market_data(validated_input: InputData) → Dict[str, Any]`

**Główna funkcja koordynująca** — orkiestruje pobieranie wszystkich danych.

**Zwraca słownik `D1_raw`:**
```python
{
    "prices": pd.DataFrame,      # dzienne ceny akcji
    "cpi_history": pd.Series,    # roczne CPI (indeks=rok, wartości=ułamkowe)
    "bond_params": pd.DataFrame, # parametry obligacji
    "metadata": {
        "history_start": date,
        "history_end": date,
        "sources_used": List[str]
    }
}
```

**Logika:**
1. Oblicza zakres dat: `[start_date - estimation_window, start_date]`
2. Sprawdza cache (plik pickle)
3. Segreguje instrumenty na akcje i obligacje
4. Pobiera ceny akcji (`fetch_gpw_data`)
5. Pobiera parametry obligacji (`fetch_bonds_data`) z fallbackiem na dane z `InputData`
6. Pobiera historię CPI (`fetch_cpi_data`)
7. Zapisuje wynik do cache

---

### 3.4 `data_processor.py`

**Cel:** Przetwarzanie surowych danych — obliczanie wartości narosłych obligacji, synchronizacja szeregów czasowych, obliczanie zwrotów.

#### Funkcje

##### `calculate_retail_bond_accrual(cpi_history, margin, first_year_rate, holding_years, bond_type, face_value) → float`

Oblicza skumulowaną wartość obligacji detalicznej po `holding_years` latach.

**Zasada oprocentowania:**
- Rok 1: stopa = `first_year_rate` (stała)
- Rok n ≥ 2: stopa = `CPI[n-2]` + `margin`

**EDO (kapitalizacja roczna):**
```
V(n) = face_value × ∏(1 + r_i) dla i = 1..n
```

**COI (kupon roczny):**
```
V(n) = face_value + Σ(face_value × r_i) dla i = 1..n
```

```python
from data_processor import calculate_retail_bond_accrual

# EDO po 3 latach, CPI = [3.5%, 5.1%], marża 1.5%, oprocentowanie 1. roku 6.5%
value = calculate_retail_bond_accrual(
    cpi_history=[0.035, 0.051],
    margin=0.015,
    first_year_rate=0.065,
    holding_years=3,
    bond_type='EDO',
    face_value=100.0
)
# value ≈ 100 × 1.065 × 1.05 × 1.066 ≈ 119.14
```

##### `_build_bond_accrual_series(valuation_dates, issue_date, bond_type, margin, first_year_rate, cpi_annual, face_value) → pd.Series`

Tworzy **dzienny szereg czasowy** wartości narosłej obligacji.

- Wartość w dniu `t` uwzględnia pełne lata + narastanie liniowe/ciągłe w bieżącym roku
- EDO: kapitalizacja ciągła wewnątrz roku: `V_full × (1+r)^(days/365)`
- COI: narastanie liniowe: `V_full + face × r × (days/365)`

##### `preprocess_data(D1_raw: Dict, policy: DataPolicy) → Dict[str, Any]`

**Główna funkcja przetwarzania** — potok 5 kroków:

1. **Obliczanie wartości obligacji** — buduje dzienny szereg cenowy dla każdej obligacji
2. **Łączenie** — outer join cen akcji i obligacji
3. **Imputacja** — forward-fill + backward-fill + usuwanie kolumn z < `min_observations`
4. **Obliczanie zwrotów** — `pct_change()` (arytmetyczne)
5. **Kontrola jakości** — wykrywanie outlierów (zwrot > 50% dziennie)

**Zwraca słownik `D2_processed`:**
```python
{
    "prices": pd.DataFrame,       # połączone, oczyszczone ceny
    "returns": pd.DataFrame,      # arytmetyczne zwroty dzienne
    "bond_metadata": {
        "bond_params": pd.DataFrame
    },
    "cpi_history": pd.Series
}
```

---

### 3.5 `database.py`

**Cel:** Warstwa dostępu do bazy danych SQLite — zarządzanie emisjami obligacji, historią analiz i wagami portfeli.

#### Schemat bazy danych

```sql
-- Tabela 1: Parametry emisji obligacji
CREATE TABLE emisje_obligacji (
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
);

-- Tabela 2: Historia analiz
CREATE TABLE analizy_historia (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    data_analizy              TEXT NOT NULL,
    nazwa_strategii           TEXT NOT NULL DEFAULT 'optymalizacja',
    cel_optymalizacji         TEXT NOT NULL DEFAULT 'min_risk',
    horyzont_inwestycyjny_lat INTEGER NOT NULL DEFAULT 4,
    prowizja_maklerska        REAL,
    cvar_alpha                REAL NOT NULL,
    oczekiwana_stopa_zwrotu   REAL NOT NULL,
    wartosc_ryzyka_cvar       REAL NOT NULL,
    koszt_rebalancingu_netto  REAL
);

-- Tabela 3: Docelowe wagi portfela (relacja 1:N z analizy_historia)
CREATE TABLE wyniki_wagi_portfela (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    analiza_id    INTEGER NOT NULL,
    ticker_aktywa TEXT NOT NULL,
    klasa_aktywa  TEXT NOT NULL CHECK(klasa_aktywa IN ('AKCJA', 'OBLIGACJA')),
    waga_docelowa REAL NOT NULL,
    FOREIGN KEY (analiza_id) REFERENCES analizy_historia(id) ON DELETE CASCADE
);
```

#### Funkcje

##### `get_connection(db_path: Path) → Generator[sqlite3.Connection]`

Context manager dla połączeń z bazą. Włącza klucze obce, auto-commit/rollback.

```python
from database import get_connection

with get_connection() as conn:
    rows = conn.execute("SELECT * FROM emisje_obligacji").fetchall()
```

##### `initialize_db(db_path: Path) → None`

Tworzy wszystkie tabele (idempotentne — kolejne wywołania to no-op).

##### `upsert_bond_emission(typ, symbol, data_poczatkowa, data_zakonczenia, dlugosc_lat, oprocentowanie_rok_1, marza, kara, db_path) → None`

Wstawia nową emisję lub aktualizuje istniejącą (UPSERT na `symbol_emisji`).

```python
from database import upsert_bond_emission, initialize_db

initialize_db()
upsert_bond_emission(
    typ="EDO",
    symbol="EDO0536",
    data_poczatkowa="2025-03-01",
    data_zakonczenia="2035-03-01",
    dlugosc_lat=10,
    oprocentowanie_rok_1=0.0535,
    marza=0.0200,
    kara=3.0
)
```

##### `get_bond_emissions(db_path) → List[Dict]`

Zwraca wszystkie emisje posortowane wg typu i symbolu.

##### `get_bond_penalties_by_type(db_path) → Dict[str, float]`

Zwraca `{typ_obligacji: kara_wykup}` z najnowszej emisji każdego typu.
Domyślne wartości: `{"EDO": 3.0, "COI": 1.5}`.

##### `save_analysis(...) → int`

Zapisuje rekord analizy. Zwraca `id` nowej analizy.

##### `get_analysis_history(limit, db_path) → List[Dict]`

Zwraca historię analiz (od najnowszej, limit=100).

##### `save_portfolio_weights(analiza_id, weights, bond_tickers, db_path) → None`

Zapisuje wagi portfela dla danej analizy. Automatycznie klasyfikuje aktywa (`AKCJA`/`OBLIGACJA`).

##### `get_portfolio_weights(analiza_id, db_path) → List[Dict]`

Zwraca wagi portfela dla analizy, malejąco wg wagi.

##### `save_full_result(optimization_result, bond_tickers, ...) → int`

Convenience function — zapisuje analizę + wagi w jednym wywołaniu.

```python
from database import save_full_result

analysis_id = save_full_result(
    optimization_result=result,
    bond_tickers={"EDO0536"},
    nazwa_strategii="Konserwatywna CVaR 95%",
    cel_optymalizacji="min_risk",
    horyzont_inwestycyjny_lat=5,
    cvar_alpha=0.95
)
```

---

### 3.6 `parameter_estimator.py`

**Cel:** Estymacja parametrów modelu — oczekiwane zwroty (μ) i macierz kowariancji (Σ) z zastosowaniem metod shrinkage.

#### Funkcje

##### `estimate_avg_cpi(cpi_history: pd.Series) → float`

Wyznacza średnią historyczną CPI.
- Fallback: 2.5% jeśli brak danych.

##### `calculate_shrinkage_mu(returns: pd.DataFrame) → pd.Series`

Implementuje **James-Stein Shrinkage** dla oczekiwanych zwrotów.

**Algorytm:**
1. Oblicza historyczną średnią dzienną (`μ_hist`)
2. Wyznacza target = grand mean (średnia ze wszystkich aktywów)
3. Oblicza λ = Noise / (Noise + Signal)
   - Noise = średnia wariancja estymacji (σ²/T)
   - Signal = wariancja cross-sectional (między średnimi aktywów)
4. Shrinkage: `μ_shrink = (1-λ)·μ_hist + λ·μ_target`

```python
from parameter_estimator import calculate_shrinkage_mu

# returns: pd.DataFrame (252 × N), dzienne zwroty
mu_daily = calculate_shrinkage_mu(returns)
mu_annual = mu_daily * 252
```

##### `compute_bond_forward_mu(bond_params_df, current_cpi, investment_horizon_years, today) → pd.Series`

Oblicza **prospektywny** roczny oczekiwany zwrot dla obligacji detalicznych.

**Wzór:**
- Rok 1: `first_year_rate`
- Lata 2..H: `current_cpi + margin`
- Skumulowana wartość: `(1 + r₁) × (1 + r_recurring)^(H-1)`
- Kara za przedterminowy wykup (jeśli H < 10 lat)
- Efektywna stopa roczna: `V(H)^(1/H) - 1`

```python
from parameter_estimator import compute_bond_forward_mu

bond_mu = compute_bond_forward_mu(
    bond_params_df=bond_params,
    current_cpi=0.036,
    investment_horizon_years=5,
)
# bond_mu["EDO0135"] ≈ 0.049 (4.9% rocznie)
```

##### `estimate_params(D2_processed, estimation_window, investment_horizon_years) → Dict[str, Any]`

**Główna funkcja estymacji** — potok parametrów:

1. **μ (James-Stein)** — shrinkage dziennych zwrotów, annualizacja ×252
2. **Σ empiryczna** — macierz kowariancji annualizowana (×252)
3. **Σ Ledoit-Wolf** — shrinkage macierzy kowariancji (sklearn)
4. **Walidacja** — condition number (ostrzeżenie jeśli > 1000)
5. **CPI** — średnia historyczna + bieżący odczyt
6. **Override μ obligacji** — zastąpienie μ historycznego prospektywnym forward estimate

**Zwraca:**
```python
{
    "mu": pd.Series,           # oczekiwane roczne zwroty (annualizowane)
    "sigma": pd.DataFrame,     # empiryczna macierz kowariancji (roczna)
    "sigma_shrink": pd.DataFrame,  # Ledoit-Wolf (roczna)
    "last_date": pd.Timestamp, # ostatnia data w danych
    "avg_cpi": float,          # średnia historyczna CPI
    "current_cpi": float,      # ostatni odczyt CPI
}
```

---

### 3.7 `optimizer.py`

**Cel:** Silnik optymalizacji portfela oparty o CVXPY z obsługą CVaR (Rockafellar & Uryasev) i Mean-Variance.

#### Funkcja główna

##### `optimize_portfolio(validated_input, model_params, processed_data, current_holdings) → Dict[str, Any]`

**Problem optymalizacji:**

```
MIN_RISK:   minimize CVaR(w) + 0.05·TC(w)
            subject to: Σw = 1, w ≥ 0, w_i ≤ max_weight,
                        Σw_bonds ≤ max_bond_weight,
                        μᵀw ≥ goal_value (opcjonalnie)

MAX_RETURN: maximize μᵀw - 0.05·TC(w)
            subject to: Σw = 1, w ≥ 0, w_i ≤ max_weight,
                        Σw_bonds ≤ max_bond_weight,
                        CVaR(w) ≤ goal_value (opcjonalnie)
```

**Model kosztów transakcyjnych:**

| Klasa | Koszt |
|-------|-------|
| **Akcje** | liniowy: `(spread/2 + slippage + prowizja) × |ΔV|` + kwadratowy: `impact × ΔV²` |
| **Obligacje** | kara za wykup: `kara_PLN × jednostki_sprzedane` (tylko przy redukcji) |

**Mechanizm Fallback:**
1. Próba rozwiązania z SCS (max_iters=50000)
2. Fallback do CLARABEL jeśli SCS zawodzi
3. Relaksacja: poluzowanie `max_weight` do 1.0
4. Relaksacja: usunięcie ograniczeń celu (Target Return/Risk)
5. Ostateczny fallback: zwraca obecne wagi portfela

**Post-processing:**
- Zaokrąglanie do pełnych jednostek (min trade unit)
- Generowanie listy transakcji (BUY/SELL)
- Szacowanie kosztów realizacji

**Zwraca:**
```python
{
    "weights": pd.Series,            # optymalne wagi (indeks = ticker)
    "target_quantities": Dict[str, float],  # docelowe ilości jednostek
    "metrics": {
        "budget": float,             # wartość portfela (PLN)
        "total_value": float,        # wartość po rebalancingu
        "cash_remainder": float,     # niewykorzystana gotówka
        "expected_return": float,    # oczekiwany roczny zwrot
        "volatility": float,         # roczna zmienność
        "sharpe_ratio": float,       # Sharpe (risk-free = CPI)
        "cost_impact_bps": float,    # koszt w punktach bazowych
        "total_rebalancing_cost": float,  # łączny koszt rebalancingu (PLN)
        "stock_cap_effective": float,
        "bond_cap_effective": float,
        "caps_auto_adjusted": bool,  # czy limity zostały automatycznie skorygowane
    },
    "transactions": [
        {
            "ticker": str,
            "action": "BUY" | "SELL",
            "quantity": float,
            "price_est": float,
            "est_cost": float,
        }
    ]
}
```

#### Przykład użycia

```python
from optimizer import optimize_portfolio

result = optimize_portfolio(
    validated_input=input_data,
    model_params=model_params,
    processed_data=processed_data,
    current_holdings={"PKN": 50, "PKO": 100, "EDO0135": 20}
)

print(f"Oczekiwany zwrot: {result['metrics']['expected_return']:.2%}")
print(f"Zmienność: {result['metrics']['volatility']:.2%}")
print(f"Sharpe: {result['metrics']['sharpe_ratio']:.3f}")
print(f"Koszt rebalancingu: {result['metrics']['total_rebalancing_cost']:.2f} PLN")

for tx in result['transactions']:
    print(f"  {tx['action']} {tx['quantity']} × {tx['ticker']} @ {tx['price_est']:.2f}")
```

---

### 3.8 `backtester.py`

**Cel:** Symulacja Monte Carlo (Geometric Brownian Motion) i stress-testy portfela pod zszokowanymi parametrami.

#### Funkcje

##### `run_monte_carlo_simulation(model_params, weights, processed_data, n_simulations, time_horizon_days, cvar_alpha) → Dict[str, Any]`

Symulacja Monte Carlo oparta o GBM (Geometryczny Ruch Browna).

**Algorytm:**
1. Skalowanie parametrów rocznych na dzienne: `μ_daily = μ/252`, `Σ_daily = Σ/252`
2. Stabilizacja PSD (wartości własne ≥ 1e-12)
3. GBM z korektą Itô: `log_r ~ MVN(μ - 0.5σ², Σ)`
4. Zwroty arytmetyczne: `r = exp(log_r) - 1`
5. Zwrot portfela: `r_port = wᵀ · r`
6. Kumulacja: `V(t) = ∏(1 + r_port_t)`
7. VaR/CVaR z rozkładu końcowych wartości portfela

**Metryki:**
- `expected_return` — roczny arytmetyczny oczekiwany zwrot (mean × 252)
- `VaR_α%` — Value at Risk na poziomie α (annualizowana strata)
- `CVaR_α%` — Conditional VaR (średnia strata w ogonie ≤ VaR)

**Konwencja:** VaR/CVaR jako dodatnie liczby (wielkość straty).

```python
from backtester import run_monte_carlo_simulation

mc = run_monte_carlo_simulation(
    model_params=model_params,
    weights=result['weights'],
    n_simulations=2000,
    time_horizon_days=5*252,  # 5 lat
    cvar_alpha=0.95
)

print(f"MC Oczekiwany zwrot: {mc['metrics']['expected_return']:.2%}")
print(f"VaR 95%: {mc['metrics']['VaR_95']:.2%}")
print(f"CVaR 95%: {mc['metrics']['CVaR_95']:.2%}")
print(f"Kształt ścieżek: {mc['paths'].shape}")  # (2000, 1260)
```

##### `run_stress_test(validated_input, processed_data, base_model_params, optimal_weights) → Dict[str, Dict]`

Stress-testy na **stałych wagach** — ocena portfela pod zszokowanymi parametrami (bez ponownej optymalizacji).

**Scenariusze:**

| Scenariusz | Opis szoku |
|------------|-----------|
| `Base` | Brak zmian (referencja) |
| `CPI +5%` | Skok inflacji +5 p.p.: obligacje indeksowane ↑, inne ↓ (−0.5×shock) |
| `Stocks -20%` | Spadek cen akcji o 20%, wzrost zmienności ×1.5 |
| `High Volatility (x2)` | Podwojenie macierzy kowariancji |

**Zwraca:**
```python
{
    "Base": {"weights": pd.Series, "metrics": {"expected_return", "volatility", "sharpe_ratio"}},
    "CPI +5%": {...},
    "Stocks -20%": {...},
    "High Volatility (x2)": {...},
}
```

##### `backtest_and_simulate(validated_input, processed_data, optimization_result, model_params) → Dict[str, Any]`

Orkiestruje symulację Monte Carlo + stress-testy.

```python
from backtester import backtest_and_simulate

sim_results = backtest_and_simulate(
    validated_input=input_data,
    processed_data=processed_data,
    optimization_result=result,
    model_params=model_params
)

# sim_results["monte_carlo"]["metrics"]
# sim_results["stress_tests"]["Stocks -20%"]["metrics"]["expected_return"]
```

---

### 3.9 `gui.py`

**Cel:** Interfejs graficzny oparty na PySide6 (Qt) — wieloekranowa aplikacja desktopowa do zarządzania portfelem i wizualizacji wyników.

#### Klasy pomocnicze

##### `_LogEmitter(QObject)`
Most sygnałowy Qt do przekazywania logów do GUI (thread-safe).

| Sygnał | Typ | Opis |
|--------|-----|------|
| `message` | `Signal(str)` | Emituje tekst logu |

##### `_QTextEditHandler(logging.Handler)`
Handler loggera Pythona → QTextEdit (wyświetlanie logów w GUI).

##### `_HelpIcon(QLabel)`
Interaktywna ikona pomocy z tooltipem wyświetlanym przy najechaniu myszą.

#### Klasy wątków tła

##### `OptimizationWorker(QObject)`
Wykonuje pełny potok optymalizacji w wątku tła.

| Sygnał | Typ | Opis |
|--------|-----|------|
| `progress` | `Signal(str)` | Postęp (tekst statusu) |
| `result_ready` | `Signal(dict)` | Wynik optymalizacji |
| `error_occurred` | `Signal(str)` | Komunikat błędu |
| `finished` | `Signal()` | Zakończenie pracy |

**Pipeline wewnętrzny:**
1. `fetch_market_data()` → D1_raw
2. `preprocess_data()` → D2_processed
3. `estimate_params()` → model_params
4. `optimize_portfolio()` → optimization_result
5. `backtest_and_simulate()` → simulation_result
6. `save_full_result()` → zapis do bazy

##### `CurrentReturnThread(QThread)`
Oblicza oczekiwany zwrot bieżącego portfela w tle (nieblokujące).

##### `_PriceFetchThread(QThread)`
Pobiera cenę akcji z Yahoo Finance dla zadanej daty.

#### Dialogi

##### `BondEmissionDialog(QDialog)`
Formularz dodawania/edycji emisji obligacji z walidacją pól.

##### `AddStockDialog(QDialog)`
Dialog dodawania pozycji akcji z automatycznym pobieraniem ceny.

##### `SelectBondDialog(QDialog)`
Dialog wyboru emisji obligacji z listy w bazie danych.

#### Ekrany główne

##### `BondsScreen(QWidget)`
Zarządzanie emisjami obligacji w bazie — tabela + operacje CRUD.

##### `HistoryScreen(QWidget)`
Przeglądanie historii analiz — tabela z możliwością podglądu wag i usuwania.

#### Główne okno aplikacji

Wieloekranowa nawigacja (`QStackedWidget`) z ekranami:
- **Portfel** — zarządzanie pozycjami, konfiguracja optymalizacji
- **Obligacje** — zarządzanie emisjami w bazie
- **Historia** — przeglądanie wyników analiz
- **Wyniki** — wizualizacja optymalnego portfela, stress-testów, Monte Carlo

---

## 4. Testy jednostkowe

Testy znajdują się w katalogu `tests/` i używają frameworka `pytest`.

### Pliki testowe

| Plik | Liczba testów | Testowany moduł |
|------|---------------|-----------------|
| `test_optimizer.py` | 14 | `optimizer.py` |
| `test_backtester.py` | 16 | `backtester.py` |
| `test_data_processor.py` | 20 | `data_processor.py` |
| `test_parameter_estimator.py` | 13 | `parameter_estimator.py` |

### Fixture'y (`conftest.py`)

| Fixture | Opis |
|---------|------|
| `prices_2stocks` | 252 dni cen dla STOCK_A, STOCK_B |
| `cpi_series` | 5-letnia historia CPI |
| `bond_params_df` | Parametry obligacji EDO |
| `processed_data_stocks` | Przetworzone dane (tylko akcje) |
| `processed_data_with_bond` | Przetworzone dane z obligacjami |
| `det_model_params` | Deterministyczne parametry modelu (μ=[10%, 8%]) |
| `input_min_risk` | InputData dla celu MIN_RISK |
| `input_max_return` | InputData dla celu MAX_RETURN |

### Kluczowe asercje testowe

**Optimizer:**
- Wagi sumują się do 1 (±tolerancja)
- Long-only: w ≥ 0
- Min-risk < equal-weight variance
- Max-return > min-risk return
- `max_bond_weight` respektowane (±3%)

**Backtester:**
- `paths.shape == (n_simulations, time_horizon_days)`
- `CVaR ≥ VaR` (zawsze)
- Ścieżki startują blisko 1.0
- Deterministyczność z ustalonym seedem

**Data Processor:**
- Zwroty arytmetyczne = `pct_change()`
- `len(returns) == len(prices) - 1`
- Brak NaN po imputacji
- Odrzucanie kolumn z < min_observations

**Parameter Estimator:**
- λ ∈ [0, 1]
- `sigma_shrink` jest PSD (positive semi-definite)
- `μ_annual = μ_daily × 252`

### Uruchamianie testów

```bash
cd licencjat
pytest tests/ -v
```

---

## 5. Zależności i wymagania

### Biblioteki Python

| Pakiet | Zastosowanie |
|--------|-------------|
| `pydantic` | Walidacja danych wejściowych (modele) |
| `pandas` | Przetwarzanie szeregów czasowych |
| `numpy` | Obliczenia numeryczne |
| `yfinance` | Pobieranie danych z Yahoo Finance |
| `cvxpy` | Formułowanie i rozwiązywanie problemów optymalizacji |
| `scikit-learn` | Ledoit-Wolf shrinkage (macierz kowariancji) |
| `PySide6` | Interfejs graficzny (Qt) |
| `pytest` | Framework testowy |

### Solvery CVXPY

| Solver | Zastosowanie |
|--------|-------------|
| **SCS** | Preferowany (LP-friendly, dobry dla CVaR) |
| **CLARABEL** | Fallback jeśli SCS zawodzi |

### Struktura plików

```
licencjat/
├── models.py                # Modele danych (Pydantic)
├── logger_setup.py          # Konfiguracja logowania
├── data_fetcher.py          # Pobieranie danych rynkowych
├── data_processor.py        # Przetwarzanie danych
├── database.py              # Warstwa SQLite
├── parameter_estimator.py   # Estymacja parametrów
├── optimizer.py             # Silnik optymalizacji (CVXPY)
├── backtester.py            # Monte Carlo i stress-testy
├── gui.py                   # Interfejs graficzny (PySide6)
├── database.db              # Plik bazy danych SQLite
├── .cache/                  # Cache danych rynkowych (pickle)
├── README.md                # Opis projektu
└── tests/
    ├── conftest.py          # Fixture'y testowe
    ├── test_optimizer.py    # Testy optymalizatora
    ├── test_backtester.py   # Testy symulacji
    ├── test_data_processor.py   # Testy przetwarzania
    └── test_parameter_estimator.py  # Testy estymacji
```
