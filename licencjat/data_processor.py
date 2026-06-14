import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
from models import DataPolicy

logger = logging.getLogger(__name__)

def calculate_retail_bond_accrual(
    cpi_history: List[float],
    margin: float,
    first_year_rate: float,
    holding_years: int,
    bond_type: str = 'EDO',
    face_value: float = 100.0
) -> float:
    """
    Oblicza skumulowaną wartość detalicznej obligacji skarbowej (EDO/COI).

    Obligacje detaliczne MF nie są notowane na giełdzie – cena nominalna to zawsze
    100 PLN. Zysk wynika wyłącznie z kapitalizacji odsetek (EDO) lub rocznej
    wypłaty kuponu (COI). Brak wyceny DCF, Duration ani Convexity.

    Zasada oprocentowania:
      - Rok 1: stopa = first_year_rate (stała)
      - Rok n (n >= 2): stopa = CPI[n-2] + margin

    EDO: kapitalizacja roczna (procent składany) – wartość rośnie co roku.
    COI: roczna wypłata kuponu – funkcja zwraca nominał + sumę wypłaconych kuponów.

    :param cpi_history: Lista rocznych odczytów CPI (ułamkowych, np. 0.035 = 3.5%).
                        cpi_history[k] to CPI stosowane przy naliczaniu odsetek
                        za rok k+2 (k=0 → rok 2, k=1 → rok 3, ...).
    :param margin: Stała marża ponad CPI (np. 0.015 = 1.5%).
    :param first_year_rate: Stałe oprocentowanie w pierwszym roku.
    :param holding_years: Liczba pełnych lat przetrzymania (>= 1).
    :param bond_type: 'EDO' – kapitalizacja; 'COI' – kupon roczny.
    :param face_value: Wartość nominalna (domyślnie 100 PLN).
    :return: Wartość obligacji po holding_years latach.
    """
    if holding_years <= 0:
        return face_value

    # Wyznaczanie stóp dla każdego roku trzymania
    rates: List[float] = [first_year_rate]
    for year_idx in range(1, holding_years):
        cpi_idx = year_idx - 1  # CPI stosowany od roku 2 (indeks 0)
        if cpi_idx < len(cpi_history):
            cpi_val = cpi_history[cpi_idx]
        else:
            # Brak danych CPI – użyj ostatniego dostępnego odczytu
            cpi_val = cpi_history[-1] if cpi_history else 0.0
        rates.append(cpi_val + margin)

    if bond_type == 'EDO':
        # Kapitalizacja roczna – procent składany
        accrued = face_value
        for r in rates:
            accrued *= (1.0 + r)
        return accrued
    else:
        # COI – roczna wypłata kuponu; nominał zwracany przy wykupie
        total_coupons = sum(face_value * r for r in rates)
        return face_value + total_coupons


def _build_bond_accrual_series(
    valuation_dates: pd.DatetimeIndex,
    issue_date: pd.Timestamp,
    bond_type: str,
    margin: float,
    first_year_rate: float,
    cpi_annual: pd.Series,
    face_value: float = 100.0
) -> pd.Series:
    """
    Tworzy dzienny szereg czasowy wartości narosłej obligacji detalicznej.

    Wartość w dniu t wyznaczana jest metodą dziennej kapitalizacji ciągłej
    wewnątrz każdego rocznego okresu naliczania, zgodnie z ustaloną stopą
    dla danego roku.

    :param valuation_dates: Indeks dat (daily).
    :param issue_date: Data emisji / nabycia obligacji.
    :param bond_type: 'EDO' lub 'COI'.
    :param margin: Marża ponad CPI.
    :param first_year_rate: Oprocentowanie w roku 1.
    :param cpi_annual: Szereg rocznych odczytów CPI (indeksowany datami rocznych
                       opublikowań lub liczbami 0,1,2,...).
    :param face_value: Wartość nominalna (100 PLN).
    :return: pd.Series wartości obligacji dla każdej daty z valuation_dates.
    """
    cpi_list: List[float] = cpi_annual.values.tolist() if not cpi_annual.empty else []

    accrued_values: List[float] = []
    for val_date in valuation_dates:
        days_held = (val_date - issue_date).days
        if days_held < 0:
            accrued_values.append(face_value)
            continue

        full_years = days_held // 365
        partial_days = days_held % 365

        # Wartość po pełnych latach
        full_year_value = calculate_retail_bond_accrual(
            cpi_list, margin, first_year_rate, full_years, bond_type, face_value
        )

        # Stopa bieżącego (niepełnego) roku
        if full_years == 0:
            current_rate = first_year_rate
        else:
            cpi_idx = full_years - 1
            cpi_val = cpi_list[cpi_idx] if cpi_idx < len(cpi_list) else (cpi_list[-1] if cpi_list else 0.0)
            current_rate = cpi_val + margin

        # Narastanie w ciągu bieżącego roku (liniowe uproszczenie, zgodne z MF)
        if bond_type == 'EDO':
            # Kapitalizacja ciągła w obrębie roku
            daily_factor = (1.0 + current_rate) ** (partial_days / 365.0)
            accrued_values.append(full_year_value * daily_factor)
        else:
            # COI – odsetki narastają liniowo, wypłata na koniec roku
            accrued_values.append(full_year_value + face_value * current_rate * (partial_days / 365.0))

    return pd.Series(accrued_values, index=valuation_dates)

def preprocess_data(D1_raw: Dict[str, Any], policy: DataPolicy = DataPolicy()) -> Dict[str, Any]:
    """
    Przetwarza surowe dane rynkowe.

    1. Oblicza wartości narosłe obligacji detalicznych (EDO/COI) na podstawie
       historii CPI + marża (zamiast DCF/YTM).
    2. Łączy ceny akcji i wartości obligacji.
    3. Synchronizuje daty.
    4. Oblicza log-zwroty.
    5. Sprawdza anomalie.

    Oczekiwana struktura D1_raw:
      - 'prices'      : pd.DataFrame  – dzienne ceny akcji (kolumny = tickery)
      - 'cpi_history' : pd.Series     – roczne odczyty CPI (indeks = daty lub int 0,1,2,...),
                                        wartości jako ułamki (np. 0.035 = 3.5%)
      - 'bond_params' : pd.DataFrame  – parametry obligacji detalicznych;
                                        indeks = symbol_emisji, kolumny:
                                        'bond_type' ('EDO'|'COI'), 'margin',
                                        'first_year_rate', 'issue_date'
    """
    logger.debug("[Preprocessing] Rozpoczynanie przetwarzania danych...")

    prices_stocks = D1_raw.get('prices', pd.DataFrame()).copy()
    cpi_history: pd.Series = D1_raw.get('cpi_history', pd.Series(dtype=float))
    bond_params_df: pd.DataFrame = D1_raw.get('bond_params', pd.DataFrame())

    # 1. Obliczanie wartości narosłych obligacji detalicznych (CPI + marża)
    bond_prices_dict: Dict[str, pd.Series] = {}

    if not bond_params_df.empty:
        logger.debug("[Preprocessing] Obliczanie wartości narosłych obligacji detalicznych...")

        # Indeks dat wyceny – bazujemy na cenach akcji (lub generujemy z metadanych)
        if not prices_stocks.empty:
            valuation_dates = prices_stocks.index
        else:
            meta = D1_raw.get('metadata', {})
            hist_start = meta.get('history_start')
            hist_end   = meta.get('history_end')
            if hist_start is not None and hist_end is not None:
                valuation_dates = pd.bdate_range(start=hist_start, end=hist_end)
                logger.debug(
                    f"[Preprocessing] Portfel bez akcji – używam {len(valuation_dates)} "
                    f"dni roboczych ({hist_start} … {hist_end}) jako dat wyceny obligacji."
                )
            else:
                valuation_dates = pd.DatetimeIndex([])

        for symbol_emisji in bond_params_df.index:
            try:
                row = bond_params_df.loc[symbol_emisji]
                bond_type = str(row.get('bond_type', 'EDO'))
                margin = float(row.get('margin', 0.0))
                first_year_rate = float(row.get('first_year_rate', 0.0))
                issue_date = pd.to_datetime(row.get('issue_date', valuation_dates[0] if len(valuation_dates) > 0 else pd.Timestamp.today()))

                series = _build_bond_accrual_series(
                    valuation_dates,
                    issue_date,
                    bond_type,
                    margin,
                    first_year_rate,
                    cpi_history,
                )
                bond_prices_dict[symbol_emisji] = series

            except Exception as e:
                logger.error(f"[Preprocessing] Błąd obliczania narosłej wartości dla {symbol_emisji}: {e}")

    bond_prices = pd.DataFrame(bond_prices_dict)
    
    # 2. Łączenie i synchronizacja
    # Łączymy akcje i obligacje (outer join, potem przycinamy do wspólnego okresu lub ffill)
    all_prices = prices_stocks.join(bond_prices, how='outer', rsuffix='_bond')
    
    # Sortowanie indeksu
    all_prices = all_prices.sort_index()
    
    # 3. Imputacja i filtrowanie wg polityki
    logger.debug(f"[Preprocessing] Stosowanie polityki danych: MinObs={policy.min_observations}")

    # Imputacja (ffill + bfill)
    all_prices = all_prices.ffill().bfill()
        
    # Min observations check
    valid_columns = []
    for col in all_prices.columns:
        valid_count = all_prices[col].count()
        if valid_count >= policy.min_observations:
            valid_columns.append(col)
        else:
            logger.warning(f"[Preprocessing] Odrzucono {col}: zbyt mało obserwacji ({valid_count} < {policy.min_observations})")
            
    all_prices = all_prices[valid_columns]
    
    # Usuwanie wierszy, które nadal mają NaN (jeśli jakieś instrumenty nie mają w ogóle danych)
    all_prices = all_prices.dropna(axis=1, how='all') # Usuń kolumny puste
    all_prices = all_prices.dropna(axis=0, how='any') # Usuń wiersze z brakami (po ffill/bfill)
    
    # Zabezpieczenie przed zerami w cenach (dzielenie przez zero przy pct_change)
    if (all_prices <= 0).any().any():
        logger.warning("[Preprocessing] Znaleziono ceny <= 0. Usuwanie tych wierszy.")
        all_prices = all_prices[all_prices > 0].dropna()

    # 4. Obliczanie prostych zwrotów arytmetycznych
    # R_t = (P_t - P_{t-1}) / P_{t-1}
    returns = all_prices.pct_change()
    returns = returns.dropna()  # Pierwszy wiersz będzie NaN

    # 5. Kontrole i anomalie
    logger.debug("[Preprocessing] Kontrola jakości danych...")

    # Sprawdzenie brakujących serii
    expected_tickers = list(prices_stocks.columns) + list(bond_prices.columns)
    missing_tickers = set(expected_tickers) - set(all_prices.columns)
    if missing_tickers:
        logger.warning(f"[Preprocessing] Ostrzeżenie: Usunięto instrumenty z powodu braku danych: {missing_tickers}")

    # Wykrywanie outlierów (zwrot > 50% dziennie)
    threshold = 0.5
    outliers = (returns.abs() > threshold).sum()
    if outliers.sum() > 0:
        logger.warning("[Preprocessing] Wykryto potencjalne anomalie (duże zmiany cen):")
        logger.warning(outliers[outliers > 0])

    logger.debug(f"[Preprocessing] Przetworzono {len(returns)} obserwacji dla {len(returns.columns)} instrumentów.")

    D2_processed = {
        "prices": all_prices,
        "returns": returns,
        "bond_metadata": {
            "bond_params": bond_params_df,
        },
        "cpi_history": cpi_history  # Przydatne do kalibracji modelu CPI w estimate_params
    }
    
    return D2_processed

if __name__ == "__main__":
    # Test
    # Symulujemy D1_raw
    dates = pd.date_range("2023-01-01", "2023-01-10")
    prices_mock = pd.DataFrame({'PKN': [100, 101, 102, 101, 103, 104, 105, 104, 106, 107]}, index=dates)
    ytm_mock = pd.DataFrame({'risk_free_rate': [0.05]*10}, index=dates)
    coupons_mock = pd.DataFrame({'coupon': [4.0]}, index=['BOND1'])
    maturity_mock = pd.DataFrame({'maturity': [pd.Timestamp("2024-01-01")]}, index=['BOND1'])
    face_mock = pd.DataFrame({'face_value': [1000.0]}, index=['BOND1'])
    
    D1 = {
        "prices": prices_mock,
        "ytm": ytm_mock,
        "coupons": coupons_mock,
        "maturity": maturity_mock,
        "face_value": face_mock
    }
    
    res = preprocess_data(D1)
    print("Returns head:\n", res['returns'].head())
