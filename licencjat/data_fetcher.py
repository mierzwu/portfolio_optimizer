import pandas as pd
import yfinance as yf
import hashlib
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import timedelta, date
from models import InputData, InstrumentType
from database import get_bond_penalties_by_type, get_bond_emissions, initialize_db

logger = logging.getLogger(__name__)


def fetch_gpw_data(tickers: List[str], start_date: date, end_date: date) -> pd.DataFrame:
    """
    Pobiera dane z GPW (via Yahoo Finance) dla podanych tickerów.
    """
    logger.debug(f"[GPW] Pobieranie danych dla: {tickers} od {start_date} do {end_date}")
    
    if not tickers:
        return pd.DataFrame()

    # Yahoo Finance wymaga sufiksu .WA dla GPW
    yf_tickers = [f"{t}.WA" if not t.endswith('.WA') else t for t in tickers]
    
    try:
        # Pobieranie danych
        data = yf.download(yf_tickers, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            logger.warning("[GPW] Ostrzeżenie: Brak danych z Yahoo Finance.")
            return pd.DataFrame(columns=tickers)

        # Obsługa struktury zwracanej przez yfinance
        if 'Close' in data.columns:
            prices = data['Close']
        else:
            # Jeśli pobrano tylko jeden ticker i struktura jest płaska (starsze wersje lub specyficzne przypadki)
            prices = data

        # Jeśli mamy tylko jeden ticker, prices może być Series lub DataFrame z jedną kolumną
        if len(tickers) == 1:
            if isinstance(prices, pd.Series):
                prices = prices.to_frame()
            prices.columns = tickers
        else:
            # Usuwanie sufiksu .WA z nazw kolumn
            prices.columns = [c.replace('.WA', '') for c in prices.columns]

        # Upewnienie się, że mamy wszystkie żądane tickery (wstawienie NaN dla brakujących)
        for t in tickers:
            if t not in prices.columns:
                logger.warning(f"[GPW] Ostrzeżenie: Brak danych dla {t}")
                prices[t] = float('nan')

        # Sortowanie kolumn zgodnie z kolejnością wejściową
        prices = prices[tickers]
        
        # Wypełnianie braków danych (Forward Fill) - typowe dla szeregów czasowych
        prices = prices.ffill()
        
        return prices

    except Exception as e:
        logger.error(f"[GPW] Błąd podczas pobierania danych: {e}")
        return pd.DataFrame(columns=tickers)

def fetch_bonds_data(tickers: List[str]) -> pd.DataFrame:
    """
    Pobiera parametry obligacji detalicznych (EDO/COI) z bazy SQLite
    (tabela emisje_obligacji).

    Dopasowanie po symbol_emisji (np. 'EDO0135'). Jeśli brak dopasowania,
    zwraca pusty DataFrame — fetch_market_data uzupełni dane z InputData.
    """
    logger.debug(f"[DB] Pobieranie parametrów obligacji dla: {tickers}")
    try:
        initialize_db()
        emissions = get_bond_emissions()
        penalties = get_bond_penalties_by_type()

        if not emissions:
            logger.warning("[DB] Tabela emisji obligacji jest pusta — parametry zostaną pobrane z danych wejściowych.")
            return pd.DataFrame(columns=['bond_type', 'margin', 'first_year_rate', 'issue_date', 'kara_wykup'])

        by_symbol: Dict[str, Any] = {e['symbol_emisji']: e for e in emissions}

        rows = []
        for ticker in tickers:
            match = by_symbol.get(ticker)
            if match is None:
                continue
            rows.append({
                'ticker': ticker,
                'bond_type': match['typ_obligacji'],
                'margin': float(match['marza_odsetkowa']),
                'first_year_rate': float(match['oprocentowanie_rok_1']),
                'issue_date': pd.to_datetime(match['data_poczatkowa']),
                'dlugosc_lat': int(match.get('dlugosc_lat', 10)),
                'kara_wykup': float(match['kara_wykup']),
            })

        if not rows:
            logger.warning(f"[DB] Brak dopasowania w SQLite dla: {tickers} — parametry z InputData.")
            return pd.DataFrame(columns=['bond_type', 'margin', 'first_year_rate', 'issue_date', 'kara_wykup'])

        result = pd.DataFrame(rows).set_index('ticker')
        result['kara_wykup'] = result['bond_type'].map(penalties).fillna(result['kara_wykup'])
        return result

    except Exception as e:
        logger.error(f"[DB] Błąd podczas pobierania parametrów obligacji z SQLite: {e}")
        return pd.DataFrame(columns=['bond_type', 'margin', 'first_year_rate', 'issue_date', 'kara_wykup'])

def fetch_cpi_data(start_year: int, end_year: int) -> pd.Series:
    """
    Zwraca roczne odczyty CPI dla Polski (wartości ułamkowe, np. 0.035 = 3.5%).
    Indeks = rok (int). Źródło: GUS.
    """
    logger.debug(f"[CPI] Roczne dane CPI za lata {start_year}-{end_year}")

    # Polska, roczny wskaźnik CPI (GUS), rok poprzedni = 100
    _CPI = {
        1996: 0.199, 1997: 0.149,
        1998: 0.118, 1999: 0.073, 2000: 0.101, 2001: 0.055,
        2002: 0.019, 2003: 0.008, 2004: 0.035, 2005: 0.021,
        2006: 0.010, 2007: 0.025, 2008: 0.042, 2009: 0.035,
        2010: 0.026, 2011: 0.043, 2012: 0.037, 2013: 0.009,
        2014: 0.000, 2015: -0.009, 2016: -0.006, 2017: 0.020,
        2018: 0.016, 2019: 0.023, 2020: 0.034, 2021: 0.051,
        2022: 0.144, 2023: 0.114, 2024: 0.036, 2025: 0.036,
    }

    series = pd.Series(
        {k: v for k, v in _CPI.items() if start_year <= k <= end_year}
    ).sort_index()
    logger.debug(f"[CPI] Zwracam {len(series)} odczytów CPI.")
    return series

def parse_estimation_window(window_str: str) -> timedelta:
    """
    Konwertuje string np. '3Y' na timedelta.
    Uproszczona implementacja.
    """
    unit = window_str[-1].upper()
    value = int(window_str[:-1])
    
    if unit == 'Y':
        return timedelta(days=value * 365)
    elif unit == 'M':
        return timedelta(days=value * 30)
    elif unit == 'D':
        return timedelta(days=value)
    else:
        raise ValueError(f"Nieznana jednostka czasu: {unit}")

def get_cache_key(validated_input: InputData, history_start_date: date, history_end_date: date) -> str:
    """Generuje unikalny klucz cache na podstawie parametrów wejściowych."""
    tickers = sorted([item.ticker for item in validated_input.portfolio])
    sources = sorted(validated_input.data_sources)
    # Klucz zależy od: wersji danych, dat, tickerów, źródeł
    key_str = f"{validated_input.data_policy.data_version}_{history_start_date}_{history_end_date}_{','.join(tickers)}_{','.join(sources)}"
    return hashlib.md5(key_str.encode()).hexdigest()

def fetch_market_data(validated_input: InputData) -> Dict[str, Any]:
    """
    Główna funkcja koordynująca pobieranie danych.
    
    Wejście: validated_input (obiekt InputData)
    Wyjście: D1_raw (słownik z DataFrame'ami: prices, ytm, coupons, maturity)
    """
    
    # 1. Określenie zakresu dat na podstawie estimation_window
    # Zakładamy, że start_date w input to data początku inwestycji/rebalancingu,
    # więc dane historyczne potrzebujemy WSTECZ od tej daty.
    # Lub jeśli start_date to początek danych historycznych, to bierzemy od start_date.
    # Zwykle estimation_window mówi ile danych wstecz potrzebujemy do optymalizacji.
    
    # Interpretacja: start_date w input to "dzisiaj" dla symulacji lub data startu portfela.
    # Dane historyczne potrzebne są z okresu [start_date - estimation_window, start_date].
    
    window_delta = parse_estimation_window(validated_input.estimation_window)
    history_end_date = validated_input.start_date
    history_start_date = history_end_date - window_delta
    
    logger.debug(f"Zakres danych historycznych: {history_start_date} - {history_end_date}")

    # Cache check
    if validated_input.data_policy.use_cache:
        cache_dir = Path(validated_input.data_policy.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = get_cache_key(validated_input, history_start_date, history_end_date)
        cache_file = cache_dir / f"market_data_{cache_key}.pkl"
        
        if cache_file.exists():
            logger.debug(f"[DataFetcher] Znaleziono cache: {cache_file}. Wczytywanie...")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"[DataFetcher] Błąd odczytu cache: {e}. Pobieranie ponowne.")

    # 2. Segregacja instrumentów
    stocks = [item.ticker for item in validated_input.portfolio if item.instrument_type == InstrumentType.STOCK]
    bonds = [item for item in validated_input.portfolio if item.instrument_type == InstrumentType.BOND]
    bond_tickers = [b.symbol_emisji or b.ticker for b in bonds]

    # 3. Pobieranie danych

    # Ceny akcji (GPW)
    prices_df = pd.DataFrame()
    if stocks:
        if "GPW" in validated_input.data_sources:
            prices_df = fetch_gpw_data(stocks, history_start_date, history_end_date)

    # Parametry obligacji detalicznych (baza SQLite)
    bonds_info_df = pd.DataFrame()
    if bond_tickers:
        bonds_info_df = fetch_bonds_data(bond_tickers)

        # Fallback: Użyj danych z inputu dla brakujących obligacji
        input_bonds_data = []
        for b in bonds:
            ticker = b.symbol_emisji or b.ticker
            if bonds_info_df.empty or ticker not in bonds_info_df.index:
                if b.bond_type is not None:
                    logger.info(f"[DataFetcher] Używanie danych z inputu dla obligacji (brak w bazie): {ticker}")
                    input_bonds_data.append({
                        'ticker': ticker,
                        'bond_type': b.bond_type.value,
                        'margin': b.margin if b.margin is not None else 0.015,
                        'first_year_rate': b.first_year_rate if b.first_year_rate is not None else 0.065,
                        'issue_date': pd.to_datetime(b.issue_date) if b.issue_date is not None else pd.Timestamp(history_start_date),
                        'dlugosc_lat': 4 if b.bond_type.value == 'COI' else 10,
                    })

        if input_bonds_data:
            db_penalties = get_bond_penalties_by_type()
            for entry in input_bonds_data:
                entry.setdefault('kara_wykup', db_penalties.get(entry.get('bond_type', ''), 3.0))
            df_input = pd.DataFrame(input_bonds_data).set_index('ticker')
            bonds_info_df = df_input if bonds_info_df.empty else pd.concat([bonds_info_df, df_input])

    # Mapowanie date_acquired z pozycji portfelowych (domyślnie: history_end_date = dziś)
    if not bonds_info_df.empty:
        for b in bonds:
            ticker = b.symbol_emisji or b.ticker
            if ticker in bonds_info_df.index:
                acq = b.date_acquired if b.date_acquired is not None else history_end_date
                bonds_info_df.at[ticker, 'date_acquired'] = pd.to_datetime(acq)

    # Dane CPI (roczne, historyczne)
    start_year = history_start_date.year
    end_year = history_end_date.year
    cpi_series = fetch_cpi_data(start_year, end_year)

    # 4. Konstrukcja obiektu wyjściowego D1_raw
    D1_raw = {
        "prices": prices_df,
        "cpi_history": cpi_series,   # pd.Series: indeks=rok(int), wartości=CPI ułamkowe
        "bond_params": bonds_info_df, # DataFrame: bond_type, margin, first_year_rate, issue_date
        "metadata": {
            "history_start": history_start_date,
            "history_end": history_end_date,
            "sources_used": validated_input.data_sources
        }
    }
    
    # Save to cache
    if validated_input.data_policy.use_cache:
        try:
            cache_dir = Path(validated_input.data_policy.cache_dir)
            cache_key = get_cache_key(validated_input, history_start_date, history_end_date)
            cache_file = cache_dir / f"market_data_{cache_key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(D1_raw, f)
            logger.debug(f"[DataFetcher] Zapisano dane do cache: {cache_file}")
        except Exception as e:
            logger.error(f"[DataFetcher] Błąd zapisu cache: {e}")
    
    return D1_raw
