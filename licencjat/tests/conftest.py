"""
Wspólne fixtures dla całego zestawu testów.
Dodaje katalog licencjat/ do sys.path, aby importy modułów działały
bez instalowania pakietu.
"""
import sys
import os

# Upewnij się, że licencjat/ jest w sys.path (fallback dla starszych pytest bez pythonpath)
_LICENCJAT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _LICENCJAT_ROOT not in sys.path:
    sys.path.insert(0, _LICENCJAT_ROOT)

import pytest
import numpy as np
import pandas as pd

from models import (
    InputData,
    PortfolioItem,
    OptimizationParameters,
    ConstraintsSettings,
    ExecutionConfig,
    GoalType,
    InstrumentType,
    DataPolicy,
    RebalancingFreq,
)

# ---------------------------------------------------------------------------
# Stałe
# ---------------------------------------------------------------------------
_LAST_DATE = pd.Timestamp("2024-01-02")  # wtorek – dzień roboczy
_TICKERS = ["STOCK_A", "STOCK_B"]
_N_DAYS = 252


# ---------------------------------------------------------------------------
# Pomocnicza fabryka cen akcji
# ---------------------------------------------------------------------------
def _make_stock_prices(tickers=None, n_days=_N_DAYS, seed=42, last_date=_LAST_DATE):
    if tickers is None:
        tickers = _TICKERS
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=last_date, periods=n_days)
    data = {
        t: 100.0 * np.cumprod(1 + rng.normal(0.0004 + i * 0.0001, 0.015, n_days))
        for i, t in enumerate(tickers)
    }
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# Fixtures – surowe dane
# ---------------------------------------------------------------------------
@pytest.fixture
def prices_2stocks():
    """252 dni cen akcji dla STOCK_A i STOCK_B."""
    return _make_stock_prices()


@pytest.fixture
def cpi_series():
    """Seria historycznych odczytów CPI (5 lat)."""
    return pd.Series([0.025, 0.030, 0.040, 0.035, 0.028])


@pytest.fixture
def bond_params_df(prices_2stocks):
    """Parametry jednej obligacji EDO (indeks = ticker)."""
    issue_date = prices_2stocks.index[0]
    return pd.DataFrame(
        {
            "bond_type": ["EDO"],
            "margin": [0.015],
            "first_year_rate": [0.0535],
            "issue_date": [issue_date],
            "kara_wykup": [2.0],
        },
        index=["EDO_TEST"],
    )


@pytest.fixture
def d1_raw_with_bond(prices_2stocks, cpi_series, bond_params_df):
    """D1_raw zawierające akcje + jedną obligację EDO."""
    return {
        "prices": prices_2stocks,
        "cpi_history": cpi_series,
        "bond_params": bond_params_df,
    }


@pytest.fixture
def d1_raw_stocks_only(prices_2stocks, cpi_series):
    """D1_raw bez obligacji – same akcje."""
    return {
        "prices": prices_2stocks,
        "cpi_history": cpi_series,
        "bond_params": pd.DataFrame(),
    }


# ---------------------------------------------------------------------------
# Fixtures – przetworzone dane
# ---------------------------------------------------------------------------
@pytest.fixture
def processed_data_stocks(d1_raw_stocks_only):
    from data_processor import preprocess_data
    return preprocess_data(d1_raw_stocks_only)


@pytest.fixture
def processed_data_with_bond(d1_raw_with_bond):
    from data_processor import preprocess_data
    return preprocess_data(d1_raw_with_bond)


# ---------------------------------------------------------------------------
# Fixtures – parametry modelu (deterministyczne)
# ---------------------------------------------------------------------------
@pytest.fixture
def det_model_params():
    """Deterministyczne parametry dla 2 akcji – znane wartości do testów."""
    tickers = _TICKERS
    mu = pd.Series([0.10, 0.08], index=tickers)
    sigma = pd.DataFrame(
        [[0.040, 0.012],
         [0.012, 0.025]],
        index=tickers,
        columns=tickers,
    )
    return {
        "mu": mu,
        "sigma": sigma,
        "sigma_shrink": sigma.copy(),
        "last_date": _LAST_DATE,
        "avg_cpi": 0.025,
        "current_cpi": 0.030,
    }


@pytest.fixture
def det_processed_data():
    """Przetworzone dane pasujące do det_model_params (last_date = _LAST_DATE)."""
    prices = _make_stock_prices()
    returns = prices.pct_change().dropna()
    return {
        "prices": prices,
        "returns": returns,
        "bond_metadata": {"bond_params": pd.DataFrame()},
        "cpi_history": pd.Series([0.025, 0.030]),
    }


# ---------------------------------------------------------------------------
# Fixtures – InputData
# ---------------------------------------------------------------------------
def _base_constraints(**overrides):
    defaults = dict(
        max_weight=1.0,
        min_trade_unit=0.0,
        transaction_cost_pct=0.001,
        max_bond_weight=1.0,
    )
    defaults.update(overrides)
    return ConstraintsSettings(**defaults)


@pytest.fixture
def input_min_risk():
    """InputData: min ryzyka, 2 akcje, faza planowania (brak kosztów)."""
    return InputData(
        portfolio=[
            PortfolioItem(ticker="STOCK_A", instrument_type=InstrumentType.STOCK, quantity=10),
            PortfolioItem(ticker="STOCK_B", instrument_type=InstrumentType.STOCK, quantity=20),
        ],
        parametry_opt=OptimizationParameters(goal_type=GoalType.MIN_RISK, cvar_alpha=None),
        estimation_window="1Y",
        is_planning_phase=True,
        ustawienia_ograniczen=_base_constraints(),
    )


@pytest.fixture
def input_max_return():
    """InputData: maks. zwrotu, 2 akcje, faza planowania."""
    return InputData(
        portfolio=[
            PortfolioItem(ticker="STOCK_A", instrument_type=InstrumentType.STOCK, quantity=10),
            PortfolioItem(ticker="STOCK_B", instrument_type=InstrumentType.STOCK, quantity=20),
        ],
        parametry_opt=OptimizationParameters(goal_type=GoalType.MAX_RETURN, cvar_alpha=None),
        estimation_window="1Y",
        is_planning_phase=True,
        ustawienia_ograniczen=_base_constraints(),
    )
