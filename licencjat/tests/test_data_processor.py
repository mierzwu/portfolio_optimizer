"""
Testy modułu data_processor.py

Obszary:
  1. calculate_retail_bond_accrual – poprawność obliczeń EDO i COI
  2. _build_bond_accrual_series    – właściwości szeregu czasowego obligacji
  3. preprocess_data               – zwroty arytmetyczne, imputacja, filtracja
"""
import pytest
import numpy as np
import pandas as pd

from data_processor import (
    calculate_retail_bond_accrual,
    _build_bond_accrual_series,
    preprocess_data,
)
from models import DataPolicy, ImputationMethod


# ===========================================================================
# 1.  calculate_retail_bond_accrual
# ===========================================================================

class TestRetailBondAccrual:
    """Weryfikacja formuł matematycznych EDO i COI."""

    def test_holding_zero_returns_face_value(self):
        """Dla 0 lat trzymania zwracana jest wartość nominalna."""
        result = calculate_retail_bond_accrual(
            cpi_history=[0.03], margin=0.015, first_year_rate=0.05,
            holding_years=0, bond_type="EDO"
        )
        assert result == pytest.approx(100.0)

    def test_edo_one_year_compound(self):
        """EDO po 1 roku: nominał × (1 + first_year_rate)."""
        result = calculate_retail_bond_accrual(
            cpi_history=[], margin=0.015, first_year_rate=0.05,
            holding_years=1, bond_type="EDO"
        )
        assert result == pytest.approx(100.0 * 1.05)

    def test_coi_one_year_coupon(self):
        """COI po 1 roku: nominał + kupon = 100 + 100 × first_year_rate."""
        result = calculate_retail_bond_accrual(
            cpi_history=[], margin=0.015, first_year_rate=0.05,
            holding_years=1, bond_type="COI"
        )
        assert result == pytest.approx(100.0 + 100.0 * 0.05)

    def test_edo_coi_equal_after_one_year(self):
        """Po 1 roku EDO i COI dają taką samą wartość (brak różnicy w kapitalizacji)."""
        edo = calculate_retail_bond_accrual([], 0.015, 0.05, 1, "EDO")
        coi = calculate_retail_bond_accrual([], 0.015, 0.05, 1, "COI")
        assert edo == pytest.approx(coi)

    def test_edo_three_years_compound_interest(self):
        """EDO po 3 latach – ręczna weryfikacja procentu składanego.

        Stopy: rok1=5%, rok2=CPI[0]+marża=3%+1%=4%, rok3=CPI[1]+marża=2%+1%=3%.
        Oczekiwana wartość: 100 × 1.05 × 1.04 × 1.03 = 112.476
        """
        result = calculate_retail_bond_accrual(
            cpi_history=[0.03, 0.02], margin=0.01, first_year_rate=0.05,
            holding_years=3, bond_type="EDO"
        )
        expected = 100.0 * 1.05 * 1.04 * 1.03
        assert result == pytest.approx(expected, rel=1e-9)

    def test_coi_three_years_simple_coupons(self):
        """COI po 3 latach – suma kuponów (bez kapitalizacji).

        Stopy: 5%, 4%, 3% → kupony: 5 + 4 + 3 = 12 PLN.
        Oczekiwana wartość: 100 + 12 = 112.
        """
        result = calculate_retail_bond_accrual(
            cpi_history=[0.03, 0.02], margin=0.01, first_year_rate=0.05,
            holding_years=3, bond_type="COI"
        )
        assert result == pytest.approx(112.0, rel=1e-9)

    def test_edo_exceeds_coi_for_positive_rates(self):
        """EDO zawsze > COI dla tych samych stóp (kapitalizacja daje więcej)."""
        edo = calculate_retail_bond_accrual([0.03, 0.02], 0.01, 0.05, 3, "EDO")
        coi = calculate_retail_bond_accrual([0.03, 0.02], 0.01, 0.05, 3, "COI")
        assert edo > coi

    def test_edo_non_default_face_value(self):
        """EDO działa poprawnie dla face_value ≠ 100."""
        result = calculate_retail_bond_accrual(
            cpi_history=[], margin=0.0, first_year_rate=0.05,
            holding_years=1, bond_type="EDO", face_value=200.0
        )
        assert result == pytest.approx(200.0 * 1.05)

    def test_edo_insufficient_cpi_uses_last_known(self):
        """Gdy brakuje danych CPI, używany jest ostatni dostępny odczyt.

        cpi=[0.03], holding_years=3:
          rok2: CPI[0]=0.03, stopa=0.04
          rok3: CPI[1] brak → CPI[-1]=0.03, stopa=0.04
        Oczekiwana wartość: 100 × 1.05 × 1.04 × 1.04 = 113.568
        """
        result = calculate_retail_bond_accrual(
            cpi_history=[0.03], margin=0.01, first_year_rate=0.05,
            holding_years=3, bond_type="EDO"
        )
        expected = 100.0 * 1.05 * 1.04 * 1.04
        assert result == pytest.approx(expected, rel=1e-9)

    def test_edo_empty_cpi_fallback_to_zero(self):
        """Bez historii CPI stopa od roku 2 = 0 + marża.

        holding_years=2, cpi=[], margin=0.01, first_year_rate=0.05:
          rok2: cpi_val=0.0 → stopa=0.01
        Oczekiwana wartość: 100 × 1.05 × 1.01 = 106.05
        """
        result = calculate_retail_bond_accrual(
            cpi_history=[], margin=0.01, first_year_rate=0.05,
            holding_years=2, bond_type="EDO"
        )
        assert result == pytest.approx(100.0 * 1.05 * 1.01, rel=1e-9)


# ===========================================================================
# 2.  _build_bond_accrual_series
# ===========================================================================

class TestBuildBondAccrualSeries:
    """Weryfikacja szeregu czasowego wartości narosłej obligacji."""

    def _make_dates(self, n=400):
        return pd.bdate_range("2023-01-02", periods=n)

    def test_value_at_issue_date_equals_face_value(self):
        """W dniu emisji wartość = nominał (0 dni trzymania)."""
        dates = self._make_dates()
        issue_date = dates[0]
        series = _build_bond_accrual_series(
            valuation_dates=dates,
            issue_date=issue_date,
            bond_type="EDO",
            margin=0.015,
            first_year_rate=0.05,
            cpi_annual=pd.Series([0.03, 0.03, 0.03]),
        )
        assert series.iloc[0] == pytest.approx(100.0, rel=1e-6)

    def test_edo_monotonically_increasing_positive_rates(self):
        """Seria EDO rośnie monotonicznie dla dodatnich stóp."""
        dates = self._make_dates(500)
        issue_date = dates[0]
        series = _build_bond_accrual_series(
            valuation_dates=dates,
            issue_date=issue_date,
            bond_type="EDO",
            margin=0.015,
            first_year_rate=0.05,
            cpi_annual=pd.Series([0.025, 0.030, 0.035]),
        )
        diffs = series.diff().dropna()
        assert (diffs >= 0).all(), "Wartość EDO powinna rosnąć monotonicznie"

    def test_edo_value_after_one_year_matches_accrual(self):
        """Wartość po roku = calculate_retail_bond_accrual(holding_years=1)."""
        dates = pd.date_range("2023-01-01", periods=366, freq="D")
        issue_date = dates[0]
        series = _build_bond_accrual_series(
            valuation_dates=dates,
            issue_date=issue_date,
            bond_type="EDO",
            margin=0.015,
            first_year_rate=0.05,
            cpi_annual=pd.Series([0.03]),
        )
        # Wartość po 365 dniach (pełny rok)
        val_after_1yr = series.iloc[365]
        expected = calculate_retail_bond_accrual(
            cpi_history=[0.03], margin=0.015, first_year_rate=0.05,
            holding_years=1, bond_type="EDO"
        )
        assert val_after_1yr == pytest.approx(expected, rel=1e-6)

    def test_dates_before_issue_return_face_value(self):
        """Dla dat przed emisją zwracana jest wartość nominalna."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        issue_date = pd.Timestamp("2023-01-01")
        series = _build_bond_accrual_series(
            valuation_dates=dates,
            issue_date=issue_date,
            bond_type="EDO",
            margin=0.015,
            first_year_rate=0.05,
            cpi_annual=pd.Series([0.03]),
        )
        assert (series == 100.0).all()

    def test_series_has_correct_length(self):
        """Długość wynikowej serii = liczba dni wyceny."""
        dates = self._make_dates(300)
        issue_date = dates[0]
        series = _build_bond_accrual_series(
            valuation_dates=dates,
            issue_date=issue_date,
            bond_type="COI",
            margin=0.015,
            first_year_rate=0.05,
            cpi_annual=pd.Series([0.03]),
        )
        assert len(series) == 300


# ===========================================================================
# 3.  preprocess_data
# ===========================================================================

class TestPreprocessData:
    """Weryfikacja funkcji preprocess_data."""

    def test_returns_are_arithmetic(self, d1_raw_stocks_only):
        """Zwroty obliczane są jako pct_change (arytmetyczne), nie logarytmiczne."""
        from data_processor import preprocess_data
        result = preprocess_data(d1_raw_stocks_only)
        prices = result["prices"]
        returns = result["returns"]

        # Ręczna weryfikacja pierwszego zwrotu dla STOCK_A
        p0 = prices["STOCK_A"].iloc[0]
        p1 = prices["STOCK_A"].iloc[1]
        r1 = returns["STOCK_A"].iloc[0]
        expected_arithmetic = (p1 - p0) / p0
        assert r1 == pytest.approx(expected_arithmetic, rel=1e-9), (
            "Zwroty powinny być arytmetyczne (pct_change), nie logarytmiczne"
        )

    def test_returns_length_is_prices_minus_one(self, d1_raw_stocks_only):
        """Liczba obserwacji zwrotów = liczba cen - 1 (pierwszy wiersz NaN)."""
        from data_processor import preprocess_data
        result = preprocess_data(d1_raw_stocks_only)
        assert len(result["returns"]) == len(result["prices"]) - 1

    def test_bond_column_included_in_output(self, d1_raw_with_bond):
        """Po przetworzeniu danych z obligacją, ticker EDO_TEST jest w wyjściu."""
        from data_processor import preprocess_data
        result = preprocess_data(d1_raw_with_bond)
        assert "EDO_TEST" in result["prices"].columns
        assert "EDO_TEST" in result["returns"].columns

    def test_no_nan_after_ffill(self, prices_2stocks, cpi_series):
        """Po imputacji FFILL w cenach nie ma wartości NaN."""
        from data_processor import preprocess_data
        prices_with_gap = prices_2stocks.copy()
        prices_with_gap.iloc[10:15, 0] = np.nan  # wprowadź luki
        d1 = {"prices": prices_with_gap, "cpi_history": cpi_series, "bond_params": pd.DataFrame()}
        result = preprocess_data(d1, policy=DataPolicy(imputation_method=ImputationMethod.FFILL))
        assert not result["prices"].isnull().any().any()

    def test_column_dropped_below_min_observations(self, cpi_series):
        """Kolumna z za małą liczbą obserwacji jest usuwana."""
        from data_processor import preprocess_data
        # Tylko 30 dni cen – poniżej domyślnego progu 100
        dates = pd.bdate_range("2023-01-02", periods=30)
        prices_sparse = pd.DataFrame({"SPARSE": np.ones(30) * 100.0}, index=dates)
        d1 = {"prices": prices_sparse, "cpi_history": cpi_series, "bond_params": pd.DataFrame()}
        policy = DataPolicy(min_observations=100, imputation_method=ImputationMethod.DROP)
        result = preprocess_data(d1, policy=policy)
        assert "SPARSE" not in result["prices"].columns

    def test_no_zero_or_negative_prices_in_output(self, prices_2stocks, cpi_series):
        """Ceny zerowe i ujemne są usuwane przed obliczeniem zwrotów."""
        from data_processor import preprocess_data
        prices_with_zeros = prices_2stocks.copy()
        prices_with_zeros.iloc[50, 0] = 0.0  # wprowadź cenę zerową
        d1 = {"prices": prices_with_zeros, "cpi_history": cpi_series, "bond_params": pd.DataFrame()}
        result = preprocess_data(d1)
        assert (result["prices"] > 0).all().all()

    def test_output_keys(self, d1_raw_stocks_only):
        """Wynik preprocess_data zawiera wymagane klucze."""
        from data_processor import preprocess_data
        result = preprocess_data(d1_raw_stocks_only)
        assert "prices" in result
        assert "returns" in result
        assert "bond_metadata" in result
        assert "cpi_history" in result
