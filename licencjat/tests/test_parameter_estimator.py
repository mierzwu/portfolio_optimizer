"""
Testy modułu parameter_estimator.py

Obszary:
  1. estimate_avg_cpi        – poprawna średnia CPI z danych historycznych
  2. calculate_shrinkage_mu  – właściwości estymacji James-Stein
  3. estimate_params         – kompletność i spójność wynikowego słownika
"""
import pytest
import numpy as np
import pandas as pd

from parameter_estimator import estimate_avg_cpi, calculate_shrinkage_mu, estimate_params


# ===========================================================================
# 1.  estimate_avg_cpi
# ===========================================================================

class TestEstimateAvgCPI:

    def test_known_values_returns_correct_mean(self):
        """Średnia CPI dla znanych wartości jest poprawna."""
        cpi = pd.Series([0.02, 0.03, 0.04, 0.05])
        result = estimate_avg_cpi(cpi)
        assert result == pytest.approx(0.035, rel=1e-9)

    def test_empty_series_returns_default(self):
        """Pusta seria → wartość domyślna 2.5%."""
        result = estimate_avg_cpi(pd.Series(dtype=float))
        assert result == pytest.approx(0.025)

    def test_nan_values_are_ignored(self):
        """Wartości NaN są pomijane przy obliczaniu średniej."""
        cpi = pd.Series([0.02, np.nan, 0.04])
        result = estimate_avg_cpi(cpi)
        # Tylko [0.02, 0.04] → średnia = 0.03
        assert result == pytest.approx(0.03, rel=1e-9)

    def test_single_value_returns_that_value(self):
        """Jeden odczyt CPI → wynik = ten odczyt."""
        cpi = pd.Series([0.055])
        result = estimate_avg_cpi(cpi)
        assert result == pytest.approx(0.055)


# ===========================================================================
# 2.  calculate_shrinkage_mu
# ===========================================================================

class TestCalculateShrinkageMu:

    def _make_returns(self, n_assets, T, seed=0):
        rng = np.random.default_rng(seed)
        cols = [f"A{i}" for i in range(n_assets)]
        data = rng.normal(0.0004, 0.015, (T, n_assets))
        return pd.DataFrame(data, columns=cols)

    def test_output_index_matches_input_columns(self):
        """Wynikowa seria ma ten sam indeks co kolumny wejściowe."""
        returns = self._make_returns(3, 200)
        result = calculate_shrinkage_mu(returns)
        assert list(result.index) == list(returns.columns)

    def test_lambda_clipped_to_unit_interval(self):
        """Współczynnik lambda musi leżeć w [0, 1] – wynik jest między historyczną
        a globalną średnią."""
        returns = self._make_returns(5, 300)
        mu_hist = returns.mean()
        mu_target = mu_hist.mean()  # grand mean
        result = calculate_shrinkage_mu(returns)
        # Każda wartość shrinkage ∈ [min(mu_hist, mu_target), max(mu_hist, mu_target)]
        lo = np.minimum(mu_hist.values, mu_target)
        hi = np.maximum(mu_hist.values, mu_target)
        assert (result.values >= lo - 1e-12).all()
        assert (result.values <= hi + 1e-12).all()

    def test_identical_returns_no_shrinkage_needed(self):
        """Gdy wszystkie zwroty są identyczne, variance_of_means = 0 → lambda = 1
        i wynik = grand mean = każda historyczna srednia."""
        # Jeden zasób z dużą próbą – variancja estymacji dominuje signal
        returns = pd.DataFrame({"A": np.full(500, 0.001), "B": np.full(500, 0.001)})
        result = calculate_shrinkage_mu(returns)
        # Gdy variance_of_means == 0 (identyczne aktywa), kod ustawia lambda=1 → wynik = mu_target
        # a mu_target == mu_hist (bo wszystkie mu_hist są równe)
        assert result["A"] == pytest.approx(0.001, abs=1e-10)
        assert result["B"] == pytest.approx(0.001, abs=1e-10)

    def test_large_sample_shrinkage_closer_to_historical(self):
        """Dla bardzo dużej próby lambda → 0, czyli wynik ≈ historyczna średnia."""
        rng = np.random.default_rng(7)
        T = 10_000  # bardzo duża próba → mała wariancja estymacji → mała lambda
        cols = ["X", "Y"]
        data = rng.normal([0.001, 0.003], 0.01, (T, 2))
        returns = pd.DataFrame(data, columns=cols)
        result = calculate_shrinkage_mu(returns)
        mu_hist = returns.mean()
        # Wynik powinien być bliski historycznej średniej (lambda mały)
        assert abs(result["X"] - mu_hist["X"]) < abs(result["X"] - mu_hist.mean())


# ===========================================================================
# 3.  estimate_params
# ===========================================================================

class TestEstimateParams:

    def test_required_keys_present(self, processed_data_stocks):
        """Wynikowy słownik zawiera wszystkie wymagane klucze."""
        result = estimate_params(processed_data_stocks)
        for key in ("mu", "sigma", "sigma_shrink", "last_date", "avg_cpi", "current_cpi"):
            assert key in result, f"Brak klucza '{key}' w wyniku estimate_params"

    def test_mu_is_annualized(self, processed_data_stocks):
        """mu_annual = shrinkage_mu_daily * 252 (dokładna równość).

        estimate_params stosuje James-Stein shrinkage do dziennych zwrotów,
        a następnie mnoży przez 252. Test weryfikuje ten proces przez
        bezpośrednie porównanie z manualną kalkulacją.
        """
        from parameter_estimator import calculate_shrinkage_mu
        result = estimate_params(processed_data_stocks)
        returns = processed_data_stocks["returns"]
        # Manualna replika annualizacji z estimate_params
        mu_daily_shrunk = calculate_shrinkage_mu(returns)
        mu_annual_expected = mu_daily_shrunk * 252
        # Sprawdzamy, że mu w wyniku = shrinkage_daily * 252
        pd.testing.assert_series_equal(
            result["mu"].reindex(mu_annual_expected.index),
            mu_annual_expected,
            check_exact=False,
            rtol=1e-6,
        )

    def test_sigma_shrink_is_positive_semidefinite(self, processed_data_stocks):
        """Macierz sigma_shrink (Ledoit-Wolf) jest dodatnio (pół)określona."""
        result = estimate_params(processed_data_stocks)
        eigvals = np.linalg.eigvalsh(result["sigma_shrink"].values)
        assert (eigvals >= -1e-10).all(), "Wszystkie wartości własne muszą być ≥ 0"

    def test_last_date_matches_prices_index(self, processed_data_stocks):
        """last_date = ostatnia data w cenach."""
        result = estimate_params(processed_data_stocks)
        expected_last = processed_data_stocks["prices"].index[-1]
        assert result["last_date"] == expected_last

    def test_sigma_annualized_approx_252x_daily(self, processed_data_stocks):
        """Sigma annualizowana ≈ 252 × sigma dzienna."""
        result = estimate_params(processed_data_stocks)
        returns = processed_data_stocks["returns"]
        sigma_daily = returns.cov()
        sigma_annual = result["sigma"]
        ratio = (sigma_annual / sigma_daily).values
        assert np.allclose(ratio, 252.0, rtol=1e-6), (
            "Sigma powinna być annualizowana przez pomnożenie × 252"
        )

    def test_current_cpi_equals_last_cpi_observation(self, processed_data_stocks):
        """current_cpi = ostatni odczyt historii CPI (jeśli dostępny)."""
        result = estimate_params(processed_data_stocks)
        cpi_history = processed_data_stocks["cpi_history"]
        if not cpi_history.empty:
            assert result["current_cpi"] == pytest.approx(float(cpi_history.iloc[-1]))
