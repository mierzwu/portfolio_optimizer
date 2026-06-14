"""
Testy modułu backtester.py

Obszary:
  1. run_monte_carlo_simulation – kształt wyników, metryki, deterministyczność
  2. Spójność VaR / CVaR
  3. Wykrycie błędu: hardkodowany percentyl 1% niezależnie od cvar_alpha
  4. run_stress_test – obecność scenariuszy, kierunek szoków
"""
import pytest
import numpy as np
import pandas as pd

from backtester import run_monte_carlo_simulation, run_stress_test


# ===========================================================================
# Fixtures lokalne
# ===========================================================================

@pytest.fixture
def mc_weights():
    """Wagi dla 2 akcji."""
    return pd.Series([0.6, 0.4], index=["STOCK_A", "STOCK_B"])


@pytest.fixture
def mc_model_params():
    """Parametry modelu do symulacji Monte Carlo."""
    tickers = ["STOCK_A", "STOCK_B"]
    mu = pd.Series([0.10, 0.08], index=tickers)
    sigma = pd.DataFrame(
        [[0.040, 0.012],
         [0.012, 0.025]],
        index=tickers, columns=tickers,
    )
    return {"mu": mu, "sigma": sigma, "sigma_shrink": sigma.copy()}


# ===========================================================================
# 1.  Kształt i struktura wyników Monte Carlo
# ===========================================================================

class TestMonteCarloShape:

    N_SIM = 500
    T_DAYS = 50  # mała liczba dla szybkości testów

    def test_paths_shape(self, mc_weights, mc_model_params):
        """paths.shape = (n_simulations, time_horizon_days)."""
        result = run_monte_carlo_simulation(
            mc_model_params, mc_weights,
            n_simulations=self.N_SIM, time_horizon_days=self.T_DAYS
        )
        assert result["paths"].shape == (self.N_SIM, self.T_DAYS)

    def test_metrics_keys_present(self, mc_weights, mc_model_params):
        """metrics zawiera expected_return, VaR_*, CVaR_*."""
        result = run_monte_carlo_simulation(
            mc_model_params, mc_weights,
            n_simulations=self.N_SIM, time_horizon_days=self.T_DAYS,
            cvar_alpha=0.95,
        )
        assert "expected_return" in result["metrics"]
        assert "VaR_95" in result["metrics"]
        assert "CVaR_95" in result["metrics"]

    def test_metric_keys_reflect_cvar_alpha(self, mc_weights, mc_model_params):
        """Klucze metryk zmieniają się w zależności od cvar_alpha."""
        r99 = run_monte_carlo_simulation(
            mc_model_params, mc_weights,
            n_simulations=self.N_SIM, time_horizon_days=self.T_DAYS, cvar_alpha=0.99
        )
        r95 = run_monte_carlo_simulation(
            mc_model_params, mc_weights,
            n_simulations=self.N_SIM, time_horizon_days=self.T_DAYS, cvar_alpha=0.95
        )
        assert "VaR_99" in r99["metrics"]
        assert "VaR_95" in r95["metrics"]

    def test_output_keys(self, mc_weights, mc_model_params):
        """Wynik zawiera klucze 'paths' i 'metrics'."""
        result = run_monte_carlo_simulation(
            mc_model_params, mc_weights,
            n_simulations=self.N_SIM, time_horizon_days=self.T_DAYS
        )
        assert "paths" in result
        assert "metrics" in result


# ===========================================================================
# 2.  Poprawność wartości VaR / CVaR
# ===========================================================================

class TestMonteCarloMetrics:

    N_SIM = 2000
    T_DAYS = 252

    def test_cvar_greater_equal_var(self, mc_weights, mc_model_params):
        """CVaR ≥ VaR (CVaR jest średnią ogona, więc musi być ≥ VaR)."""
        result = run_monte_carlo_simulation(
            mc_model_params, mc_weights,
            n_simulations=self.N_SIM, time_horizon_days=self.T_DAYS, cvar_alpha=0.99
        )
        var = result["metrics"]["VaR_99"]
        cvar = result["metrics"]["CVaR_99"]
        assert cvar >= var - 1e-8, (
            f"CVaR ({cvar:.4f}) powinno być ≥ VaR ({var:.4f})"
        )

    def test_var_and_cvar_are_positive_for_risky_portfolio(self, mc_weights, mc_model_params):
        """Dla portfela z realnym ryzykiem (vol ~15%) VaR i CVaR są dodatnie (strata).

        VaR/CVaR mogą być ujemne, jeśli nawet najgorszy ogon rozkładu
        pokazuje zysk – ale dla typowych parametrów testowych strata jest
        wyraźna i obie wartości powinny przekraczać zero.
        """
        result = run_monte_carlo_simulation(
            mc_model_params, mc_weights,
            n_simulations=self.N_SIM, time_horizon_days=self.T_DAYS, cvar_alpha=0.99
        )
        assert result["metrics"]["VaR_99"] > 0.0, "VaR powinno być > 0 dla ryzykownego portfela"
        assert result["metrics"]["CVaR_99"] > 0.0, "CVaR powinno być > 0 dla ryzykownego portfela"

    def test_expected_return_reasonable_range(self, mc_weights, mc_model_params):
        """Oczekiwany zwrot roczny mieści się w rozsądnym zakresie (-50%, +100%)."""
        result = run_monte_carlo_simulation(
            mc_model_params, mc_weights,
            n_simulations=self.N_SIM, time_horizon_days=self.T_DAYS
        )
        er = result["metrics"]["expected_return"]
        assert -0.5 < er < 1.0

    def test_deterministic_with_fixed_seed(self, mc_weights, mc_model_params):
        """Symulacja jest deterministyczna – dwa kolejne uruchomienia dają ten sam wynik.

        Uwaga: np.random.seed(42) jest wywoływany wewnątrz funkcji,
        więc kolejne wywołania zawsze zwracają identyczne wyniki.
        """
        r1 = run_monte_carlo_simulation(
            mc_model_params, mc_weights,
            n_simulations=200, time_horizon_days=50, cvar_alpha=0.99
        )
        r2 = run_monte_carlo_simulation(
            mc_model_params, mc_weights,
            n_simulations=200, time_horizon_days=50, cvar_alpha=0.99
        )
        assert r1["metrics"]["VaR_99"] == pytest.approx(r2["metrics"]["VaR_99"])
        assert r1["metrics"]["expected_return"] == pytest.approx(r2["metrics"]["expected_return"])

    def test_paths_all_positive_start_at_one(self, mc_weights, mc_model_params):
        """Ścieżki skumulowanego bogactwa zaczynają się w okolicach 1.0."""
        result = run_monte_carlo_simulation(
            mc_model_params, mc_weights,
            n_simulations=100, time_horizon_days=50
        )
        # paths[s, 0] = cumprod(1 + r)[0] = 1 + r[0] – może być bliskie 1
        first_day = result["paths"][:, 0]
        assert (first_day > 0).all(), "Wszystkie ścieżki muszą mieć dodatnią wartość"


# ===========================================================================
# 3.  Wykrycie błędu: cvar_alpha nie wpływa na obliczenia kwantyla
#
#     Obecna implementacja twardokoduje percentyl 1% niezależnie od cvar_alpha,
#     zmieniając tylko nazwy kluczy w słowniku wyników.
#     Poprawna implementacja powinna używać (1 - cvar_alpha) * 100 jako percentyla.
#
#     Test PRZEJDZIE gdy implementacja jest poprawna.
#     Test WYKRYJE BŁĄD (assert failure) gdy implementacja jest błędna.
# ===========================================================================

class TestCvarAlphaQuantileBug:

    N_SIM = 2000
    T_DAYS = 252

    def test_var99_strictly_greater_than_var95(self, mc_weights, mc_model_params):
        """VaR_99 (1% ogon) musi być > VaR_95 (5% ogon) – bardziej ekstremalna strata.

        Oba wywołania używają tego samego seed, więc produkowałby identyczne
        skumulowane zwroty. Różnica polega wyłącznie na wyborze kwantyla:
          - alpha=0.99 → percentyl 1%  (bardziej ekstremalna strata)
          - alpha=0.95 → percentyl 5%  (mniej ekstremalna strata)
        Dlatego VaR_99 MUSI być większe od VaR_95.

        UWAGA: Jeśli ten test FAILUJE, oznacza to, że implementacja
        twardokoduje percentyl 1% niezależnie od alpha (znany błąd).
        """
        r99 = run_monte_carlo_simulation(
            mc_model_params, mc_weights,
            n_simulations=self.N_SIM, time_horizon_days=self.T_DAYS, cvar_alpha=0.99
        )
        r95 = run_monte_carlo_simulation(
            mc_model_params, mc_weights,
            n_simulations=self.N_SIM, time_horizon_days=self.T_DAYS, cvar_alpha=0.95
        )
        assert r99["metrics"]["VaR_99"] > r95["metrics"]["VaR_95"], (
            "BŁĄD IMPLEMENTACJI: cvar_alpha nie wpływa na percentyl używany "
            "do obliczenia VaR/CVaR. Należy użyć percentyla "
            "(1 - cvar_alpha) * 100 zamiast hardkodowanego 1%."
        )


# ===========================================================================
# 4.  run_stress_test
# ===========================================================================

class TestStressTest:

    @pytest.fixture
    def stress_setup(self, input_min_risk, det_model_params, det_processed_data):
        """Zestaw danych do stress-testów."""
        return input_min_risk, det_processed_data, det_model_params

    def test_all_scenarios_present(self, stress_setup):
        """Wynik zawiera wszystkie 4 scenariusze."""
        vinput, proc, params = stress_setup
        results = run_stress_test(vinput, proc, params)
        expected = {"Base", "CPI +5%", "Stocks -20%", "High Volatility (x2)"}
        assert set(results.keys()) == expected

    def test_each_scenario_has_weights_and_metrics(self, stress_setup):
        """Każdy scenariusz zawiera 'weights' i 'metrics'."""
        vinput, proc, params = stress_setup
        results = run_stress_test(vinput, proc, params)
        for name, res in results.items():
            assert "weights" in res, f"Scenariusz '{name}': brak 'weights'"
            assert "metrics" in res, f"Scenariusz '{name}': brak 'metrics'"

    def test_scenario_weights_sum_to_one(self, stress_setup):
        """W każdym scenariuszu suma wag ≈ 1."""
        vinput, proc, params = stress_setup
        results = run_stress_test(vinput, proc, params)
        for name, res in results.items():
            total = res["weights"].sum()
            assert total == pytest.approx(1.0, abs=1e-3), (
                f"Scenariusz '{name}': suma wag = {total:.6f}, oczekiwano ≈ 1.0"
            )

    def test_stocks_minus20_reduces_expected_return(self, stress_setup):
        """Szok 'Stocks -20%' obniża mu akcji, co powinno zmniejszyć oczekiwany zwrot."""
        vinput, proc, params = stress_setup
        results = run_stress_test(vinput, proc, params)
        base_ret = results["Base"]["metrics"]["expected_return"]
        shocked_ret = results["Stocks -20%"]["metrics"]["expected_return"]
        assert shocked_ret <= base_ret + 1e-4, (
            f"'Stocks -20%': oczekiwany zwrot ({shocked_ret:.4f}) nie spadł "
            f"względem base ({base_ret:.4f})"
        )

    def test_high_volatility_increases_portfolio_volatility(self, stress_setup):
        """Szok 'High Volatility (x2)' podwaja sigma, co musi zwiększyć volatility portfela."""
        vinput, proc, params = stress_setup
        results = run_stress_test(vinput, proc, params)
        base_vol = results["Base"]["metrics"]["volatility"]
        hv_vol = results["High Volatility (x2)"]["metrics"]["volatility"]
        # Podwojenie macierzy kowariancji → sqrt(2)× większa volatility
        # Tolerancja: optimizer może zmienić wagi, ale efekt musi być istotny
        assert hv_vol > base_vol * 1.2, (
            f"High Volatility: vol ({hv_vol:.4f}) nie wzrosła wystarczająco "
            f"względem base ({base_vol:.4f})"
        )
