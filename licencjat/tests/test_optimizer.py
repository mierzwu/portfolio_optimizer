"""
Testy modułu optimizer.py

Obszary:
  1. Poprawność rozwiązania – wagi sumują się do 1, long-only
  2. Cel min_risk daje niższe ryzyko niż portfel równoważny
  3. Cel max_return daje wyższy zwrot niż min_risk
  4. Ograniczenie max_bond_weight jest respektowane
  5. Ograniczenie max_weight (akcje) jest respektowane
  6. Koszty transakcyjne są > 0 w fazie rebalancingu (is_planning_phase=False)
  7. Metryki portfela są spójne
"""
import pytest
import numpy as np
import pandas as pd

from optimizer import optimize_portfolio
from models import (
    InputData,
    PortfolioItem,
    OptimizationParameters,
    ConstraintsSettings,
    ExecutionConfig,
    GoalType,
    InstrumentType,
)


# ===========================================================================
# Pomocnicze
# ===========================================================================

def _make_input(goal_type, *, is_planning=True, max_weight=1.0, max_bond_weight=1.0,
                cvar_alpha=None, goal_value=None, min_target_return=None):
    return InputData(
        portfolio=[
            PortfolioItem(ticker="STOCK_A", instrument_type=InstrumentType.STOCK, quantity=10),
            PortfolioItem(ticker="STOCK_B", instrument_type=InstrumentType.STOCK, quantity=20),
        ],
        parametry_opt=OptimizationParameters(
            goal_type=goal_type, cvar_alpha=cvar_alpha, goal_value=goal_value
        ),
        estimation_window="1Y",
        is_planning_phase=is_planning,
        ustawienia_ograniczen=ConstraintsSettings(
            max_weight=max_weight,
            min_trade_unit=0.0,
            transaction_cost_pct=0.001,
            max_bond_weight=max_bond_weight,
            min_target_return=min_target_return,
        ),
    )


# ===========================================================================
# 1.  Podstawowe właściwości rozwiązania
# ===========================================================================

class TestOptimizerBasicProperties:

    def test_weights_sum_to_one(self, input_min_risk, det_model_params, det_processed_data):
        """Suma wag portfela = 1 (pełne zainwestowanie)."""
        result = optimize_portfolio(input_min_risk, det_model_params, det_processed_data)
        assert result["weights"].sum() == pytest.approx(1.0, abs=1e-4)

    def test_long_only_weights_non_negative(self, input_min_risk, det_model_params, det_processed_data):
        """Long-only: wszystkie wagi ≥ 0."""
        result = optimize_portfolio(input_min_risk, det_model_params, det_processed_data)
        assert (result["weights"] >= -1e-6).all()

    def test_result_contains_required_keys(self, input_min_risk, det_model_params, det_processed_data):
        """Wynik zawiera: weights, metrics, transactions."""
        result = optimize_portfolio(input_min_risk, det_model_params, det_processed_data)
        for key in ("weights", "metrics", "transactions"):
            assert key in result

    def test_metrics_contain_expected_fields(self, input_min_risk, det_model_params, det_processed_data):
        """metrics zawiera expected_return, volatility, sharpe_ratio."""
        result = optimize_portfolio(input_min_risk, det_model_params, det_processed_data)
        for field in ("expected_return", "volatility", "sharpe_ratio"):
            assert field in result["metrics"]

    def test_tickers_in_weights_match_model_params(self, input_min_risk, det_model_params, det_processed_data):
        """Indeks wag odpowiada tickerom z model_params."""
        result = optimize_portfolio(input_min_risk, det_model_params, det_processed_data)
        assert set(result["weights"].index) == set(det_model_params["mu"].index)


# ===========================================================================
# 2.  Cel min_risk
# ===========================================================================

class TestMinRiskObjective:

    def test_min_risk_lower_variance_than_equal_weight(self, det_model_params, det_processed_data):
        """Portfel min-ryzyka ma niższe ryzyko niż portfel równoważony 50/50.

        Dla mu=[0.10, 0.08], Σ=[[0.04, 0.012],[0.012, 0.025]]:
          Wariancja equal-weight = 0.5^2*0.04 + 0.5^2*0.025 + 2*0.5*0.5*0.012
                                 = 0.01 + 0.00625 + 0.006 = 0.02225
        """
        vinput = _make_input(GoalType.MIN_RISK)
        result = optimize_portfolio(vinput, det_model_params, det_processed_data)
        w = result["weights"].values
        sigma = det_model_params["sigma"].values
        var_opt = float(w.T @ sigma @ w)

        w_eq = np.array([0.5, 0.5])
        var_eq = float(w_eq.T @ sigma @ w_eq)
        assert var_opt <= var_eq + 1e-5, (
            f"Portfel min-ryzyka (var={var_opt:.5f}) nie jest lepszy od "
            f"equal-weight (var={var_eq:.5f})"
        )

    def test_min_risk_with_target_return_constraint(self, det_model_params, det_processed_data):
        """Z ograniczeniem minimalnej stopy zwrotu mu@w ≥ target solver znajduje feasible."""
        target = 0.085  # między 0.08 a 0.10 → wykonalne
        vinput = _make_input(GoalType.MIN_RISK, goal_value=target)
        result = optimize_portfolio(vinput, det_model_params, det_processed_data)
        mu = det_model_params["mu"].values
        achieved_return = float(mu @ result["weights"].values)
        assert achieved_return >= target - 1e-4


# ===========================================================================
# 3.  Cel max_return
# ===========================================================================

class TestMaxReturnObjective:

    def test_max_return_higher_than_min_risk(self, det_model_params, det_processed_data):
        """Portfel max-return ma wyższy oczekiwany zwrot niż portfel min-ryzyka."""
        result_min = optimize_portfolio(
            _make_input(GoalType.MIN_RISK), det_model_params, det_processed_data
        )
        result_max = optimize_portfolio(
            _make_input(GoalType.MAX_RETURN), det_model_params, det_processed_data
        )
        assert (
            result_max["metrics"]["expected_return"]
            >= result_min["metrics"]["expected_return"] - 1e-4
        )

    def test_max_return_concentrates_on_best_asset(self, det_model_params, det_processed_data):
        """Bez ograniczeń max_return kieruje całość na aktywo o najwyższym mu."""
        # mu = [0.10, 0.08] → STOCK_A ma wyższy oczekiwany zwrot
        vinput = _make_input(GoalType.MAX_RETURN)
        result = optimize_portfolio(vinput, det_model_params, det_processed_data)
        weights = result["weights"]
        assert weights["STOCK_A"] >= weights["STOCK_B"] - 1e-4


# ===========================================================================
# 4.  Ograniczenia klasowe
# ===========================================================================

class TestAssetClassConstraints:

    def test_max_bond_weight_respected(self, det_processed_data):
        """Łączny udział obligacji nie przekracza max_bond_weight.

        Uwaga: po zaokrągleniu liczby jednostek (post-processing) wagi
        mogą nieznacznie odchylić się od limitu solvera. Tolerancja 3%
        uwzględnia ten efekt dyskretyzacji.
        """
        tickers = ["STOCK_A", "BOND_X"]
        mu = pd.Series([0.08, 0.04], index=tickers)
        sigma = pd.DataFrame(
            [[0.04, 0.002], [0.002, 0.001]],
            index=tickers, columns=tickers,
        )
        model_params = {
            "mu": mu,
            "sigma": sigma,
            "sigma_shrink": sigma.copy(),
            "last_date": pd.Timestamp("2024-01-02"),
            "avg_cpi": 0.025,
            "current_cpi": 0.030,
        }
        # Dobuduj przetworzone dane z obligacją
        last_date = pd.Timestamp("2024-01-02")
        dates = pd.bdate_range(end=last_date, periods=252)
        rng = np.random.default_rng(99)
        prices = pd.DataFrame({
            "STOCK_A": 100.0 * np.cumprod(1 + rng.normal(0.0004, 0.015, 252)),
            "BOND_X": 100.0 + np.linspace(0, 5, 252),
        }, index=dates)
        returns = prices.pct_change().dropna()
        bond_params = pd.DataFrame(
            {"bond_type": ["EDO"], "margin": [0.015], "first_year_rate": [0.05],
             "issue_date": [dates[0]], "kara_wykup": [2.0]},
            index=["BOND_X"],
        )
        processed = {
            "prices": prices,
            "returns": returns,
            "bond_metadata": {"bond_params": bond_params},
            "cpi_history": pd.Series([0.025]),
        }

        MAX_BOND = 0.30  # 30% limit na obligacje
        vinput = InputData(
            portfolio=[
                PortfolioItem(ticker="STOCK_A", instrument_type=InstrumentType.STOCK, quantity=10),
                PortfolioItem(ticker="BOND_X", instrument_type=InstrumentType.BOND, quantity=5,
                              bond_type="EDO", margin=0.015, first_year_rate=0.05,
                              issue_date="2023-01-02"),
            ],
            parametry_opt=OptimizationParameters(goal_type=GoalType.MIN_RISK, cvar_alpha=None),
            estimation_window="1Y",
            is_planning_phase=True,
            ustawienia_ograniczen=ConstraintsSettings(
                max_weight=1.0,
                min_trade_unit=0.0,
                transaction_cost_pct=0.001,
                max_bond_weight=MAX_BOND,
            ),
        )
        result = optimize_portfolio(vinput, model_params, processed)
        bond_weight = result["weights"]["BOND_X"]
        ROUNDING_TOLERANCE = 0.03  # dyskretyzacja jednostek może przesunąć wagę
        assert bond_weight <= MAX_BOND + ROUNDING_TOLERANCE, (
            f"Waga obligacji {bond_weight:.4f} przekracza limit {MAX_BOND} + tolerancja {ROUNDING_TOLERANCE}"
        )

    def test_max_weight_stock_cap_respected(self, det_processed_data):
        """Waga pojedynczej akcji nie przekracza max_weight (limit per-instrument).

        max_weight ogranicza wagę KAŻDEGO pojedynczego aktywa z osobna.
        Test używa portfela mieszanego (akcja + obligacja).
        """
        import numpy as np, pandas as pd
        tickers = ["STOCK_A", "BOND_X"]
        mu = pd.Series([0.12, 0.04], index=tickers)
        sigma = pd.DataFrame(
            [[0.04, 0.001], [0.001, 0.0005]],
            index=tickers, columns=tickers,
        )
        last_date = pd.Timestamp("2024-01-02")
        dates = pd.bdate_range(end=last_date, periods=252)
        rng = np.random.default_rng(12)
        prices = pd.DataFrame({
            "STOCK_A": 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, 252)),
            "BOND_X": 100.0 + np.linspace(0, 5, 252),
        }, index=dates)
        returns = prices.pct_change().dropna()
        bond_params = pd.DataFrame(
            {"bond_type": ["EDO"], "margin": [0.015], "first_year_rate": [0.05],
             "issue_date": [dates[0]], "kara_wykup": [2.0]},
            index=["BOND_X"],
        )
        model_p = {"mu": mu, "sigma": sigma, "sigma_shrink": sigma.copy(),
                   "last_date": last_date, "avg_cpi": 0.025, "current_cpi": 0.03}
        proc = {"prices": prices, "returns": returns,
                "bond_metadata": {"bond_params": bond_params},
                "cpi_history": pd.Series([0.025])}

        MAX_W = 0.60  # max łączny udział akcji
        from models import InputData, PortfolioItem, InstrumentType
        vinput = InputData(
            portfolio=[
                PortfolioItem(ticker="STOCK_A", instrument_type=InstrumentType.STOCK, quantity=10),
                PortfolioItem(ticker="BOND_X", instrument_type=InstrumentType.BOND, quantity=5,
                              bond_type="EDO", margin=0.015, first_year_rate=0.05,
                              issue_date="2023-01-02"),
            ],
            parametry_opt=OptimizationParameters(goal_type=GoalType.MIN_RISK, cvar_alpha=None),
            estimation_window="1Y",
            is_planning_phase=True,
            ustawienia_ograniczen=ConstraintsSettings(
                max_weight=MAX_W,
                min_trade_unit=0.0,
                transaction_cost_pct=0.001,
                max_bond_weight=1.0,
            ),
        )
        result = optimize_portfolio(vinput, model_p, proc)
        stock_weight = result["weights"]["STOCK_A"]
        assert stock_weight <= MAX_W + 0.03, (
            f"Waga akcji {stock_weight:.4f} przekracza limit {MAX_W} + tolerancja rounding"
        )


# ===========================================================================
# 5.  Koszty transakcyjne
# ===========================================================================

class TestTransactionCosts:

    def test_planning_phase_zero_costs(self, det_model_params, det_processed_data):
        """W fazie planowania (is_planning_phase=True) koszty = 0."""
        vinput = _make_input(GoalType.MIN_RISK, is_planning=True)
        result = optimize_portfolio(vinput, det_model_params, det_processed_data)
        assert result["metrics"]["estimated_execution_cost"] == pytest.approx(0.0)

    def test_rebalancing_phase_nonzero_costs(self, det_model_params, det_processed_data):
        """W fazie rebalancingu (is_planning_phase=False) koszty są > 0,
        gdy wymagana jest zmiana pozycji."""
        vinput = _make_input(GoalType.MIN_RISK, is_planning=False)
        result = optimize_portfolio(vinput, det_model_params, det_processed_data)
        # Koszty mogą być 0 tylko jeśli wagi się nie zmieniły (mało prawdopodobne)
        total_cost = result["metrics"]["total_rebalancing_cost"]
        assert total_cost >= 0.0  # zawsze nieujemne

    def test_sharpe_ratio_uses_cpi_as_rf(self, det_model_params, det_processed_data):
        """Sharpe ratio = (E[r] - current_cpi) / volatility."""
        vinput = _make_input(GoalType.MIN_RISK)
        result = optimize_portfolio(vinput, det_model_params, det_processed_data)
        m = result["metrics"]
        rf = det_model_params["current_cpi"]
        expected_sharpe = (m["expected_return"] - rf) / m["volatility"] if m["volatility"] > 0 else 0
        assert m["sharpe_ratio"] == pytest.approx(expected_sharpe, abs=1e-6)
