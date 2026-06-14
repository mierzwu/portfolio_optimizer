import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import copy
import logging

from models import InputData

logger = logging.getLogger(__name__)

def run_monte_carlo_simulation(
    model_params: Dict[str, Any],
    weights: pd.Series,
    processed_data: Optional[Dict[str, Any]] = None,
    n_simulations: int = 2000,
    time_horizon_days: int = 252,
    cvar_alpha: Optional[float] = 0.95
) -> Dict[str, Any]:
    """
    Symulacja Monte Carlo oparta o dzienne zwroty arytmetyczne z MVN.
    """
    if cvar_alpha is None:
        cvar_alpha = 0.95
    cvar_alpha = float(cvar_alpha)
    if not (0.0 < cvar_alpha < 1.0):
        raise ValueError("cvar_alpha musi należeć do przedziału (0, 1).")

    logger.debug(
        f"[Simulation] Rozpoczynanie Monte Carlo ({n_simulations} symulacji, "
        f"{time_horizon_days} dni)..."
    )

    # Use the canonical tickers order from model_params (same as optimizer)
    tickers = model_params['mu'].index.tolist()
    # Reindex weights to exactly match model_params order (ensures MC and optimizer see same mu)
    weights_aligned = weights.reindex(tickers).fillna(0.0)
    mu_annual = model_params['mu'].values
    sigma_annual = model_params.get('sigma_shrink', model_params['sigma']).values

    # 1. Scale annual parameters to daily
    mu_daily = mu_annual / 252.0
    cov_daily = sigma_annual / 252.0
    trading_days = time_horizon_days  # already = years * 252

    # Stabilizacja numeryczna (PSD) – wymagana przez multivariate_normal
    eigval, eigvec = np.linalg.eigh(cov_daily)
    eigval[eigval < 1e-12] = 1e-12
    cov_daily = eigvec @ np.diag(eigval) @ eigvec.T

    # 2. GBM: log-returns with Ito drift correction mu - 0.5*sigma^2
    #    log_r_i ~ MVN((mu_i - 0.5*var_i)*dt, Sigma*dt)
    #    Arithmetic return per asset: r_i = exp(log_r_i) - 1
    log_drift_daily = mu_daily - 0.5 * np.diag(cov_daily)
    np.random.seed(42)
    log_returns = np.random.multivariate_normal(log_drift_daily, cov_daily, (n_simulations, trading_days))
    simulated_returns = np.exp(log_returns) - 1

    # 3. Calculate daily portfolio returns: shape (n_simulations, trading_days)
    optimal_weights_array = weights_aligned.values
    portfolio_daily = np.dot(simulated_returns, optimal_weights_array)

    # 4. Compound arithmetic returns to get cumulative return per simulation
    cumulative_returns = np.prod(1 + portfolio_daily, axis=1) - 1

    # Build paths for chart (cumulative wealth index per simulation)
    paths = np.cumprod(1 + portfolio_daily, axis=1)

    # 5. Annualised arithmetic expected return – consistent with optimizer's w⊤μ.
    #    Geometric compounding of cumulative returns inflates a 35% arithmetic
    #    drift to exp(0.35)−1 ≈ 42%.  Using the arithmetic daily mean × 252
    #    recovers exactly w⊤μ_annual (the number the optimizer reports).
    annualized_expected_return = float(np.mean(portfolio_daily)) * 252

    # 6. VaR / CVaR from the left tail of the cumulative-return distribution.
    years = trading_days / 252.0
    # Exact tail size: floor((1-alpha) * N) worst simulations.
    tail_count = max(1, int(np.floor((1.0 - cvar_alpha) * n_simulations)))
    sorted_cumret = np.sort(cumulative_returns)            # ascending (worst first)
    var_raw  = float(sorted_cumret[tail_count - 1])        # worst boundary = VaR level
    cvar_raw = float(np.mean(sorted_cumret[:tail_count]))  # mean of tail ≤ var_raw → CVaR

    # Annualise both risk measures
    var_ann  = float((1 + var_raw)  ** (1.0 / years) - 1)
    cvar_ann = float((1 + cvar_raw) ** (1.0 / years) - 1)
    # Convention: loss = positive number (matches GUI labels "próg straty" / "śr. strata").
    # max(0, …) applied to BOTH so signs are always unified:
    #   VaR  > 0  → tail shows a loss of that magnitude
    #   VaR  = 0  → even the worst-percentile scenario is a gain (not a calibration artefact
    #               now that the GBM drift is correctly Ito-corrected)
    # CVaR ≥ VaR guaranteed because −cvar_ann ≥ −var_ann (cvar_raw ≤ var_raw by definition).
    VaR  = max(0.0, -var_ann)
    CVaR = max(0.0, -cvar_ann)

    logger.info(f"[Simulation] Oczekiwany zwrot roczny (MC): {annualized_expected_return:.2%}")
    logger.info(f"[Simulation] VaR {cvar_alpha:.0%} roczny (strata): {VaR:.2%}")
    logger.info(f"[Simulation] CVaR {cvar_alpha:.0%} roczny (strata): {CVaR:.2%}")

    return {
        "paths": paths,
        "metrics": {
            "expected_return": annualized_expected_return,
            f"VaR_{int(cvar_alpha * 100)}": VaR,
            f"CVaR_{int(cvar_alpha * 100)}": CVaR,
        }
    }

def run_stress_test(
    validated_input: InputData,
    processed_data: Dict[str, Any],
    base_model_params: Dict[str, Any],
    optimal_weights: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Stress-testy na STAŁYCH wagach portfela (bez ponownej optymalizacji).

    Każdy scenariusz oblicza oczekiwany zwrot, zmienność i Sharpe dla tych
    samych wag co optymalny portfel, ale pod zszokowanymi parametrami mu/Sigma.
    Dzięki temu scenariusz "Base" daje dokładnie te same wartości co zakładka
    Portfel – różnice w pozostałych scenariuszach to czysty wpływ szoku.

    Scenariusze:
      - Base               : bez zmian
      - CPI +5%            : skok inflacji o 5 p.p.
      - Stocks -20%        : nagły spadek cen akcji o 20%, wzrost zmienności ×1.5
      - High Volatility x2 : podwojenie całej macierzy kowariancji
    """
    logger.debug("[StressTest] Rozpoczynanie stress-testów (fixed-weight evaluation)...")

    scenarios = {
        "Base": {},
        "CPI +5%": {"cpi_shock": 0.05},
        "Stocks -20%": {"stock_shock": -0.20, "stock_vol_scale": 1.5},
        "High Volatility (x2)": {"vol_shock": 2.0},
    }

    # Canonical ticker order from model_params (same as optimizer)
    tickers = base_model_params['mu'].index.tolist()

    # Fixed weights to evaluate under each scenario
    if optimal_weights is not None:
        w = optimal_weights.reindex(tickers).fillna(0.0).values
    else:
        # Fallback: equal-weight portfolio (used in unit tests without optimizer)
        n = max(len(tickers), 1)
        w = np.ones(n) / n
    weights_series = pd.Series(w, index=tickers)

    results = {}

    for name, shocks in scenarios.items():
        logger.debug(f"[StressTest] Scenariusz: {name}")

        params = copy.deepcopy(base_model_params)

        if "cpi_shock" in shocks:
            shock = shocks["cpi_shock"]
            params['current_cpi'] = params.get('current_cpi', 0.025) + shock
            params['avg_cpi'] = params.get('avg_cpi', 0.025) + shock
            # COI/EDO: coupon = CPI + margin, so nominal return rises 1:1 with CPI shock.
            # Stocks and non-indexed bonds: higher inflation lifts discount rates,
            # reducing real/nominal returns (standard approximation: −0.5 × shock).
            bond_params_df_cpi = (
                processed_data.get('bond_metadata', {}).get('bond_params', pd.DataFrame())
            )
            inflation_indexed: set = set()
            if not bond_params_df_cpi.empty and 'bond_type' in bond_params_df_cpi.columns:
                inflation_indexed = set(
                    bond_params_df_cpi.index[
                        bond_params_df_cpi['bond_type'].isin(['COI', 'EDO'])
                    ]
                )
            for ticker in params['mu'].index:
                if ticker in inflation_indexed:
                    params['mu'][ticker] += shock          # CPI↑ → coupon↑
                else:
                    params['mu'][ticker] -= shock * 0.5   # CPI↑ → real return↓

        if "stock_shock" in shocks:
            bond_params_df = processed_data.get('bond_metadata', {}).get('bond_params', pd.DataFrame())
            stock_tickers = [
                t for t in params['mu'].index
                if bond_params_df.empty or t not in bond_params_df.index
            ]
            for ticker in stock_tickers:
                params['mu'][ticker] += shocks['stock_shock']
            if stock_tickers and 'stock_vol_scale' in shocks:
                scale = shocks['stock_vol_scale']
                for key in ('sigma', 'sigma_shrink'):
                    if key in params:
                        params[key].loc[stock_tickers, stock_tickers] = (
                            params[key].loc[stock_tickers, stock_tickers] * scale
                        )

        if "vol_shock" in shocks:
            params['sigma'] = params['sigma'] * shocks['vol_shock']
            if 'sigma_shrink' in params:
                params['sigma_shrink'] = params['sigma_shrink'] * shocks['vol_shock']

        # Evaluate fixed weights under (possibly shocked) parameters
        mu_vec = params['mu'].reindex(tickers).values
        sig_mat = (
            params.get('sigma_shrink', params['sigma'])
            .reindex(tickers)
            .reindex(tickers, axis=1)
            .values
        )
        # Ensure PSD
        eigval, eigvec = np.linalg.eigh(sig_mat)
        eigval[eigval < 1e-12] = 1e-12
        sig_mat = eigvec @ np.diag(eigval) @ eigvec.T

        exp_ret  = float(np.dot(w, mu_vec))
        port_var = float(np.dot(w.T, np.dot(sig_mat, w)))
        port_vol = float(np.sqrt(max(0.0, port_var)))
        rf       = params.get('current_cpi', 0.025)
        sharpe   = (exp_ret - rf) / port_vol if port_vol > 1e-12 else 0.0

        results[name] = {
            "weights": weights_series,
            "metrics": {
                "expected_return": exp_ret,
                "volatility":      port_vol,
                "sharpe_ratio":    sharpe,
            },
        }
        logger.debug(
            f"[StressTest] {name}: r={exp_ret:.2%}, σ={port_vol:.2%}, Sharpe={sharpe:.3f}"
        )

    return results

def backtest_and_simulate(
    validated_input: InputData,
    processed_data: Dict[str, Any],
    optimization_result: Dict[str, Any],
    model_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Główna funkcja modułu.
    """
    
    # 1. Monte Carlo (na podstawie obecnego portfela optymalnego)
    mc_results = {}
    if optimization_result:
        weights = optimization_result['weights']
        cvar_alpha = validated_input.parametry_opt.cvar_alpha
        time_horizon_days = max(1, int(validated_input.investment_horizon_years * 252))
        mc_results = run_monte_carlo_simulation(
            model_params,
            weights,
            processed_data=processed_data,
            n_simulations=2000,
            time_horizon_days=time_horizon_days,
            cvar_alpha=cvar_alpha,
        )

    # 2. Stress Tests – evaluate optimal weights under shocked parameters
    optimal_weights = optimization_result.get('weights') if optimization_result else None
    stress_results = run_stress_test(validated_input, processed_data, model_params, optimal_weights)
    
    # 3. Walk-Forward Backtest (opcjonalnie, jeśli dane pozwalają)
    # Tutaj pomijamy pełny walk-forward w demo, bo mamy tylko 3Y danych, 
    # które zostały w całości użyte do estymacji.
    # Można by zaimplementować "In-Sample Backtest" (symulacja na danych historycznych)
    
    return {
        "monte_carlo": mc_results,
        "stress_tests": stress_results
    }
