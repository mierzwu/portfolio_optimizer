import cvxpy as cp
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Set, List
from models import InputData, GoalType, InstrumentType
import logging

logger = logging.getLogger(__name__)

def optimize_portfolio(
    validated_input: InputData, 
    model_params: Dict[str, Any],
    processed_data: Dict[str, Any],
    current_holdings: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Rozwiązuje problem optymalizacji portfela.
    
    :param validated_input: Dane wejściowe (ograniczenia, cel).
    :param model_params: Parametry modelu (mu, sigma).
    :param processed_data: Przetworzone dane (ceny, zwroty) - potrzebne do wyceny portfela.
    :param current_holdings: Aktualny stan posiadania {ticker: quantity}. Jeśli None, bierze z validated_input.
    :return: Wynik optymalizacji.
    """
    logger.debug("[Optimizer] Rozpoczynanie optymalizacji...")
    
    # 1. Przygotowanie danych
    mu = model_params['mu'].values
    # Używamy sigma_shrink jeśli dostępna, w przeciwnym razie zwykła sigma
    sigma = model_params.get('sigma_shrink', model_params['sigma']).values
    
    tickers = model_params['mu'].index.tolist()
    n_assets = len(tickers)

    # Dane historyczne do CVaR (jeśli dostępne)
    returns_df = processed_data.get('returns')
    if returns_df is not None:
        # Dopasowanie kolumn do kolejności w 'tickers'
        # model_params['mu'] ma indeks zgodny z tickers
        R = returns_df[tickers].values # Macierz T x N
        T_scenarios = R.shape[0]
    else:
        R = None
        T_scenarios = 0
    
    # Pobranie ostatnich cen do wyceny portfela
    last_date = model_params['last_date']
    current_prices = processed_data['prices'].loc[last_date]
    
    # Obliczenie obecnej wartości portfela i wag
    if current_holdings is None:
        current_holdings = {item.ticker: item.quantity for item in validated_input.portfolio}
        
    total_value = 0.0
    w_current_vals = np.zeros(n_assets)
    
    for i, ticker in enumerate(tickers):
        qty = current_holdings.get(ticker, 0.0)
        price = current_prices.get(ticker, 0.0)
        val = qty * price
        w_current_vals[i] = val
        total_value += val
        
    if total_value > 0:
        w_current = w_current_vals / total_value
    else:
        w_current = np.zeros(n_assets)
        # Jeśli portfel jest pusty (start), zakładamy gotówkę = np. 10000 (do symulacji) 
        # lub po prostu w_current = 0
        total_value = 10000.0 # Domyślna kwota startowa jeśli brak portfela

    # Wolny kapitał: powiększ pulę do optymalnej alokacji.
    # w_current zostaje przeliczone względem nowego total_value – wagi nie sumują
    # się do 1, więc optymalizator (sum(w)==1) wdroży gotówkę do aktywów.
    additional_cash = max(validated_input.additional_cash or 0.0, 0.0)
    if additional_cash > 0:
        total_value += additional_cash
        w_current = w_current_vals / total_value  # przeliczyć na nową bazę

    logger.debug(
        f"[Optimizer] Wartość portfela: {total_value:.2f} PLN"
        + (f" (w tym wolny kapita\u0142: {additional_cash:.2f} PLN)" if additional_cash > 0 else "")
    )
    
    # 2. Zmienne decyzyjne
    w = cp.Variable(n_assets)
    
    # 3. Ograniczenia
    constraints = [
        cp.sum(w) == 1, # Pełne zainwestowanie
    ]
    
    # Long only (zawsze)
    constraints.append(w >= 0)
        
    # Max weight (interpretowane jako maksymalny łączny udział akcji)
    max_w = validated_input.ustawienia_ograniczen.max_weight
    
    # 4. Model kosztów transakcyjnych z rozróżnieniem klas aktywów
    #
    # Akcje: koszt liniowy = (spread/2 + slippage + commission) * |zmiana wartości|
    #        + koszt kwadratowy (market impact)
    # Obligacje detaliczne: kara za przedterminowy wykup = penalty_per_unit * jedn_sprzedane
    #        (kara pobierana TYLKO przy redukcji pozycji; jednostka = 100 PLN nominału)

    exec_cfg = validated_input.execution_config
    base_cost_pct = validated_input.ustawienia_ograniczen.transaction_cost_pct
    impact_factor = exec_cfg.market_impact_factor
    constraints_cfg = validated_input.ustawienia_ograniczen

    # Zbiór tickerów obligacji
    bond_tickers: Set[str] = {
        item.ticker for item in validated_input.portfolio
        if item.instrument_type == InstrumentType.BOND
    }

    # Parametry obligacji z processed_data (zawierają kara_wykup z SQLite)
    bond_params_df = processed_data.get('bond_metadata', {}).get('bond_params', pd.DataFrame())

    # Ograniczenia klas aktywów
    # Explicit int() cast ensures CVXPY indexing uses Python ints, not numpy.int64
    stock_indices = [int(i) for i, t in enumerate(tickers) if t not in bond_tickers]
    stock_cap = 1.0
    stock_cap_effective = stock_cap
    if stock_indices:
        stock_cap = float(max_w)
        stock_cap_effective = stock_cap
        # Per-instrument cap: każda akcja z osobna nie przekracza max_weight
        for i in stock_indices:
            constraints.append(w[i] <= max_w)

    # Ograniczenie klasy obligacji: maks. udział konfigurowalny (domyślnie 60%)
    # Zapobiega pułapce zerowego ryzyka (solver alokuje 100% w obligacje z var≈0)
    bond_indices = [int(i) for i, t in enumerate(tickers) if t in bond_tickers]
    bond_cap_requested = float(constraints_cfg.max_bond_weight)
    bond_cap_effective = bond_cap_requested
    if bond_indices:
        # Jeśli oba limity klas są sprzeczne (suma maks. wag akcji + bond_cap < 1),
        # podnosimy limit obligacji do minimalnie wykonalnego poziomu.
        # Używamy łącznego maks. udziału akcji (n_akcji × max_w), nie per-instrument.
        max_total_stock_weight = min(1.0, len(stock_indices) * float(max_w))
        min_bond_needed = max(0.0, 1.0 - max_total_stock_weight)
        if bond_cap_effective < min_bond_needed:
            logger.warning(
                f"[Optimizer] Niewykonalne limity klas: max_stock_total={max_total_stock_weight:.3f}, "
                f"max_bond={bond_cap_effective:.3f}. Podnoszę max_bond do {min_bond_needed:.3f}."
            )
            bond_cap_effective = min_bond_needed

        # Dla portfela z samymi obligacjami limit obligacji musi pozwalać na sum(w)=1.
        if not stock_indices and bond_cap_effective < 1.0:
            logger.warning(
                f"[Optimizer] Portfel zawiera wyłącznie obligacje, podnoszę max_bond "
                f"z {bond_cap_effective:.3f} do 1.000."
            )
            bond_cap_effective = 1.0

        constraints.append(cp.sum(w[bond_indices]) <= bond_cap_effective)

    caps_auto_adjusted = (
        abs(stock_cap_effective - float(max_w)) > 1e-12
        or abs(bond_cap_effective - bond_cap_requested) > 1e-12
    )

    # Ograniczenie minimalnej stopy zwrotu (alternatywa dla max_bond_weight)
    # Wymusza zakup akcji gdy użytkownik nie chce ograniczenia wagowego
    if constraints_cfg.min_target_return is not None:
        constraints.append(mu @ w >= constraints_cfg.min_target_return)

    # Budujemy składniki kosztów jako listę wyrażeń CVXPY
    # Kara za przedterminowy wykup obligacji traktowana jako proporcjonalny koszt
    # (kara_PLN / 100 PLN nominału). Dokładna płaska kara naliczana post-hoc.
    cost_terms = []
    trade_vector = w - w_current  # wektor zmian wag
    is_planning_phase = validated_input.is_planning_phase

    for i, ticker in enumerate(tickers):
        if ticker in bond_tickers:
            # Pobierz karę z bazy danych (bond_params_df)
            kara = float(bond_params_df.loc[ticker, 'kara_wykup'])
            bond_cost_pct = kara / 100.0  # przeliczenie PLN/obligację na % nominału
            # Kara dotyczy wyłącznie redukcji pozycji obligacji
            units_sold = cp.pos(w_current[i] - w[i]) * total_value / max(float(current_prices.get(ticker, 1.0)), 1e-9)
            cost_terms.append(bond_cost_pct * units_sold * 100.0)
        else:
            # Akcja: spread + slippage + prowizja + market impact
            stock_linear_pct = base_cost_pct + (exec_cfg.spread_pct / 2.0) + exec_cfg.slippage_pct
            trade_val = cp.abs(trade_vector[i]) * total_value
            cost_terms.append(stock_linear_pct * trade_val)
            cost_terms.append(impact_factor * total_value * cp.square(trade_vector[i]))

    total_transaction_cost = sum(cost_terms) if cost_terms else cp.Constant(0.0)
    # Normalizacja kosztów do skali stopy zwrotu/CVaR, aby nie dominowały celu.
    normalized_tx_cost = total_transaction_cost / max(total_value, 1e-9)
    
    # Parametry celu
    goal_type = validated_input.parametry_opt.goal_type
    goal_value = validated_input.parametry_opt.goal_value
    cvar_alpha = validated_input.parametry_opt.cvar_alpha
    
    # Funkcja pomocnicza do definicji ryzyka (Wariancja lub CVaR)
    def get_risk_expression(w_var, constraints_list):
        if cvar_alpha is not None and R is not None and T_scenarios > 0:
            # CVaR Optimization (Rockafellar & Uryasev)
            # CVaR_alpha = zeta + 1/((1-alpha)*T) * sum(u)
            # u >= 0
            # u >= -R@w - zeta
            
            zeta = cp.Variable()
            u = cp.Variable(T_scenarios)
            
            constraints_list.append(u >= 0)
            # R @ w to wektor zwrotów portfela dla każdego scenariusza
            # Strata to -(R @ w)
            constraints_list.append(u >= -R @ w_var - zeta)
            
            cvar_term = zeta + (1.0 / ((1.0 - cvar_alpha) * T_scenarios)) * cp.sum(u)
            logger.debug(f"[Optimizer] Używanie CVaR ({cvar_alpha}) jako miary ryzyka.")
            return cvar_term
        else:
            # Mean-Variance
            if cvar_alpha is not None:
                logger.warning("[Optimizer] cvar_alpha podane, ale brak danych historycznych (R). Używanie wariancji.")
            return cp.quad_form(w_var, sigma)

    # Konfiguracja problemu
    if goal_type == GoalType.MIN_RISK:
        # Minimalizacja ryzyka (CVaR lub Wariancja)
        risk = get_risk_expression(w, constraints)
        if is_planning_phase:
            objective = cp.Minimize(risk)
        else:
            # Mała waga kosztów: preferuj silniej redukcję ryzyka/CVaR niż minimalny obrót.
            objective = cp.Minimize(risk + 0.05 * normalized_tx_cost)
        
        if goal_value is not None:
            constraints.append(mu @ w >= goal_value)
            
    elif goal_type == GoalType.MAX_RETURN:
        # Maksymalizacja zwrotu przy zadanym maksymalnym ryzyku
        ret = mu @ w
        if is_planning_phase:
            objective = cp.Maximize(ret)
        else:
            objective = cp.Maximize(ret - 0.05 * normalized_tx_cost)
        
        if goal_value is not None:
            risk = get_risk_expression(w, constraints)
            constraints.append(risk <= goal_value)
            
    # 5. Rozwiązanie z mechanizmem Fallback
    logger.debug("[Optimizer] Rozwiązywanie problemu optymalizacji...")

    # Wariancja jako fallback ryzyka (nie wymaga zmiennych pomocniczych CVaR)
    variance_risk = cp.quad_form(w, sigma)

    def _build_fallback_objective(use_variance: bool = True):
        """Buduje cel bez zmiennych pomocniczych CVaR (bezpieczny fallback)."""
        if goal_type == GoalType.MIN_RISK:
            return cp.Minimize(variance_risk)
        else:
            return cp.Maximize(mu @ w)

    status_trace: List[str] = []

    def solve_with_constraints(current_constraints, description="Podstawowe ograniczenia", fallback_obj=None):
        obj = fallback_obj if fallback_obj is not None else objective
        prob = cp.Problem(obj, current_constraints)
        try:
            # Preferujemy SCS (LP-friendly, dobry dla CVaR) z wyższym limitem iteracji
            prob.solve(solver=cp.SCS, max_iters=50000, eps=1e-5)
            status_trace.append(f"{description}:SCS={prob.status}")
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return prob
            # Fallback do CLARABEL jeśli SCS nie dał wyniku
            prob2 = cp.Problem(obj, current_constraints)
            prob2.solve(solver=cp.CLARABEL)
            status_trace.append(f"{description}:CLARABEL={prob2.status}")
            return prob2
        except Exception as e:
            logger.warning(f"[Optimizer] Wyjątek solvera ({description}): {e}")
            status_trace.append(f"{description}:EXCEPTION={e}")
            return None

    # Próba 1: Oryginalne ograniczenia
    prob = solve_with_constraints(constraints, "Próba 1")

    # Logika Fallback
    if prob is None or prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        logger.warning(f"[Optimizer] Nie znaleziono rozwiązania (Status: {prob.status if prob else 'Error'}). Uruchamianie procedury naprawczej...")

        # Budujemy fallback objective (wariancja, bez zmiennych pomocniczych CVaR)
        fb_obj = _build_fallback_objective()

        # Krok 1: Relaksacja max_weight (jeśli jest restrykcyjna)
        if max_w < 1.0:
            logger.debug("[Optimizer] Fallback Krok 1: Poluzowanie limitu akcji do 1.0")
            base_constraints = [cp.sum(w) == 1]
            base_constraints.append(w >= 0)
            if goal_type == GoalType.MIN_RISK and goal_value is not None:
                base_constraints.append(mu @ w >= goal_value)
            elif goal_type == GoalType.MAX_RETURN and goal_value is not None:
                base_constraints.append(variance_risk <= goal_value)
            if bond_indices:
                base_constraints.append(cp.sum(w[bond_indices]) <= bond_cap_effective)
            prob = solve_with_constraints(base_constraints, "Relax Max Weight", fallback_obj=fb_obj)

        # Krok 2: Relaksacja celu (Target Return / Risk)
        if prob is None or prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            logger.debug("[Optimizer] Fallback Krok 2: Usunięcie ograniczeń celu (Target Return/Risk)")
            fallback_constraints = [cp.sum(w) == 1]
            fallback_constraints.append(w >= 0)
            if bond_indices:
                fallback_constraints.append(cp.sum(w[bond_indices]) <= bond_cap_effective)
            prob = solve_with_constraints(fallback_constraints, "Relax Goal", fallback_obj=fb_obj)
            
    if prob is None or prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        trace_txt = " | ".join(status_trace) if status_trace else "brak statusów"
        msg = f"[Optimizer] INFEASIBLE/FAILED. Nie znaleziono rozwiązania po relaksacjach. Statusy: {trace_txt}"
        logger.error(msg)
        if total_value > 0:
            w_fallback = w_current.copy()
        else:
            w_fallback = np.ones(n_assets) / max(n_assets, 1)

        expected_return = float(np.dot(w_fallback, mu))
        volatility = float(np.sqrt(np.dot(w_fallback.T, np.dot(sigma, w_fallback))))

        return {
            "weights": pd.Series(w_fallback, index=tickers),
            "metrics": {
                "expected_return": expected_return,
                "volatility": volatility,
                "sharpe_ratio": (expected_return - model_params.get('current_cpi', 0.0)) / volatility if volatility > 0 else 0,
                "estimated_execution_cost": 0.0,
                "total_rebalancing_cost": 0.0,
                "cost_impact_bps": 0.0,
                "stock_cap_requested": float(max_w),
                "stock_cap_effective": float(stock_cap_effective),
                "bond_cap_requested": float(bond_cap_requested),
                "bond_cap_effective": float(bond_cap_effective),
                "caps_auto_adjusted": bool(caps_auto_adjusted),
                "solver_status": "fallback_current_portfolio",
                "solver_message": msg,
            },
            "transactions": [],
        }
        
    logger.debug(f"[Optimizer] Znaleziono rozwiązanie. Status: {prob.status}")
    w_opt = w.value
    
    # 6. Post-processing (Min Trade Unit)
    # Heurystyka: Obliczamy docelowe kwoty, zaokrąglamy do pełnych jednostek (akcji/obligacji)
    # i przeliczamy wagi.
    
    logger.debug("[Optimizer] Post-processing (Min Trade Unit)...")
    min_trade_unit = validated_input.ustawienia_ograniczen.min_trade_unit
    
    # Kwoty docelowe
    target_values = w_opt * total_value
    
    # Liczba jednostek (teoretyczna)
    # Dla akcji: Value / Price
    # Dla obligacji: Value / Price (Price jest w % nominału czy kwotowo? W preprocessingu liczyliśmy kwotowo)
    
    units = np.zeros(n_assets)
    final_values = np.zeros(n_assets)
    
    for i, ticker in enumerate(tickers):
        price = current_prices.get(ticker, 0.0)
        if price > 0:
            # Teoretyczna liczba jednostek
            raw_units = target_values[i] / price
            current_qty = current_holdings.get(ticker, 0.0)
            proposed_units = round(raw_units)
            delta_units = proposed_units - current_qty
            delta_value = abs(delta_units) * price

            # Minimalna jednostka transakcji dotyczy zmiany pozycji, nie całej pozycji docelowej.
            if 0 < delta_value < min_trade_unit:
                final_units = current_qty
            else:
                final_units = proposed_units
                
            units[i] = final_units
            final_values[i] = final_units * price
        else:
            final_values[i] = 0
            
    # Nowe wagi po zaokrągleniu
    new_total_value = np.sum(final_values)
    if new_total_value > 0:
        w_final = final_values / new_total_value
    else:
        w_final = w_opt # Fallback jeśli wszystko wyzerowano
        
    # 7. Wyniki
    expected_return = np.dot(w_final, mu)
    volatility = np.sqrt(np.dot(w_final.T, np.dot(sigma, w_final)))
    
    # Propozycja transakcji z szacunkiem kosztów
    transactions = []
    total_estimated_cost = 0.0

    for i, ticker in enumerate(tickers):
        current_qty = current_holdings.get(ticker, 0.0)
        target_qty = units[i]
        diff = target_qty - current_qty

        if diff != 0:
            price = current_prices.get(ticker, 0.0)
            trade_val = abs(diff) * price
            is_bond = ticker in bond_tickers

            if is_bond:
                if diff < 0:
                    # Redukcja pozycji obligacji – dokładna płaska kara z bazy danych
                    units_sold = abs(diff)
                    kara = float(bond_params_df.loc[ticker, 'kara_wykup'])
                    item_cost = 0.0 if is_planning_phase else (kara * units_sold)
                else:
                    item_cost = 0.0  # Kupno obligacji detalicznej bez kosztu transakcyjnego
            else:
                # Akcje: spread + slippage + prowizja + market impact
                stock_linear_pct = base_cost_pct + (exec_cfg.spread_pct / 2.0) + exec_cfg.slippage_pct
                lin_cost = trade_val * stock_linear_pct
                quad_cost = (trade_val ** 2) * impact_factor
                item_cost = 0.0 if is_planning_phase else (lin_cost + quad_cost)

            total_estimated_cost += item_cost

            transactions.append({
                "ticker": ticker,
                "action": "BUY" if diff > 0 else "SELL",
                "quantity": abs(diff),
                "price_est": price,
                "est_cost": item_cost,
            })
            
    cash_remainder = total_value - new_total_value
    # Ujemna reszta oznacza, że zaokrąglenie w górę "przekroczyło" budżet —
    # traktujemy nadwyżkę jako dodatkowy koszt rebalancingu.
    if cash_remainder < 0:
        total_estimated_cost += abs(cash_remainder)
        cash_remainder = 0.0
    metrics = {
        "budget": total_value,
        "total_value": new_total_value,
        "cash_remainder": cash_remainder,
        "expected_return": expected_return,
        "sharpe_ratio": (expected_return - model_params.get('current_cpi', 0.0)) / volatility if volatility > 0 else 0,
        "volatility": volatility,
        "cost_impact_bps": (total_estimated_cost / new_total_value * 10000) if new_total_value > 0 else 0,
        "total_rebalancing_cost": total_estimated_cost,
        "estimated_execution_cost": total_estimated_cost,
        "stock_cap_requested": float(max_w),
        "stock_cap_effective": float(stock_cap_effective),
        "bond_cap_requested": float(bond_cap_requested),
        "bond_cap_effective": float(bond_cap_effective),
        "caps_auto_adjusted": bool(caps_auto_adjusted),
    }

    return {
        "weights": pd.Series(w_final, index=tickers),
        "target_quantities": {ticker: units[i] for i, ticker in enumerate(tickers)},
        "metrics": metrics,
        "transactions": transactions
    }
