import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def estimate_avg_cpi(cpi_history: pd.Series) -> float:
    """
    Wyznacza średnią historyczną CPI (podejście Historical Simulation).

    Zastępuje skomplikowany proces Ornsteina-Uhlenbecka prostym estymatorem
    historycznej średniej inflacji. Prognoza przyszłych kuponów obligacji
    wyznaczana jest deterministycznie jako avg_cpi + margin.

    :param cpi_history: Szereg rocznych odczytów CPI (wartości ułamkowe, np. 0.035).
    :return: Średnioroczny CPI jako liczba zmiennoprzecinkowa.
    """
    if cpi_history.empty:
        logger.warning("[Estimation] Brak danych CPI – używam wartości domyślnej 2.5%.")
        return 0.025
    avg = float(cpi_history.dropna().mean())
    logger.debug(f"[Estimation] Średnioroczny CPI (historyczny): {avg:.4f}")
    return avg


def calculate_shrinkage_mu(returns: pd.DataFrame) -> pd.Series:
    """
    Oblicza oczekiwane zwroty metodą James-Stein Shrinkage (ku średniej globalnej).
    """
    T, N = returns.shape
    mu_hist = returns.mean()
    sigma_hist = returns.cov()
    
    # Target: Grand Mean (średnia ze wszystkich aktywów)
    mu_target = mu_hist.mean()
    
    # Wariancja estymacji średniej (dla każdego aktywa: sigma^2 / T)
    # Używamy średniej wariancji estymacji jako miary "szumu"
    avg_estimation_variance = np.diag(sigma_hist).mean() / T
    
    # Wariancja między średnimi (cross-sectional) - miara "sygnału"
    variance_of_means = mu_hist.var()
    
    if variance_of_means == 0 or np.isnan(variance_of_means):
        lambda_ = 1.0
    else:
        # Lambda = Noise / (Noise + Signal)
        lambda_ = avg_estimation_variance / (avg_estimation_variance + variance_of_means)
        
    # Clip lambda [0, 1]
    lambda_ = max(0.0, min(1.0, lambda_))
    
    logger.debug(f"[Estimation] James-Stein Shrinkage Lambda: {lambda_:.4f} (Target Daily: {mu_target:.6f})")
    
    mu_shrink = (1 - lambda_) * mu_hist + lambda_ * mu_target
    return mu_shrink


def compute_bond_forward_mu(
    bond_params_df: pd.DataFrame,
    current_cpi: float,
    investment_horizon_years: int,
    today=None,
) -> pd.Series:
    """
    Oblicza prospektywny roczny oczekiwany zwrot dla obligacji detalicznych
    na podstawie bieżącego CPI + marża (oderwany od historii).

    Wzór:
      - Rok 1: first_year_rate (stała)
      - Lata 2..H: current_cpi + margin
    Przy wcześniejszym wykupie (H < 10): odejmuje karę za wykup.
    Efektywna stopa roczna = V(H)^(1/H) - 1.
    """
    MATURITY_YEARS = 10  # EDO i COI to obligacje 10-letnie
    result: dict = {}
    for ticker in bond_params_df.index:
        row = bond_params_df.loc[ticker]
        margin = float(row.get('margin', 0.015))
        first_year_rate = float(row.get('first_year_rate', 0.0))
        # kara_wykup w PLN na 100 PLN nominału → jako ułamek wartości nominalnej
        kara_fraction = float(row.get('kara_wykup', 2.0)) / 100.0

        H = min(investment_horizon_years, MATURITY_YEARS)
        if H <= 0:
            result[ticker] = first_year_rate
            continue

        recurring_rate = current_cpi + margin  # oprocentowanie lat 2..H

        # Wartość skumulowana 1 jednostki nominału po H latach
        if H == 1:
            accrued = 1.0 + first_year_rate
        else:
            accrued = (1.0 + first_year_rate) * (1.0 + recurring_rate) ** (H - 1)

        # Kara za przedterminowy wykup (jeśli horyzont < termin zapadalności)
        if investment_horizon_years < MATURITY_YEARS:
            accrued -= kara_fraction

        # Efektywna annualizowana stopa zwrotu
        effective_annual = max(accrued ** (1.0 / H) - 1.0, 0.0)
        result[ticker] = effective_annual
        logger.debug(
            f"[Estimation] Bond μ {ticker}: CPI={current_cpi:.4f}, margin={margin:.4f}, "
            f"H={H}y, effective_annual={effective_annual:.4f}"
        )
    return pd.Series(result)


def estimate_params(D2_processed: Dict[str, Any], estimation_window: str = "3Y", investment_horizon_years: int = 5) -> Dict[str, Any]:
    """
    Estymuje parametry modelu (mu, Sigma, kalibracja CPI).

    Zmiany względem poprzedniej wersji:
      - Usunięto Duration i Convexity (nieadekwatne dla obligacji detalicznych).
      - Kalibracja modelu stóp zastąpiona kalibracją procesu CPI (Ornstein-Uhlenbeck).
      - Symulacje ścieżek CPI zastępują symulacje stopy wolnej od ryzyka.
    """
    logger.debug("[Estimation] Rozpoczynanie estymacji parametrów...")

    returns = D2_processed['returns']
    prices = D2_processed['prices']
    cpi_history: pd.Series = D2_processed.get('cpi_history', pd.Series(dtype=float))

    if returns.empty:
        logger.error("[Estimation] Błąd: Brak danych zwrotów do estymacji.")
        return {}

    # 1. Średnie zwroty (mu) - annualizowane z Shrinkage
    logger.debug("[Estimation] Obliczanie oczekiwanych zwrotów (James-Stein Shrinkage)...")
    mu_daily = calculate_shrinkage_mu(returns)
    mu_annual = mu_daily * 252

    # 2. Macierz kowariancji (Sigma) - empiryczna annualizowana
    sigma_daily = returns.cov()
    sigma_annual = sigma_daily * 252

    # 3. Ledoit-Wolf Shrinkage
    logger.debug("[Estimation] Obliczanie Ledoit-Wolf shrinkage...")
    try:
        lw = LedoitWolf()
        lw.fit(returns)
        sigma_shrink_daily = lw.covariance_
        sigma_shrink_annual = sigma_shrink_daily * 252
        sigma_shrink_df = pd.DataFrame(
            sigma_shrink_annual,
            index=returns.columns,
            columns=returns.columns
        )
    except Exception as e:
        logger.warning(f"[Estimation] Błąd Ledoit-Wolf: {e}. Używam zwykłej kowariancji.")
        sigma_shrink_df = sigma_annual

    # 4. Walidacja (Condition Number)
    try:
        cond_number = np.linalg.cond(sigma_shrink_df)
        logger.debug(f"[Estimation] Condition number macierzy (Shrinkage): {cond_number:.2f}")
        if cond_number > 1000:
            logger.warning("[Estimation] Ostrzeżenie: Wysoki wskaźnik uwarunkowania macierzy!")
    except Exception:
        logger.warning("[Estimation] Nie udało się obliczyć condition number.")

    # 5. Szacowanie CPI (średnia historyczna – Historical Simulation)
    logger.debug("[Estimation] Szacowanie średniego CPI (metoda historyczna)...")
    avg_cpi = estimate_avg_cpi(cpi_history)
    # Bieżący CPI: ostatni dostępny odczyt (lub średnia, jeśli brak)
    current_cpi = float(cpi_history.iloc[-1]) if not cpi_history.empty else avg_cpi
    logger.debug(f"[Estimation] Bieżący CPI: {current_cpi:.4f}, Średni historyczny CPI: {avg_cpi:.4f}")

    last_date = prices.index[-1]

    # 6. Nadpisanie μ dla obligacji detalicznych: CPI + marża (prospektywnie)
    bond_params_df: pd.DataFrame = D2_processed.get('bond_metadata', {}).get('bond_params', pd.DataFrame())
    if not bond_params_df.empty:
        logger.debug("[Estimation] Wstrzykiwanie prospektywnego μ dla obligacji detalicznych...")
        bond_mu = compute_bond_forward_mu(
            bond_params_df, current_cpi, investment_horizon_years, today=last_date
        )
        for ticker in bond_mu.index:
            if ticker in mu_annual.index:
                mu_annual[ticker] = bond_mu[ticker]

    model_params = {
        "mu": mu_annual,
        "sigma": sigma_annual,
        "sigma_shrink": sigma_shrink_df,
        "last_date": last_date,
        "avg_cpi": avg_cpi,
        "current_cpi": current_cpi,
    }

    return model_params

if __name__ == "__main__":
    # Test
    pass
