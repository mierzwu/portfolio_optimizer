from typing import List, Optional, Dict, Any
from datetime import date
from enum import Enum
from pydantic import BaseModel, Field

# Enums based on the description
class InstrumentType(str, Enum):
    STOCK = "stock"
    BOND = "bond"

class BondType(str, Enum):
    EDO = "EDO"  # 10-letnie obligacje emerytalne (kapitalizacja roczna)
    COI = "COI"  # 4-letnie obligacje oszczędnościowe (roczna wypłata kuponu)

class GoalType(str, Enum):
    MIN_RISK = "min_risk"
    MAX_RETURN = "max_return"

# Sub-models
class PortfolioItem(BaseModel):
    ticker: str
    instrument_type: InstrumentType
    quantity: float

    # Identyfikator obligacji detalicznej (np. 'EDO0135') – używany zamiast tickera dla obligacji
    symbol_emisji: Optional[str] = None

    # Pola specyficzne dla obligacji detalicznych (EDO/COI)
    bond_type: Optional[BondType] = None          # Typ obligacji: 'EDO' lub 'COI'
    margin: Optional[float] = None                 # Marża ponad CPI (np. 0.015 = 1.5%)
    first_year_rate: Optional[float] = None        # Stałe oprocentowanie w 1. roku
    issue_date: Optional[date] = None              # Data emisji / nabycia obligacji
    date_acquired: Optional[date] = None
    price_acquired: Optional[float] = None

class OptimizationParameters(BaseModel):
    goal_type: GoalType
    goal_value: Optional[float] = None
    cvar_alpha: Optional[float] = Field(None, ge=0, le=1)

class ConstraintsSettings(BaseModel):
    max_weight: float = Field(..., ge=0, le=1)
    min_trade_unit: float
    transaction_cost_pct: float
    max_bond_weight: float = Field(0.60, ge=0, le=1)  # Maks. łączny udział obligacji (anti-zero-risk-trap)
    min_target_return: Optional[float] = None  # Min. roczna stopa zwrotu portfela (wymusza zakup akcji)

class ExecutionConfig(BaseModel):
    spread_pct: float = 0.002                    # Bid-Ask spread dla akcji (np. 0.2%)
    slippage_pct: float = 0.001                  # Slippage dla akcji (np. 0.1%)
    market_impact_factor: float = 1e-7           # Współczynnik wpływu na rynek (kwadratowy)

class DataPolicy(BaseModel):
    min_observations: int = 100
    use_cache: bool = True
    cache_dir: str = ".cache"
    data_version: str = "v1"

# Main Model
class InputData(BaseModel):
    portfolio: List[PortfolioItem]
    parametry_opt: OptimizationParameters
    estimation_window: str
    investment_horizon_years: int = Field(5, ge=1, le=40)
    data_sources: List[str] = ["GPW"] #w domyśle GPW, ale można dodać inne źródła w przyszłości
    data_policy: DataPolicy = DataPolicy() # Default policy
    execution_config: ExecutionConfig = ExecutionConfig() # Default execution settings
    start_date: Optional[date] = None # Jeśli None, zostanie ustawione na dzisiaj
    is_planning_phase: bool = False # Flaga: True = planowanie (brak kosztów/transakcji), False = rebalancing
    ustawienia_ograniczen: ConstraintsSettings
    
    additional_cash: Optional[float] = None  # Dodatkowe środki PLN do alokacji w strategii

if __name__ == "__main__":
    # Prosty test walidacji przy użyciu pliku example_input.json
    from pathlib import Path
    
    input_path = Path("example_input.json")
    if input_path.exists():
        try:
            with open(input_path, 'r') as f:
                data = InputData.model_validate_json(f.read())
            print("Test modelu: JSON poprawny.")
            print(f"Liczba pozycji w portfelu: {len(data.portfolio)}")
        except Exception as e:
            print(f"Błąd walidacji: {e}")
    else:
        print("Brak pliku example_input.json do testów.")
