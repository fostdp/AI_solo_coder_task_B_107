from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from datetime import datetime


class PressureFlowDataPointSchema(BaseModel):
    pressure_kpa: float
    flow_rate_mls: float
    is_reliable: bool = True


class PressureFlowFitRequest(BaseModel):
    data_points: List[PressureFlowDataPointSchema]
    max_degree: int = 5
    ridge_alpha: float = 1e-3


class PressureFlowFitResponse(BaseModel):
    r_squared: float
    optimal_degree: int
    coefficients: List[float]


class PressureFlowPredictRequest(BaseModel):
    pressures_kpa: List[float]
    coefficients: List[float]
    optimal_degree: int


class PressureFlowPredictResponse(BaseModel):
    predictions: List[float]


class PressureFlowCurvePointSchema(BaseModel):
    pressure_kpa: float
    flow_rate_mls: float
    delamination_risk_pct: float


class PressureFlowOptimizeRequest(BaseModel):
    model_coefficients: List[float]
    optimal_degree: int
    r_squared: Optional[float] = None
    pressure_range: List[float] = Field(default_factory=lambda: [100.0, 500.0])
    target_radius_mm: Optional[float] = None
    max_allowable_risk_pct: float = 30.0
    delamination_threshold_pressure_kpa: float = 450.0
    max_flow_rate_mls: float = 500.0


class InjectionRateOptimizationResultSchema(BaseModel):
    optimal_pressure_kpa: float
    optimal_flow_rate_mls: float
    polynomial_degree: int
    polynomial_coefficients: List[float]
    r_squared: float
    secondary_delamination_risk_pct: float
    recommended_max_flow_mls: float
    pressure_flow_curve: List[PressureFlowCurvePointSchema]
    warning_message: Optional[str] = None
