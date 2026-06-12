from pydantic import BaseModel
from typing import List, Optional


class DryingShrinkagePredictionRequest(BaseModel):
    formulation_id: str = "sintered_stone_powder_ps_basic"
    ambient_temperature_c: float = 20.0
    ambient_humidity_pct: float = 60.0
    constraint_factor: float = 0.7
    wall_thickness_mm: float = 50.0
    prediction_horizon_days: float = 180.0
    initial_curing_days: float = 3.0


class ShrinkageTimeCurve(BaseModel):
    days: List[float]
    shrinkage_strain: List[float]
    tensile_stress_mpa: List[float]
    crack_risk_index: List[float]
    hardening_degree: List[float]


class DryingShrinkagePredictionResponse(BaseModel):
    formulation_id: str
    formulation_name: str
    ambient_temperature_c: float
    ambient_humidity_pct: float
    constraint_factor: float
    wall_thickness_mm: float
    final_shrinkage_strain: float
    final_tensile_stress_mpa: float
    crack_risk_index: float
    predicted_crack_width_mm: float
    crack_risk_level: str
    critical_period_days: float
    tensile_strength_mpa: float
    elastic_modulus_mpa: float
    shrinkage_time_curve: ShrinkageTimeCurve
    recommendations: List[str]


class FormulationComparisonItem(BaseModel):
    formulation_id: str
    formulation_name: str
    shrinkage_strain: float
    crack_risk_index: float
    crack_risk_level: str
    tensile_strength_mpa: float


class FormulationComparisonRequest(BaseModel):
    ambient_temperature_c: float = 20.0
    ambient_humidity_pct: float = 60.0
    constraint_factor: float = 0.7


class FormulationComparisonResponse(BaseModel):
    ambient_temperature_c: float
    ambient_humidity_pct: float
    formulations: List[FormulationComparisonItem]
