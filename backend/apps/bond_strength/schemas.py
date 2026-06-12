from pydantic import BaseModel, ConfigDict
from typing import List, Optional


class BondStrengthAssessRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    surface_id: str
    baseline_damping_ratios: List[float]
    current_damping_ratios: List[float]
    frequencies: Optional[List[float]] = None
    ambient_temperature_c: Optional[float] = None
    timestamp: Optional[str] = None


class BondStrengthAssessResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    surface_id: str
    timestamp: str
    baseline_damping_ratios: List[float]
    current_damping_ratios: List[float]
    raw_current_damping_ratios: List[float]
    frequencies_hz: List[float]
    energy_weights: List[float]
    per_mode_debonding_pct: List[float]
    per_mode_dissipation_energy: List[float]
    bond_strength_degradation_pct: float
    remaining_bond_strength_mpa: float
    baseline_bond_strength_mpa: float
    critical_mode_index: Optional[int]
    assessment_confidence: float
    risk_level: str
    ambient_temperature_c: Optional[float]
    temperature_correction_factor: Optional[float]
    recommendations: List[str]


class TemperatureCalibrateRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    damping_ratios: List[float]
    temperature_c: float
    reference_temperature_c: float = 20.0


class TemperatureCalibrateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    corrected_damping_ratios: List[float]
    correction_factor: float
