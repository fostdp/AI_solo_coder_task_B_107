from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Optional, Any
from datetime import datetime


class Point3D(BaseModel):
    x: float
    y: float
    z: float


class VibrationDataBatch(BaseModel):
    timestamp: Optional[str] = None
    sensors: Dict[str, Dict[str, List[float]]]


class ThermalImageData(BaseModel):
    timestamp: Optional[str] = None
    camera_id: str
    temperature_matrix: Optional[List[List[float]]] = None
    thermal_bytes: Optional[bytes] = None
    image_path: Optional[str] = None
    max_temp: Optional[float] = None
    min_temp: Optional[float] = None
    avg_temp: Optional[float] = None
    hotspot_regions: Optional[List[Dict]] = None


class DataIngestResponse(BaseModel):
    success: bool
    message: str
    records_stored: int
    timestamp: str


class ModalAnalysisResponse(BaseModel):
    surface_id: str
    frequencies: List[float]
    damping_ratios: List[float]
    regions_count: int
    processing_ms: int
    regions: List[Dict]


class DiffusionSimulateRequest(BaseModel):
    task_id: str
    elapsed_seconds: Optional[int] = None


class DiffusionResponse(BaseModel):
    injection_point_id: str
    predicted_radius_mm: float
    penetration_depth_mm: float
    flow_rate_mls: float
    volume_ml: float
    elapsed_seconds: int
    diffusion_front_points: int
    streamlines_count: int


class EffectivenessAssessmentResponse(BaseModel):
    task_id: str
    surface_id: str
    pre_grout: Dict
    post_grout: Dict
    assessment: Dict


class AlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    alert_id: int
    cave_id: str
    surface_id: str
    alert_type: str
    severity: str
    message: str
    metrics: Optional[Dict] = None
    status: Optional[str] = None
    push_channels: Optional[Dict] = None
    created_at: Optional[datetime] = None


class GroutSimulateEvent(BaseModel):
    task_id: str
    surface_id: str
    injection_points: List[Dict]
    pressure_kpa: float
    elapsed_seconds: Optional[int] = None


class VibrationRawEvent(BaseModel):
    timestamp: str
    sensors: Dict[str, Dict[str, List[float]]]


class VibrationDenoisedEvent(BaseModel):
    surface_id: str
    timestamp: str
    sensors: Dict[str, Dict[str, List[float]]]


class ModalResultEvent(BaseModel):
    surface_id: str
    timestamp: str
    natural_frequencies: List[float]
    damping_ratios: List[float]
    model_order: int
    processing_ms: int


class DelaminationEvent(BaseModel):
    surface_id: str
    timestamp: str
    regions: List[Dict]


class PressureFlowDataPoint(BaseModel):
    pressure_kpa: float
    flow_rate_mls: float
    elapsed_seconds: float
    temperature_c: Optional[float] = None
    is_reliable: bool = True


class InjectionRateOptimizationRequest(BaseModel):
    surface_id: Optional[str] = None
    grouting_task_id: Optional[str] = None
    measured_data: Optional[List[PressureFlowDataPoint]] = None
    use_simulated_data: bool = True
    simulation_point_count: int = 30
    target_radius_mm: Optional[float] = None
    max_allowable_risk_pct: float = 30.0
    pressure_range_kpa: Optional[List[float]] = None
    max_polynomial_degree: int = 5
    delamination_threshold_pressure_kpa: float = 450.0
    max_flow_rate_mls: float = 500.0


class PressureFlowCurvePoint(BaseModel):
    pressure_kpa: float
    flow_rate_mls: float
    delamination_risk_pct: float


class InjectionRateOptimizationResponse(BaseModel):
    surface_id: Optional[str] = None
    grouting_task_id: Optional[str] = None
    optimal_pressure_kpa: float
    optimal_flow_rate_mls: float
    polynomial_degree: int
    polynomial_coefficients: List[float]
    r_squared: float
    secondary_delamination_risk_pct: float
    recommended_max_flow_mls: float
    pressure_flow_curve: List[PressureFlowCurvePoint]
    warning_message: Optional[str] = None
    data_source: str
    data_points_count: int
    processing_timestamp: str


class BondStrengthAssessmentRequest(BaseModel):
    surface_id: str
    baseline_damping_ratios: List[float]
    current_damping_ratios: List[float]
    frequencies: Optional[List[float]] = None
    mode_shapes: Optional[List[List[float]]] = None
    baseline_bond_strength_mpa: float = 0.8
    damping_sensitivity_coefficient: float = 2.5


class BondStrengthAssessmentResponse(BaseModel):
    surface_id: str
    timestamp: str
    baseline_damping_ratios: List[float]
    current_damping_ratios: List[float]
    frequencies_hz: Optional[List[float]] = None
    energy_weights: Optional[List[float]] = None
    per_mode_debonding_pct: List[float]
    per_mode_dissipation_energy: List[float]
    bond_strength_degradation_pct: float
    remaining_bond_strength_mpa: float
    baseline_bond_strength_mpa: float
    critical_mode_index: Optional[int] = None
    assessment_confidence: float
    risk_level: str
    recommendations: List[str]


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


class FormulationComparisonResponse(BaseModel):
    ambient_temperature_c: float
    ambient_humidity_pct: float
    formulations: List[FormulationComparisonItem]


class CuringScheduleOptimizationRequest(BaseModel):
    formulation_id: str = "sintered_stone_powder_ps_basic"
    ambient_temperature_c: float = 20.0
    ambient_humidity_pct: float = 60.0
    target_risk_index: float = 0.5


class OptimalCuringSchedule(BaseModel):
    curing_humidity_pct: float
    curing_duration_days: int
    achieved_risk_index: float
    predicted_crack_width_mm: float
    note: Optional[str] = None


class CuringScheduleOptimizationResponse(BaseModel):
    optimization_needed: bool
    baseline_risk: float
    target_risk: float
    recommendation: str
    optimal_schedule: Optional[OptimalCuringSchedule] = None


class VisitorFlowStatsRequest(BaseModel):
    cave_id: str
    days: int = 365
    base_daily_visitors: float = 500.0


class VisitorFlowStatsResponse(BaseModel):
    cave_id: str
    avg_daily: float
    peak_daily: int
    min_daily: int
    total_annual: int
    peak_season_avg: float
    off_peak_avg: float
    daily_visitors: Optional[List[int]] = None
    dates: Optional[List[str]] = None


class CavePriorityInput(BaseModel):
    cave_id: str
    cave_name: str
    total_delamination_area_sqm: float
    max_severity_score: float
    historical_repair_count: int
    last_repair_years_ago: float
    avg_visitor_flow_daily: float
    peak_visitor_flow_daily: float
    cultural_significance_score: float
    structural_importance: float
    avg_bond_strength_remaining_mpa: float
    active_alerts_count: int
    wall_surfaces_count: int = 5


class CavePriorityRankingRequest(BaseModel):
    caves: List[CavePriorityInput]
    weights: Optional[Dict[str, float]] = None
    use_nsga_ii: bool = True


class CavePriorityRankingItem(BaseModel):
    cave_id: str
    cave_name: str
    priority_rank: int
    urgency_score: float
    cost_efficiency_score: float
    cultural_impact_score: float
    aggregated_score: float
    weights_used: Dict[str, float]
    method: str
    pareto_rank: Optional[int] = None
    crowding_distance: Optional[float] = None
    nsga_ii_summary: Optional[Dict] = None
    detail_metrics: Dict


class CavePriorityRankingResponse(BaseModel):
    ranking: List[CavePriorityRankingItem]
    method: str
    total_caves: int
    timestamp: str
    pareto_front_size: Optional[int] = None
