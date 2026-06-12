from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, Text,
    ForeignKey, Index, BigInteger, Numeric, ARRAY, LargeBinary
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, DOUBLE_PRECISION, GEOGRAPHY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from geoalchemy2 import Geography

Base = declarative_base()


class Cave(Base):
    __tablename__ = "caves"
    cave_id = Column(String(20), primary_key=True)
    cave_name = Column(String(100), nullable=False)
    dynasty = Column(String(50))
    location = Column(Geography(geometry_type="POINT", srid=4326))
    description = Column(Text)
    dimensions = Column(JSONB)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)

    wall_surfaces = relationship("WallSurface", back_populates="cave")


class WallSurface(Base):
    __tablename__ = "wall_surfaces"
    surface_id = Column(String(30), primary_key=True)
    cave_id = Column(String(20), ForeignKey("caves.cave_id"))
    wall_type = Column(String(20), nullable=False)
    area_sqm = Column(Numeric(10, 4))
    bounding_box_3d = Column(JSONB)
    description = Column(Text)

    cave = relationship("Cave", back_populates="wall_surfaces")
    vibration_sensors = relationship("VibrationSensor", back_populates="surface")
    thermal_cameras = relationship("ThermalCamera", back_populates="surface")


class VibrationSensor(Base):
    __tablename__ = "vibration_sensors"
    sensor_id = Column(String(30), primary_key=True)
    cave_id = Column(String(20), ForeignKey("caves.cave_id"))
    surface_id = Column(String(30), ForeignKey("wall_surfaces.surface_id"))
    location_3d = Column(JSONB, nullable=False)
    sampling_rate_hz = Column(Integer, default=2000)
    sensitivity = Column(Numeric(10, 6))
    model = Column(String(50))
    status = Column(String(20), default="active")
    installed_at = Column(DateTime)

    surface = relationship("WallSurface", back_populates="vibration_sensors")


class ThermalCamera(Base):
    __tablename__ = "thermal_cameras"
    camera_id = Column(String(30), primary_key=True)
    cave_id = Column(String(20), ForeignKey("caves.cave_id"))
    surface_id = Column(String(30), ForeignKey("wall_surfaces.surface_id"))
    location_3d = Column(JSONB, nullable=False)
    resolution = Column(String(20))
    temp_range_min = Column(Numeric(6, 2))
    temp_range_max = Column(Numeric(6, 2))
    model = Column(String(50))
    status = Column(String(20), default="active")
    installed_at = Column(DateTime)

    surface = relationship("WallSurface", back_populates="thermal_cameras")


class GroutingTask(Base):
    __tablename__ = "grouting_tasks"
    task_id = Column(String(40), primary_key=True)
    cave_id = Column(String(20), ForeignKey("caves.cave_id"))
    surface_id = Column(String(30), ForeignKey("wall_surfaces.surface_id"))
    material_type = Column(String(50), default="烧结石粉+PS")
    injection_points = Column(JSONB)
    start_time = Column(TIMESTAMP(timezone=True))
    end_time = Column(TIMESTAMP(timezone=True))
    total_volume_ml = Column(Numeric(12, 4))
    pressure_kpa = Column(Numeric(8, 4))
    status = Column(String(20), default="pending")
    operator = Column(String(50))
    notes = Column(Text)


class Alert(Base):
    __tablename__ = "alerts"
    alert_id = Column(BigInteger, primary_key=True, autoincrement=True)
    cave_id = Column(String(20), ForeignKey("caves.cave_id"))
    surface_id = Column(String(30), ForeignKey("wall_surfaces.surface_id"))
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    metrics = Column(JSONB)
    status = Column(String(20), default="active")
    push_channels = Column(JSONB)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    resolved_at = Column(TIMESTAMP(timezone=True))


class VibrationRawData(Base):
    __tablename__ = "vibration_raw_data"
    time = Column(TIMESTAMP(timezone=True), primary_key=True)
    sensor_id = Column(String(30), ForeignKey("vibration_sensors.sensor_id"), primary_key=True)
    x_axis_accel = Column(ARRAY(DOUBLE_PRECISION))
    y_axis_accel = Column(ARRAY(DOUBLE_PRECISION))
    z_axis_accel = Column(ARRAY(DOUBLE_PRECISION))
    sample_count = Column(Integer)
    raw_data_hash = Column(String(64))
    received_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)


class ThermalImage(Base):
    __tablename__ = "thermal_images"
    time = Column(TIMESTAMP(timezone=True), primary_key=True)
    camera_id = Column(String(30), ForeignKey("thermal_cameras.camera_id"), primary_key=True)
    thermal_data = Column(LargeBinary)
    temperature_matrix = Column(ARRAY(ARRAY(DOUBLE_PRECISION)))
    max_temp = Column(Numeric(8, 4))
    min_temp = Column(Numeric(8, 4))
    avg_temp = Column(Numeric(8, 4))
    hotspot_regions = Column(JSONB)
    image_path = Column(String(255))
    received_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)


class ModalAnalysisResult(Base):
    __tablename__ = "modal_analysis_results"
    time = Column(TIMESTAMP(timezone=True), primary_key=True)
    surface_id = Column(String(30), ForeignKey("wall_surfaces.surface_id"), primary_key=True)
    natural_frequencies = Column(ARRAY(DOUBLE_PRECISION))
    damping_ratios = Column(ARRAY(DOUBLE_PRECISION))
    mode_shapes = Column(JSONB)
    ssi_model_order = Column(Integer)
    stability_diagram = Column(JSONB)
    analyzed_sensors = Column(ARRAY(String(30)))
    processing_time_ms = Column(Integer)


class DelaminationRegion(Base):
    __tablename__ = "delamination_regions"
    time = Column(TIMESTAMP(timezone=True), primary_key=True)
    surface_id = Column(String(30), ForeignKey("wall_surfaces.surface_id"), primary_key=True)
    region_id = Column(String(40), primary_key=True)
    bounding_polygon_3d = Column(JSONB, nullable=False)
    area_sqm = Column(Numeric(10, 6), nullable=False)
    depth_mm = Column(Numeric(8, 4))
    severity_score = Column(Numeric(5, 2))
    confidence = Column(Numeric(5, 4))
    frequency_drop_pct = Column(Numeric(8, 4))
    detection_method = Column(String(30), default="SSI+Thermal")
    is_active = Column(Boolean, default=True)


class GroutingDiffusion(Base):
    __tablename__ = "grouting_diffusion"
    time = Column(TIMESTAMP(timezone=True), primary_key=True)
    task_id = Column(String(40), ForeignKey("grouting_tasks.task_id"), primary_key=True)
    injection_point_id = Column(String(30), primary_key=True)
    predicted_radius_mm = Column(Numeric(10, 4))
    actual_radius_mm = Column(Numeric(10, 4))
    penetration_depth_mm = Column(Numeric(10, 4))
    pressure_kpa = Column(Numeric(8, 4))
    viscosity_pa_s = Column(Numeric(10, 6))
    porosity = Column(Numeric(6, 4))
    flow_rate_mls = Column(Numeric(10, 4))
    diffusion_front = Column(JSONB)
    particle_pathlines = Column(JSONB)
    elapsed_seconds = Column(Integer)
    model_version = Column(String(20), default="Newtonian_Spherical_v1")


class ReinforcementEffectiveness(Base):
    __tablename__ = "reinforcement_effectiveness"
    time = Column(TIMESTAMP(timezone=True), primary_key=True)
    surface_id = Column(String(30), ForeignKey("wall_surfaces.surface_id"), primary_key=True)
    task_id = Column(String(40), ForeignKey("grouting_tasks.task_id"), primary_key=True)
    pre_grout_frequencies = Column(ARRAY(DOUBLE_PRECISION))
    post_grout_frequencies = Column(ARRAY(DOUBLE_PRECISION))
    frequency_recovery_pct = Column(Numeric(8, 4))
    delamination_area_reduction_pct = Column(Numeric(8, 4))
    bonding_strength_mpa = Column(Numeric(10, 6))
    overall_score = Column(Numeric(5, 2))
    assessment_notes = Column(Text)


class GroutingPressureFlowData(Base):
    __tablename__ = "grouting_pressure_flow_data"
    time = Column(TIMESTAMP(timezone=True), primary_key=True)
    measurement_id = Column(String(40), primary_key=True)
    task_id = Column(String(40), ForeignKey("grouting_tasks.task_id"))
    surface_id = Column(String(30), ForeignKey("wall_surfaces.surface_id"))
    pressure_kpa = Column(Numeric(10, 4), nullable=False)
    flow_rate_mls = Column(Numeric(10, 4), nullable=False)
    elapsed_seconds = Column(Numeric(12, 4))
    temperature_c = Column(Numeric(6, 2))
    is_reliable = Column(Boolean, default=True)
    data_source = Column(String(20), default="measured")
    polynomial_degree = Column(Integer)
    r_squared = Column(Numeric(8, 6))
    optimal_pressure_kpa = Column(Numeric(10, 4))
    optimal_flow_rate_mls = Column(Numeric(10, 4))
    secondary_delamination_risk_pct = Column(Numeric(8, 4))
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)


class BondStrengthAssessment(Base):
    __tablename__ = "bond_strength_assessments"
    time = Column(TIMESTAMP(timezone=True), primary_key=True)
    assessment_id = Column(String(40), primary_key=True)
    surface_id = Column(String(30), ForeignKey("wall_surfaces.surface_id"), nullable=False)
    baseline_damping_ratios = Column(ARRAY(DOUBLE_PRECISION))
    current_damping_ratios = Column(ARRAY(DOUBLE_PRECISION))
    frequencies_hz = Column(ARRAY(DOUBLE_PRECISION))
    energy_weights = Column(ARRAY(DOUBLE_PRECISION))
    per_mode_debonding_pct = Column(ARRAY(DOUBLE_PRECISION))
    per_mode_dissipation_energy = Column(ARRAY(DOUBLE_PRECISION))
    bond_strength_degradation_pct = Column(Numeric(8, 4), nullable=False)
    remaining_bond_strength_mpa = Column(Numeric(10, 6), nullable=False)
    baseline_bond_strength_mpa = Column(Numeric(10, 6))
    critical_mode_index = Column(Integer)
    assessment_confidence = Column(Numeric(5, 4))
    risk_level = Column(String(20), nullable=False)
    recommendations = Column(JSONB)
    damping_sensitivity_coefficient = Column(Numeric(8, 4))
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)


class DryingShrinkagePrediction(Base):
    __tablename__ = "drying_shrinkage_predictions"
    time = Column(TIMESTAMP(timezone=True), primary_key=True)
    prediction_id = Column(String(40), primary_key=True)
    task_id = Column(String(40), ForeignKey("grouting_tasks.task_id"))
    surface_id = Column(String(30), ForeignKey("wall_surfaces.surface_id"))
    formulation_id = Column(String(50), nullable=False)
    formulation_name = Column(String(100))
    ambient_temperature_c = Column(Numeric(6, 2), nullable=False)
    ambient_humidity_pct = Column(Numeric(6, 2), nullable=False)
    constraint_factor = Column(Numeric(5, 3))
    wall_thickness_mm = Column(Numeric(8, 2))
    final_shrinkage_strain = Column(Numeric(12, 8))
    final_tensile_stress_mpa = Column(Numeric(10, 6))
    crack_risk_index = Column(Numeric(8, 4))
    predicted_crack_width_mm = Column(Numeric(8, 4))
    crack_risk_level = Column(String(20), nullable=False)
    critical_period_days = Column(Numeric(8, 2))
    tensile_strength_mpa = Column(Numeric(10, 6))
    elastic_modulus_mpa = Column(Numeric(10, 4))
    shrinkage_time_curve = Column(JSONB)
    recommendations = Column(JSONB)
    aht_model_version = Column(String(20), default="AHT_v1")
    prediction_horizon_days = Column(Numeric(8, 2))
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)


class VisitorFlowStats(Base):
    __tablename__ = "visitor_flow_stats"
    date = Column(TIMESTAMP(timezone=True), primary_key=True)
    cave_id = Column(String(20), ForeignKey("caves.cave_id"), primary_key=True)
    daily_visitors = Column(Integer, nullable=False)
    is_weekend = Column(Boolean, default=False)
    is_peak_season = Column(Boolean, default=False)
    is_holiday = Column(Boolean, default=False)
    temperature_c = Column(Numeric(6, 2))
    humidity_pct = Column(Numeric(6, 2))
    special_event = Column(String(100))
    notes = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("idx_visitor_flow_cave_date", "cave_id", "date"),
    )


class CavePriorityRanking(Base):
    __tablename__ = "cave_priority_rankings"
    time = Column(TIMESTAMP(timezone=True), primary_key=True)
    ranking_id = Column(String(40), primary_key=True)
    cave_id = Column(String(20), ForeignKey("caves.cave_id"), nullable=False)
    priority_rank = Column(Integer, nullable=False)
    total_caves = Column(Integer)
    urgency_score = Column(Numeric(8, 4))
    cost_efficiency_score = Column(Numeric(8, 4))
    cultural_impact_score = Column(Numeric(8, 4))
    aggregated_score = Column(Numeric(8, 4), nullable=False)
    weights_used = Column(JSONB)
    method = Column(String(30), default="nsga_ii_multi_objective")
    pareto_rank = Column(Integer)
    crowding_distance = Column(Numeric(12, 6))
    pareto_front_size = Column(Integer)
    nsga_ii_summary = Column(JSONB)
    input_metrics = Column(JSONB)
    recommendations = Column(JSONB)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)
