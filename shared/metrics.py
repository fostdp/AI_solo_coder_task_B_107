from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response

REGISTRY = CollectorRegistry()

VIBRATION_BATCHES_INGESTED = Counter(
    "mogao_vibration_batches_total",
    "Total vibration batches ingested",
    ["service"],
    registry=REGISTRY,
)

THERMAL_IMAGES_INGESTED = Counter(
    "mogao_thermal_images_total",
    "Total thermal images ingested",
    ["service"],
    registry=REGISTRY,
)

DENOISE_BATCHES = Counter(
    "mogao_denoise_batches_total",
    "Total denoised vibration batches",
    ["service"],
    registry=REGISTRY,
)

SSI_ANALYSES = Counter(
    "mogao_ssi_analyses_total",
    "Total SSI modal analyses executed",
    ["service", "status"],
    registry=REGISTRY,
)

SSI_DURATION_MS = Histogram(
    "mogao_ssi_duration_seconds",
    "SSI analysis duration histogram",
    ["service"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY,
)

DELAMINATIONS_DETECTED = Counter(
    "mogao_detections_total",
    "Total delamination regions detected",
    ["service", "severity"],
    registry=REGISTRY,
)

GROUT_DIFFUSIONS = Counter(
    "mogao_grout_diffusions_total",
    "Total grout diffusion simulations",
    ["service"],
    registry=REGISTRY,
)

ALERTS_TRIGGERED = Counter(
    "mogao_alerts_total",
    "Total alerts triggered",
    ["service", "type", "severity"],
    registry=REGISTRY,
)

ALERT_PUSHES = Counter(
    "mogao_alert_pushes_total",
    "Total alert push results",
    ["service", "channel", "status"],
    registry=REGISTRY,
)

WS_CONNECTIONS = Gauge(
    "mogao_ws_connections",
    "Current active WebSocket connections",
    ["service", "cave_id"],
    registry=REGISTRY,
)

STREAM_LAG_MS = Gauge(
    "mogao_stream_lag_seconds",
    "Current Redis Stream lag per consumer",
    ["service", "stream"],
    registry=REGISTRY,
)

DELAMINATION_AREA_SQM = Gauge(
    "mogao_delamination_area_sqm",
    "Current total delamination area per surface",
    ["service", "surface_id"],
    registry=REGISTRY,
)

GROUT_RADIUS_MM = Gauge(
    "mogao_grout_radius_mm",
    "Current grout penetration radius per injection point",
    ["service", "injection_point_id"],
    registry=REGISTRY,
)

API_REQUEST_DURATION = Histogram(
    "mogao_api_request_duration_seconds",
    "API request duration histogram",
    ["service", "method", "path", "status_code"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY,
)

FEATURE_COMPUTATIONS = Counter(
    "mogao_feature_computations_total",
    "Total feature computations executed",
    ["feature_type"],
    registry=REGISTRY,
)

FEATURE_OPTIMAL_PRESSURE = Gauge(
    "mogao_optimal_injection_pressure_kpa",
    "Optimal injection pressure per task",
    ["task_id"],
    registry=REGISTRY,
)

FEATURE_BOND_STRENGTH = Gauge(
    "mogao_remaining_bond_strength_mpa",
    "Remaining bond strength per surface",
    ["surface_id", "risk_level"],
    registry=REGISTRY,
)

FEATURE_CRACK_RISK = Gauge(
    "mogao_crack_risk_index",
    "Predicted crack risk index per prediction",
    ["formulation_id", "risk_level"],
    registry=REGISTRY,
)

FEATURE_PRIORITY_RANK = Gauge(
    "mogao_priority_rank_score",
    "Priority ranking score per cave",
    ["cave_id", "rank"],
    registry=REGISTRY,
)


def metrics_endpoint():
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
