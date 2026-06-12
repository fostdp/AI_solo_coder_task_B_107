import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


def _load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class SharedSettings(BaseSettings):
    APP_NAME: str = "古代壁画地仗层剥离监测与灌浆加固效果评估系统"
    DEBUG: bool = False

    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/mogao_monitor"
    SYNC_DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/mogao_monitor"

    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_STREAM_VIBRATION_RAW: str = "stream:vibration_raw"
    REDIS_STREAM_VIBRATION_DENOISED: str = "stream:vibration_denoised"
    REDIS_STREAM_MODAL_RESULTS: str = "stream:modal_results"
    REDIS_STREAM_DELAMINATION: str = "stream:delamination_detected"
    REDIS_STREAM_GROUT_REQUEST: str = "stream:grout_request"
    REDIS_STREAM_GROUT_RESULTS: str = "stream:grout_results"
    REDIS_STREAM_PRESSURE_FLOW_OPTIMIZE: str = "stream:pressure_flow_optimize"
    REDIS_STREAM_BOND_STRENGTH_ASSESS: str = "stream:bond_strength_assess"
    REDIS_STREAM_SHRINKAGE_PREDICT: str = "stream:shrinkage_predict"
    REDIS_STREAM_PRIORITY_RANK: str = "stream:priority_rank"
    REDIS_STREAM_FEATURE_RESULTS: str = "stream:feature_results"

    CONSUMER_GROUP: str = "mogao_workers"
    CONSUMER_NAME: str = "worker-01"
    STREAM_BLOCK_MS: int = 5000
    STREAM_MAX_LEN: int = 10000

    SSI_MODEL_ORDER_MIN: int = 10
    SSI_MODEL_ORDER_MAX: int = 50
    SSI_STABILITY_FREQ_TOL: float = 0.01
    SSI_STABILITY_DAMP_TOL: float = 0.05
    SSI_STABILITY_MAC_TOL: float = 0.97
    SSI_BLOCK_ROWS_MAX: int = 200
    SSI_SAMPLING_RATE_HZ: float = 2000.0

    WAVELET_NAME: str = "db8"
    WAVELET_LEVEL: int = 5
    WAVELET_MODE: str = "soft"
    WAVELET_THRESHOLD: str = "rigrsure"

    GROUT_VISCOSITY_PA_S: float = 0.25
    GROUT_PRESSURE_KPA_DEFAULT: float = 50.0
    GROUT_POROSITY_DEFAULT: float = 0.35
    GROUT_PERMEABILITY_M2: float = 1e-12
    GROUT_WALL_THICKNESS_MM: float = 50.0

    ALERT_AREA_INCREASE_PCT: float = 10.0
    ALERT_FREQ_DROP_PCT: float = 5.0
    ALERT_CHECK_INTERVAL_MINUTES: int = 30

    WECOM_WEBHOOK_URL: Optional[str] = None
    SATELLITE_SMS_API_URL: Optional[str] = None
    SATELLITE_SMS_API_KEY: Optional[str] = None

    PARAMS_FILE: str = ""

    class Config:
        env_file = ".env"
        env_prefix = "MOGAO_"

    def load_params_from_yaml(self, path: Optional[str] = None):
        yaml_path = path or self.PARAMS_FILE
        if not yaml_path:
            base = Path(__file__).resolve().parent.parent / "params" / "default.yaml"
            yaml_path = str(base)
        data = _load_yaml(yaml_path)
        if not data:
            return
        for key, value in data.items():
            key_upper = key.upper()
            if hasattr(self, key_upper):
                setattr(self, key_upper, type(getattr(self, key_upper))(value))


settings = SharedSettings()
settings.load_params_from_yaml()
