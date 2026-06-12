from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    APP_NAME: str = "古代壁画地仗层剥离监测与灌浆加固效果评估系统"
    DEBUG: bool = True

    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/mogao_monitor"
    SYNC_DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/mogao_monitor"

    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    VIBRATION_SENSOR_COUNT: int = 60
    THERMAL_CAMERA_COUNT: int = 20
    REPORT_INTERVAL_SECONDS: int = 3600

    SSI_MODEL_ORDER_MIN: int = 10
    SSI_MODEL_ORDER_MAX: int = 50
    SSI_STABILITY_FREQ_TOL: float = 0.01
    SSI_STABILITY_DAMP_TOL: float = 0.05
    SSI_STABILITY_MAC_TOL: float = 0.97

    GROUT_VISCOSITY_PA_S: float = 0.25
    GROUT_PRESSURE_KPA_DEFAULT: float = 50.0
    GROUT_POROSITY_DEFAULT: float = 0.35

    ALERT_AREA_INCREASE_PCT: float = 10.0
    ALERT_FREQ_DROP_PCT: float = 5.0
    ALERT_CHECK_INTERVAL_MINUTES: int = 30

    WECOM_WEBHOOK_URL: Optional[str] = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=YOUR_KEY_HERE"
    SATELLITE_SMS_API_URL: Optional[str] = "https://api.satellite-sms.example.com/send"
    SATELLITE_SMS_API_KEY: Optional[str] = "YOUR_SATELLITE_API_KEY"

    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:8000",
    ]

    class Config:
        env_file = ".env"


settings = Settings()
