from fastapi import FastAPI

from backend.apps.bond_strength.api import router as bond_strength_router
from backend.config import settings

app = FastAPI(
    title="粘结强度评估子应用",
    description="基于阻尼比变化的壁画地仗层粘结强度评估与温度修正API",
    version="1.0.0",
)

app.include_router(bond_strength_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root():
    return {
        "app": "粘结强度评估子应用",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "api_prefix": settings.API_V1_PREFIX,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": "2026-06-12T00:00:00Z",
    }
