from fastapi import FastAPI
from datetime import datetime

from .api import router as pressure_regression_router
from backend.config import settings

app = FastAPI(
    title="压力流量多项式回归子应用",
    description="基于Ridge回归的压力-流量数据拟合、预测与优化子应用",
    version="1.0.0",
)

app.include_router(pressure_regression_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root():
    return {
        "app": "压力流量多项式回归子应用",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "api_prefix": settings.API_V1_PREFIX,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=8002,
        reload=settings.DEBUG,
        log_level="info",
    )
