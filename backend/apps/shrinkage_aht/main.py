from fastapi import FastAPI
from datetime import datetime

from backend.apps.shrinkage_aht.api import router as shrinkage_aht_router
from backend.apps.shrinkage_aht.api import USE_CYTHON
from backend.config import settings

app = FastAPI(
    title="干燥收缩预测子应用",
    description="基于AHT模型的灌浆材料干燥收缩预测与配方比较API，支持Cython加速",
    version="1.0.0",
)

app.include_router(shrinkage_aht_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root():
    return {
        "app": "干燥收缩预测子应用",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "api_prefix": settings.API_V1_PREFIX,
        "cython_accelerated": USE_CYTHON,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cython_accelerated": USE_CYTHON,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=8001,
        reload=settings.DEBUG,
        log_level="info",
    )
