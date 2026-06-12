from fastapi import FastAPI
from datetime import datetime

from .api import router as priority_nsga2_router
from backend.config import settings

app = FastAPI(
    title="窟室优先级NSGA-II多目标优化子应用",
    description="基于pymoo的NSGA-II多目标优化窟室优先级排序子应用，支持独立进程计算",
    version="1.0.0",
)

app.include_router(priority_nsga2_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root():
    return {
        "app": "窟室优先级NSGA-II多目标优化子应用",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "api_prefix": settings.API_V1_PREFIX,
        "library": "pymoo",
        "endpoints": {
            "POST /priority-nsga2/rank": "提交窟室列表，启动NSGA-II计算",
            "GET /priority-nsga2/result/{task_id}": "查询计算结果",
            "GET /priority-nsga2/health": "检查worker状态",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "library": "pymoo",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=8003,
        reload=settings.DEBUG,
        log_level="info",
    )
