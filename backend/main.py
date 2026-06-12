import logging
import sys
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import settings
from backend.models import test_connection, get_db, AsyncSessionLocal
from backend.api.inventory import router as inventory_router
from backend.api.data_routes import router as data_router
from backend.api.grouting import router as grouting_router
from backend.api.visualization import router as viz_router
from backend.services.data_processing import (
    VibrationProcessingService, AlertDetectionService, GroutingAnalysisService,
)

from backend.apps.pressure_regression.main import app as pressure_regression_app
from backend.apps.bond_strength.main import app as bond_strength_app
from backend.apps.shrinkage_aht.main import app as shrinkage_aht_app
from backend.apps.priority_nsga2.main import app as priority_nsga2_app

logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")
vibration_svc = VibrationProcessingService()
alert_svc = AlertDetectionService()
grouting_svc = GroutingAnalysisService()


async def run_hourly_modal_analysis():
    logger.info("开始执行定时模态分析任务...")
    from backend.models.database import WallSurface

    async with AsyncSessionLocal() as db:
        stmt = WallSurface.__table__.select()
        result = await db.execute(stmt)
        surfaces = result.fetchall()

        for row in surfaces:
            try:
                await vibration_svc.run_modal_analysis(db, row.surface_id)
            except Exception as e:
                logger.error(f"墙面 {row.surface_id} 模态分析失败: {e}")

        try:
            triggered = await alert_svc.check_and_push_alerts(db)
            if triggered:
                logger.info(f"触发告警 {len(triggered)} 条")
            await db.commit()
        except Exception as e:
            logger.error(f"告警检查失败: {e}")
            await db.rollback()

    logger.info("定时模态分析任务完成")


async def run_diffusion_update():
    logger.info("开始执行灌浆扩散定时更新...")
    from backend.models.database import GroutingTask

    async with AsyncSessionLocal() as db:
        stmt = GroutingTask.__table__.select().where(
            GroutingTask.status.in_(["in_progress", "pending"])
        )
        result = await db.execute(stmt)
        tasks = result.fetchall()

        for row in tasks:
            try:
                await grouting_svc.simulate_diffusion(db, row.task_id)
            except Exception as e:
                logger.error(f"任务 {row.task_id} 扩散更新失败: {e}")

        await db.commit()
    logger.info("灌浆扩散更新完成")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"启动 {settings.APP_NAME}...")

    db_ok = await test_connection()
    if db_ok:
        logger.info("数据库连接正常")
    else:
        logger.warning("数据库连接失败，请检查TimescaleDB服务是否启动")

    scheduler.add_job(
        run_hourly_modal_analysis,
        trigger=IntervalTrigger(minutes=settings.ALERT_CHECK_INTERVAL_MINUTES),
        id="modal_analysis_job",
        replace_existing=True,
    )
    logger.info(f"模态分析定时任务已配置: 每 {settings.ALERT_CHECK_INTERVAL_MINUTES} 分钟执行一次")

    scheduler.add_job(
        run_diffusion_update,
        trigger=IntervalTrigger(minutes=5),
        id="diffusion_update_job",
        replace_existing=True,
    )
    logger.info("灌浆扩散更新任务已配置: 每 5 分钟执行一次")

    scheduler.start()
    logger.info("APScheduler 定时任务调度器已启动")

    yield

    scheduler.shutdown(wait=False)
    await alert_svc.close()
    logger.info("应用关闭完成")


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "敦煌研究院莫高窟古代壁画地仗层剥离监测与灌浆加固效果评估系统。"
        "基于微振动传感器（60台）和红外热成像仪（20台）的5G实时监测数据，"
        "采用随机子空间识别（SSI）进行振动模态分析识别剥离区域，"
        "采用牛顿流体球面扩散模型预测灌浆材料（烧结石粉+PS）渗透半径。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inventory_router, prefix=settings.API_V1_PREFIX)
app.include_router(data_router, prefix=settings.API_V1_PREFIX)
app.include_router(grouting_router, prefix=settings.API_V1_PREFIX)
app.include_router(viz_router, prefix=settings.API_V1_PREFIX)

app.mount("/apps/pressure-regression", pressure_regression_app)
app.mount("/apps/bond-strength", bond_strength_app)
app.mount("/apps/shrinkage-aht", shrinkage_aht_app)
app.mount("/apps/priority-nsga2", priority_nsga2_app)


@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "frontend": "/frontend/index.html",
        "api_prefix": settings.API_V1_PREFIX,
        "sub_apps": {
            "pressure_regression": "/apps/pressure-regression",
            "bond_strength": "/apps/bond-strength",
            "shrinkage_aht": "/apps/shrinkage-aht",
            "priority_nsga2": "/apps/priority-nsga2",
        },
    }


@app.get("/health")
async def health_check():
    db_status = await test_connection()
    return {
        "status": "healthy" if db_status else "degraded",
        "database": "connected" if db_status else "disconnected",
        "scheduler": "running" if scheduler.running else "stopped",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/frontend")
async def frontend_index():
    import os
    index_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    return FileResponse(index_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
