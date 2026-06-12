import hashlib
import json
import time
import numpy as np
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from shared.config import settings
from shared.logger_setup import setup_logging
from shared.schemas import VibrationDataBatch, ThermalImageData, DataIngestResponse
from shared.redis_client import get_redis, close_redis, xadd_msg, ensure_group
from shared.database import AsyncSessionLocal, test_connection
from shared import metrics as m
from shared.metrics import metrics_endpoint

logger = setup_logging("5g_ingest")

SURFACE_MAP = {
    f"VB-{i:03d}": f"C096-{'N' if i<10 else 'S' if i<20 else 'E' if i<28 else 'W' if i<36 else 'C'}"
    for i in range(41)
}
for i in range(41, 52):
    SURFACE_MAP[f"VB-{i:03d}"] = f"C257-{'N' if i<44 else 'S' if i<47 else 'E' if i<48 else 'W'}"
for i in range(52, 60):
    SURFACE_MAP[f"VB-{i:03d}"] = f"C285-{'N' if i<55 else 'S'}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    r = await get_redis()
    await ensure_group(r, settings.REDIS_STREAM_VIBRATION_RAW, settings.CONSUMER_GROUP)
    db_ok = await test_connection()
    logger.info(f"5G数据接入服务启动 | DB={'OK' if db_ok else 'FAIL'} | Redis=OK")
    yield
    await close_redis()
    logger.info("5G数据接入服务关闭")


app = FastAPI(title="5G Data Ingest Service", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    m.API_REQUEST_DURATION.labels(
        "5g_ingest", request.method, request.url.path, str(response.status_code)
    ).observe(duration)
    return response


@app.get("/health")
async def health():
    return {"service": "5g_ingest", "status": "running"}


@app.get("/metrics")
async def metrics():
    return metrics_endpoint()


@app.post("/ingest/vibration", response_model=DataIngestResponse)
async def ingest_vibration(batch: VibrationDataBatch):
    r = await get_redis()
    ts = batch.timestamp or datetime.utcnow().isoformat()
    count = 0

    sensor_map = {}
    for sensor_id, data in batch.sensors.items():
        surface_id = SURFACE_MAP.get(sensor_id, "C096-N")
        if surface_id not in sensor_map:
            sensor_map[surface_id] = {}
        sensor_map[surface_id][sensor_id] = data
        count += 1

    for surface_id, sensors in sensor_map.items():
        payload = {
            "type": "vibration_raw",
            "timestamp": ts,
            "surface_id": surface_id,
            "sensors": json.dumps(sensors),
        }
        await xadd_msg(r, settings.REDIS_STREAM_VIBRATION_RAW, payload)

    async with AsyncSessionLocal() as db:
        try:
            for sensor_id, data in batch.sensors.items():
                from shared.orm_models import VibrationRawData
                x_data = list(data.get("x", []))
                sample_count = len(x_data)
                data_hash = hashlib.sha256(
                    f"{sensor_id}|{ts}|{sample_count}".encode()
                ).hexdigest()
                record = VibrationRawData(
                    time=datetime.fromisoformat(ts.replace("Z", "+00:00")) if "T" in ts else datetime.utcnow(),
                    sensor_id=sensor_id,
                    x_axis_accel=x_data,
                    y_axis_accel=list(data.get("y", [])),
                    z_axis_accel=list(data.get("z", [])),
                    sample_count=sample_count,
                    raw_data_hash=data_hash,
                )
                db.add(record)
            await db.commit()
        except Exception as e:
            logger.error(f"振动数据DB写入失败: {e}")
            await db.rollback()

    logger.info(f"振动数据接入: {count}传感器 → Redis+DB")
    m.VIBRATION_BATCHES_INGESTED.labels("5g_ingest").inc(count)
    return DataIngestResponse(
        success=True,
        message=f"成功存储 {count} 个传感器的振动数据",
        records_stored=count,
        timestamp=ts,
    )


@app.post("/ingest/thermal", response_model=DataIngestResponse)
async def ingest_thermal(image: ThermalImageData):
    r = await get_redis()
    ts = image.timestamp or datetime.utcnow().isoformat()

    payload = {
        "type": "thermal_raw",
        "timestamp": ts,
        "camera_id": image.camera_id,
        "max_temp": str(image.max_temp or 0),
        "min_temp": str(image.min_temp or 0),
        "avg_temp": str(image.avg_temp or 0),
        "hotspot_regions": json.dumps(image.hotspot_regions or []),
    }
    await xadd_msg(r, settings.REDIS_STREAM_VIBRATION_RAW, payload)

    async with AsyncSessionLocal() as db:
        try:
            from shared.orm_models import ThermalImage
            record = ThermalImage(
                time=datetime.fromisoformat(ts.replace("Z", "+00:00")) if "T" in ts else datetime.utcnow(),
                camera_id=image.camera_id,
                temperature_matrix=image.temperature_matrix,
                max_temp=image.max_temp,
                min_temp=image.min_temp,
                avg_temp=image.avg_temp,
                hotspot_regions=image.hotspot_regions,
                image_path=image.image_path,
            )
            db.add(record)
            await db.commit()
        except Exception as e:
            logger.error(f"热成像DB写入失败: {e}")
            await db.rollback()

    m.THERMAL_IMAGES_INGESTED.labels("5g_ingest").inc()
    return DataIngestResponse(
        success=True, message="热成像数据存储成功",
        records_stored=1, timestamp=ts,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, log_level="info")
