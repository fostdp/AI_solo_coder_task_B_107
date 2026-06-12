import json
import asyncio
import time
import numpy as np
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Dict, Optional, Set

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from shared.config import settings
from shared.logger_setup import setup_logging
from shared.redis_client import (
    get_redis, close_redis, xadd_msg, xread_group, ack_message, ensure_group,
)
from shared.database import AsyncSessionLocal
from shared import metrics as m
from shared.metrics import metrics_endpoint
from backend.services.alert_push import AlertPushService

logger = setup_logging("alert_ws")


class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, Set[WebSocket]] = {}

    async def connect(self, ws: WebSocket, cave_id: str):
        await ws.accept()
        if cave_id not in self.active:
            self.active[cave_id] = set()
        self.active[cave_id].add(ws)
        m.WS_CONNECTIONS.labels("alert_ws", cave_id).set(len(self.active[cave_id]))
        logger.info("WebSocket客户端连接: cave={cave_id} | 总连接数={n}",
                    cave_id=cave_id, n=sum(len(s) for s in self.active.values()))

    def disconnect(self, ws: WebSocket, cave_id: str):
        if cave_id in self.active:
            self.active[cave_id].discard(ws)
            m.WS_CONNECTIONS.labels("alert_ws", cave_id).set(len(self.active[cave_id]))
            if not self.active[cave_id]:
                del self.active[cave_id]

    async def broadcast(self, cave_id: str, data: dict):
        if cave_id not in self.active:
            return
        dead = []
        for ws in self.active[cave_id]:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active[cave_id].discard(ws)


manager = ConnectionManager()
push_service = AlertPushService()


class AlertWorker:
    def __init__(self):
        self.running = False

    async def run(self):
        r = await get_redis()
        group = settings.CONSUMER_GROUP
        consumer = f"{settings.CONSUMER_NAME}-alert"

        modal_stream = settings.REDIS_STREAM_MODAL_RESULTS
        delam_stream = settings.REDIS_STREAM_DELAMINATION
        grout_stream = settings.REDIS_STREAM_GROUT_RESULTS

        await ensure_group(r, modal_stream, group)
        await ensure_group(r, delam_stream, group)
        await ensure_group(r, grout_stream, group)

        self.running = True
        logger.info("告警WebSocket Worker启动, 监听多流")

        while self.running:
            for stream in [modal_stream, delam_stream, grout_stream]:
                messages = await xread_group(r, stream, group, consumer, count=5, block_ms=1000)
                for msg in messages:
                    try:
                        await self._process(r, msg)
                        await ack_message(r, stream, group, msg["_msg_id"])
                    except Exception as e:
                        logger.opt(exception=True).error("告警处理失败")

    async def _process(self, r, msg):
        msg_type = msg.get("type", "")
        surface_id = msg.get("surface_id", "")
        timestamp = msg.get("timestamp", "")

        if msg_type == "delamination_detected":
            regions = msg.get("regions", [])
            if isinstance(regions, str):
                regions = json.loads(regions)
            cave_id = surface_id.split("-")[0] if "-" in surface_id else "C096"

            await manager.broadcast(cave_id, {
                "event": "delamination_update",
                "surface_id": surface_id,
                "timestamp": timestamp,
                "regions": regions,
            })

            alert = await self._check_area_alert(surface_id)
            if alert:
                await self._push_and_store(alert)

        elif msg_type == "modal_result":
            freqs = msg.get("natural_frequencies", [])
            if isinstance(freqs, str):
                freqs = json.loads(freqs)
            cave_id = surface_id.split("-")[0] if "-" in surface_id else "C096"

            await manager.broadcast(cave_id, {
                "event": "modal_update",
                "surface_id": surface_id,
                "timestamp": timestamp,
                "frequencies": freqs,
            })

            alert = await self._check_freq_alert(surface_id)
            if alert:
                await self._push_and_store(alert)

        elif msg_type == "grout_diffusion_result":
            cave_id = surface_id.split("-")[0] if "-" in surface_id else "C096"
            results = msg.get("results", [])
            if isinstance(results, str):
                results = json.loads(results)

            await manager.broadcast(cave_id, {
                "event": "grout_update",
                "surface_id": surface_id,
                "timestamp": timestamp,
                "diffusion_results": results,
            })

    async def _check_area_alert(self, surface_id: str) -> Optional[Dict]:
        try:
            async with AsyncSessionLocal() as db:
                from shared.orm_models import DelaminationRegion
                from sqlalchemy import select, and_, func

                now = datetime.utcnow()
                month_ago = now - timedelta(days=30)

                stmt = (
                    select(
                        func.time_bucket("1 day", DelaminationRegion.time).label("day"),
                        func.sum(DelaminationRegion.area_sqm).label("total_area"),
                    )
                    .where(
                        and_(
                            DelaminationRegion.surface_id == surface_id,
                            DelaminationRegion.is_active == True,
                            DelaminationRegion.time >= month_ago,
                        )
                    )
                    .group_by("day")
                    .order_by("day.desc()")
                    .limit(35)
                )
                res = await db.execute(stmt)
                daily = [(r.day, float(r.total_area or 0)) for r in res.all()]

                if len(daily) < 7:
                    return None

                recent_avg = float(np.mean([a for _, a in daily[:7]]))
                older_avg = float(np.mean([a for _, a in daily[-7:]])) if len(daily) >= 14 else recent_avg

                if older_avg > 0:
                    increase_pct = (recent_avg - older_avg) / older_avg * 100.0
                else:
                    increase_pct = 0.0

                if increase_pct >= settings.ALERT_AREA_INCREASE_PCT:
                    cave_id = surface_id.split("-")[0]
                    severity = "critical" if increase_pct >= 25 else "warning"
                    return {
                        "cave_id": cave_id, "surface_id": surface_id,
                        "alert_type": "剥离面积快速增长", "severity": severity,
                        "message": f"剥离空鼓面积增长 {increase_pct:.1f}%，超过阈值 {settings.ALERT_AREA_INCREASE_PCT}%",
                        "metrics": {"increase_pct": f"{increase_pct:.2f}%"},
                    }
        except Exception as e:
            logger.opt(exception=True).error("面积告警检查失败")
        return None

    async def _check_freq_alert(self, surface_id: str) -> Optional[Dict]:
        try:
            async with AsyncSessionLocal() as db:
                from shared.orm_models import ModalAnalysisResult
                from sqlalchemy import select, and_, desc

                now = datetime.utcnow()
                month_ago = now - timedelta(days=30)

                stmt = select(ModalAnalysisResult).where(
                    and_(ModalAnalysisResult.surface_id == surface_id,
                         ModalAnalysisResult.time >= month_ago)
                ).order_by(desc(ModalAnalysisResult.time)).limit(500)
                res = await db.execute(stmt)
                results = res.scalars().all()

                if len(results) < 5:
                    return None

                recent = [np.array(r.natural_frequencies) for r in results[:7] if r.natural_frequencies]
                older = [np.array(r.natural_frequencies) for r in results[-7:] if r.natural_frequencies]

                if not recent or not older:
                    return None

                n = 3
                recent_avg = np.mean([f[:n] for f in recent if len(f) >= n], axis=0)
                older_avg = np.mean([f[:n] for f in older if len(f) >= n], axis=0)

                min_len = min(len(recent_avg), len(older_avg))
                drops = [(older_avg[i] - recent_avg[i]) / older_avg[i] * 100
                         for i in range(min_len) if older_avg[i] > 0]

                if not drops:
                    return None

                max_drop = float(np.max(drops))
                if max_drop >= settings.ALERT_FREQ_DROP_PCT:
                    cave_id = surface_id.split("-")[0]
                    severity = "critical" if max_drop >= 15 else "warning"
                    which_mode = int(np.argmax(drops)) + 1
                    return {
                        "cave_id": cave_id, "surface_id": surface_id,
                        "alert_type": "固有频率显著下降", "severity": severity,
                        "message": f"第{which_mode}阶频率下降 {max_drop:.1f}%",
                        "metrics": {"max_drop_pct": f"{max_drop:.2f}%", "mode_order": which_mode},
                    }
        except Exception as e:
            logger.opt(exception=True).error("频率告警检查失败")
        return None

    async def _push_and_store(self, alert: Dict):
        try:
            channels = ["wecom", "satellite_sms"]
            push_results = await push_service.push_alert(alert, channels)
            for ch, res in push_results.items():
                status = "ok" if res.get("success") else "fail"
                m.ALERT_PUSHES.labels("alert_ws", ch, status).inc()

            m.ALERTS_TRIGGERED.labels("alert_ws", alert["alert_type"], alert["severity"]).inc()

            async with AsyncSessionLocal() as db:
                from shared.orm_models import Alert
                record = Alert(
                    cave_id=alert["cave_id"],
                    surface_id=alert["surface_id"],
                    alert_type=alert["alert_type"],
                    severity=alert["severity"],
                    message=alert["message"],
                    metrics=alert.get("metrics"),
                    status="active",
                    push_channels={"requested": channels, "results": push_results},
                )
                db.add(record)
                await db.commit()

            cave_id = alert.get("cave_id", "C096")
            await manager.broadcast(cave_id, {"event": "alert", "data": alert})
            logger.info("告警推送: {t} → {c}", t=alert["alert_type"], c=cave_id)
        except Exception as e:
            logger.opt(exception=True).error("告警推送失败")

    def stop(self):
        self.running = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker = AlertWorker()
    task = asyncio.create_task(worker.run())
    logger.info("告警WebSocket服务启动 (API+Worker)")
    yield
    worker.stop()
    task.cancel()
    await push_service.close()
    await close_redis()


app = FastAPI(title="Alert & WebSocket Service", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    m.API_REQUEST_DURATION.labels(
        "alert_ws", request.method, request.url.path, str(response.status_code)
    ).observe(duration)
    return response


@app.get("/health")
async def health():
    return {
        "service": "alert_ws",
        "status": "running",
        "ws_connections": sum(len(s) for s in manager.active.values()),
    }


@app.get("/metrics")
async def metrics():
    return metrics_endpoint()


@app.websocket("/ws/cave/{cave_id}")
async def ws_cave(ws: WebSocket, cave_id: str):
    await manager.connect(ws, cave_id)
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_json({"event": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(ws, cave_id)


@app.get("/alerts")
async def list_alerts(cave_id: Optional[str] = None, limit: int = 50):
    try:
        async with AsyncSessionLocal() as db:
            from shared.orm_models import Alert
            from sqlalchemy import select, desc
            stmt = select(Alert).order_by(desc(Alert.created_at)).limit(limit)
            if cave_id:
                stmt = stmt.where(Alert.cave_id == cave_id)
            res = await db.execute(stmt)
            rows = res.scalars().all()
            return [
                {
                    "alert_id": r.alert_id, "cave_id": r.cave_id,
                    "cave_id": r.cave_id,
                    "surface_id": r.surface_id,
                    "alert_id_real": r.alert_id,
                    "alert_type": r.alert_type,
                    "severity": r.severity,
                    "message": r.message,
                    "status": r.status,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8005, log_level="info")
