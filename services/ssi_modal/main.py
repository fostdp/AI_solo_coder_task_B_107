import json
import asyncio
import time
import numpy as np
from datetime import datetime

from shared.config import settings
from shared.logger_setup import setup_logging
from shared.redis_client import (
    get_redis, close_redis, xadd_msg, xread_group, ack_message, ensure_group,
)
from shared.database import AsyncSessionLocal
from shared import metrics as m
from backend.algorithms.ssi_modal import StochasticSubspaceIdentification, detect_delamination_regions

logger = setup_logging("ssi_modal")


class SSIModalWorker:
    def __init__(self):
        self.ssi = StochasticSubspaceIdentification(
            fs=settings.SSI_SAMPLING_RATE_HZ,
            order_min=settings.SSI_MODEL_ORDER_MIN,
            order_max=settings.SSI_MODEL_ORDER_MAX,
            freq_tol=settings.SSI_STABILITY_FREQ_TOL,
            damp_tol=settings.SSI_STABILITY_DAMP_TOL,
            mac_tol=settings.SSI_STABILITY_MAC_TOL,
            wavelet_denoise=False,
        )
        self.running = False

    async def run(self):
        r = await get_redis()
        stream = settings.REDIS_STREAM_VIBRATION_DENOISED
        group = settings.CONSUMER_GROUP
        consumer = f"{settings.CONSUMER_NAME}-ssi"
        await ensure_group(r, stream, group)
        self.running = True
        logger.info("SSI模态分析Worker启动, 监听 {stream}", stream=stream)

        while self.running:
            messages = await xread_group(r, stream, group, consumer, count=3)
            if not messages:
                continue

            for msg in messages:
                try:
                    await self._process(r, msg)
                    await ack_message(r, stream, group, msg["_msg_id"])
                except Exception as e:
                    logger.opt(exception=True).error("SSI分析失败")
                    m.SSI_ANALYSES.labels("ssi_modal", "error").inc()

    async def _process(self, r, msg):
        surface_id = msg.get("surface_id", "")
        timestamp = msg.get("timestamp", "")
        sensors_raw = msg.get("sensors", {})
        if isinstance(sensors_raw, str):
            sensors_raw = json.loads(sensors_raw)

        if not sensors_raw:
            return

        vibration_data = {}
        sensor_positions = []
        for sensor_id, axes in sensors_raw.items():
            vibration_data[sensor_id] = axes
            sensor_positions.append({"x": 0, "y": 0, "z": 0})

        t0 = time.time()
        modal_result = self.ssi.identify(vibration_data)
        processing_ms = int((time.time() - t0) * 1000)
        m.SSI_DURATION_MS.labels("ssi_modal").observe(processing_ms / 1000.0)
        m.SSI_ANALYSES.labels("ssi_modal", "success").inc()

        baseline_freqs = await self._get_baseline(surface_id)
        regions = detect_delamination_regions(
            modal_result, baseline_freqs, sensor_positions
        )
        for reg in regions:
            sev = reg.get("severity_score", 0)
            bucket = "high" if sev > 70 else "mid" if sev > 40 else "low"
            m.DELAMINATIONS_DETECTED.labels("ssi_modal", bucket).inc()
            m.DELAMINATION_AREA_SQM.labels("ssi_modal", surface_id).set(float(reg.get("area_sqm", 0)))

        modal_payload = {
            "type": "modal_result",
            "timestamp": timestamp,
            "surface_id": surface_id,
            "natural_frequencies": json.dumps(modal_result.frequencies.tolist()),
            "damping_ratios": json.dumps(modal_result.damping_ratios.tolist()),
            "model_order": str(modal_result.model_order),
            "processing_ms": str(processing_ms),
            "n_stable_poles": str(len(modal_result.stable_poles)),
        }
        await xadd_msg(r, settings.REDIS_STREAM_MODAL_RESULTS, modal_payload)

        if regions:
            delam_payload = {
                "type": "delamination_detected",
                "timestamp": timestamp,
                "surface_id": surface_id,
                "regions": json.dumps(regions),
                "n_regions": str(len(regions)),
            }
            await xadd_msg(r, settings.REDIS_STREAM_DELAMINATION, delam_payload)

        await self._store_results(surface_id, modal_result, regions, processing_ms)
        logger.info(
            "SSI完成: {sid} | {n_modal}模态 | {n_reg}剥离区域 | {ms}ms",
            sid=surface_id, n_modal=len(modal_result.frequencies),
            n_reg=len(regions), ms=processing_ms,
        )

    async def _get_baseline(self, surface_id):
        try:
            async with AsyncSessionLocal() as db:
                from shared.orm_models import ModalAnalysisResult
                from sqlalchemy import select, desc
                stmt = (
                    select(ModalAnalysisResult)
                    .where(ModalAnalysisResult.surface_id == surface_id)
                    .order_by(desc(ModalAnalysisResult.time))
                    .limit(1)
                )
                res = await db.execute(stmt)
                row = res.scalar_one_or_none()
                if row and row.natural_frequencies:
                    return np.array(row.natural_frequencies)
        except Exception:
            pass
        return None

    async def _store_results(self, surface_id, modal_result, regions, processing_ms):
        try:
            async with AsyncSessionLocal() as db:
                from shared.orm_models import ModalAnalysisResult, DelaminationRegion
                record = ModalAnalysisResult(
                    time=datetime.utcnow(),
                    surface_id=surface_id,
                    natural_frequencies=modal_result.frequencies.tolist(),
                    damping_ratios=modal_result.damping_ratios.tolist(),
                    mode_shapes=modal_result.mode_shapes.tolist(),
                    ssi_model_order=modal_result.model_order,
                    stability_diagram={"stable_poles_count": len(modal_result.stable_poles)},
                    analyzed_sensors=[],
                    processing_time_ms=processing_ms,
                )
                db.add(record)

                for reg in regions:
                    delam = DelaminationRegion(
                        time=datetime.utcnow(),
                        surface_id=surface_id,
                        region_id=reg["region_id"],
                        bounding_polygon_3d=reg["bounding_polygon_3d"],
                        area_sqm=reg["area_sqm"],
                        depth_mm=reg["depth_mm"],
                        severity_score=reg["severity_score"],
                        confidence=reg["confidence"],
                        frequency_drop_pct=reg["frequency_drop_pct"],
                    )
                    db.add(delam)
                await db.commit()
        except Exception as e:
            logger.opt(exception=True).error("结果入库失败")

    def stop(self):
        self.running = False


async def main():
    worker = SSIModalWorker()
    try:
        await worker.run()
    except KeyboardInterrupt:
        worker.stop()
    finally:
        await close_redis()


if __name__ == "__main__":
    asyncio.run(main())
