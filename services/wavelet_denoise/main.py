import json
import asyncio
import numpy as np

from shared.config import settings
from shared.logger_setup import setup_logging
from shared.redis_client import (
    get_redis, close_redis, xadd_msg, xread_group, ack_message, ensure_group,
)
from shared import metrics as m
from backend.algorithms.modal.wavelet_denoise import WaveletThresholdDenoiser

logger = setup_logging("wavelet_denoise")


class WaveletDenoiseWorker:
    def __init__(self):
        self.denoiser = WaveletThresholdDenoiser(
            wavelet=settings.WAVELET_NAME,
            level=settings.WAVELET_LEVEL,
            mode=settings.WAVELET_MODE,
            threshold_method=settings.WAVELET_THRESHOLD,
        )
        self.running = False

    async def run(self):
        r = await get_redis()
        stream = settings.REDIS_STREAM_VIBRATION_RAW
        group = settings.CONSUMER_GROUP
        consumer = f"{settings.CONSUMER_NAME}-wavelet"
        await ensure_group(r, stream, group)
        self.running = True
        logger.info("小波去噪Worker启动, 监听 {stream}", stream=stream)

        while self.running:
            messages = await xread_group(r, stream, group, consumer, count=5)
            if not messages:
                continue

            for msg in messages:
                try:
                    await self._process(r, msg)
                    await ack_message(r, stream, group, msg["_msg_id"])
                except Exception as e:
                    logger.opt(exception=True).error("去噪处理失败")

    async def _process(self, r, msg):
        msg_type = msg.get("type", "")
        if msg_type != "vibration_raw":
            return

        surface_id = msg.get("surface_id", "")
        timestamp = msg.get("timestamp", "")
        sensors_raw = msg.get("sensors", {})
        if isinstance(sensors_raw, str):
            sensors_raw = json.loads(sensors_raw)

        if not sensors_raw:
            return

        denoised_sensors = {}
        for sensor_id, axes in sensors_raw.items():
            denoised_axes = {}
            for axis_name in ("x", "y", "z"):
                sig = np.array(axes.get(axis_name, []), dtype=np.float64)
                if len(sig) > 32:
                    sig = sig - np.mean(sig)
                    sig = self.denoiser.denoise(sig)
                denoised_axes[axis_name] = sig.tolist()
            denoised_sensors[sensor_id] = denoised_axes

        payload = {
            "type": "vibration_denoised",
            "timestamp": timestamp,
            "surface_id": surface_id,
            "sensors": json.dumps(denoised_sensors),
        }
        await xadd_msg(r, settings.REDIS_STREAM_VIBRATION_DENOISED, payload)
        m.DENOISE_BATCHES.labels("wavelet_denoise").inc()
        logger.info("去噪完成: {sid} | {n}传感器", sid=surface_id, n=len(denoised_sensors))

    def stop(self):
        self.running = False


async def main():
    worker = WaveletDenoiseWorker()
    try:
        await worker.run()
    except KeyboardInterrupt:
        worker.stop()
    finally:
        await close_redis()


if __name__ == "__main__":
    asyncio.run(main())
