import asyncio
import aiohttp
import numpy as np
import json
import time
import random
import argparse
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pathlib import Path

from shared.logger_setup import setup_logging
from shared import metrics as m
from shared.metrics import metrics_endpoint

logger = setup_logging("5g_simulator")


class FiveGSimulatorConfig:
    def __init__(self, api_base: Optional[str] = None,
                 n_vibration_sensors: int = 60,
                 n_thermal_cameras: int = 20,
                 interval_seconds: int = 3600,
                 delamination_surfaces: Optional[List[str]] = None):
        self.API_BASE = api_base or os.environ.get("MOGAO_INGEST_URL", "http://localhost:8001")
        self.VIBRATION_ENDPOINT = f"{self.API_BASE}/ingest/vibration"
        self.THERMAL_ENDPOINT = f"{self.API_BASE}/ingest/thermal"

        self.N_VIBRATION_SENSORS = n_vibration_sensors
        self.N_THERMAL_CAMERAS = n_thermal_cameras
        self.TOTAL_DEVICES = n_vibration_sensors + n_thermal_cameras
        self.VIBRATION_SENSORS = [f"VB-{i:03d}" for i in range(n_vibration_sensors)]
        self.THERMAL_CAMERAS = [f"TH-{i:03d}" for i in range(n_thermal_cameras)]

        self.SAMPLING_RATE_HZ = 2000
        self.WINDOW_SECONDS = 5
        self.SAMPLES_PER_BATCH = self.SAMPLING_RATE_HZ * self.WINDOW_SECONDS
        self.REPORT_INTERVAL_SECONDS = interval_seconds

        self.SURFACE_MAP = {}
        caves = [("C096", 0.55), ("C257", 0.25), ("C285", 0.20)]
        wall_types = ["N", "S", "E", "W", "C"]
        for i, sid in enumerate(self.VIBRATION_SENSORS):
            r = i / n_vibration_sensors
            acc = 0.0
            for cave_id, frac in caves:
                if r < acc + frac:
                    w = wall_types[random.randint(0, len(wall_types) - 1)]
                    self.SURFACE_MAP[sid] = f"{cave_id}-{w}"
                    break
                acc += frac

        for i, cid in enumerate(self.THERMAL_CAMERAS):
            r = i / n_thermal_cameras
            acc = 0.0
            for cave_id, frac in caves:
                if r < acc + frac:
                    w = wall_types[random.randint(0, len(wall_types) - 1)]
                    break
                acc += frac

        self.DELAMINATION_SURFACES = delamination_surfaces or ["C096-N", "C096-E", "C257-N"]


class VibrationSignalGenerator:
    def __init__(self, n_samples: int, fs: float = 2000.0):
        self.n_samples = n_samples
        self.fs = fs
        self.healthy_modes = [2.5, 7.8, 15.3, 22.1, 31.0]
        self.t = np.arange(n_samples) / fs

    def generate_healthy(self, seed: Optional[int] = None) -> Dict[str, List[float]]:
        rng = np.random.RandomState(seed)
        output = {}
        for axis in ("x", "y", "z"):
            sig = np.zeros(self.n_samples)
            for idx, f in enumerate(self.healthy_modes):
                amp = 0.15 / (idx + 1) * (1 + rng.rand() * 0.2)
                phase = rng.rand() * 2 * np.pi
                sig += amp * np.sin(2 * np.pi * f * self.t + phase)
            sig += 0.02 * rng.randn(self.n_samples)
            floor_vibration = 0.05 * np.sin(2 * np.pi * (1 + rng.rand() * 0.3) * self.t + rng.rand() * np.pi)
            sig += floor_vibration
            output[axis] = sig.tolist()
        return output

    def generate_delaminated(self, severity: float = 0.3,
                              seed: Optional[int] = None) -> Dict[str, List[float]]:
        rng = np.random.RandomState(seed)
        freq_shift = 1.0 - (0.02 + severity * 0.12)
        amp_factor = 1.0 + severity * 2.0
        output = {}
        for axis in ("x", "y", "z"):
            sig = np.zeros(self.n_samples)
            for idx, f in enumerate(self.healthy_modes):
                amp = 0.15 / (idx + 1) * amp_factor * (1 + rng.rand() * 0.3)
                phase = rng.rand() * 2 * np.pi
                shifted_freq = f * freq_shift
                sig += amp * np.sin(2 * np.pi * shifted_freq * self.t + phase)
            if severity > 0.25:
                abnormal_freq = 1.8 + severity * 3.5
                sig += 0.08 * severity * np.sin(2 * np.pi * abnormal_freq * self.t)
            num_impulses = int(severity * 5)
            for _ in range(num_impulses):
                impulse_time = rng.rand() * self.t[-1]
                impulse_idx = int(impulse_time * self.fs)
                decay = int(min(self.fs * 0.02, self.n_samples - impulse_idx))
                if decay > 0 and impulse_idx + decay <= self.n_samples:
                    sig[impulse_idx:impulse_idx + decay] += (
                        severity * 0.5 * np.exp(-np.arange(decay) / (decay * 0.3)) * rng.randn(decay)
                    )
            sig += 0.03 * rng.randn(self.n_samples)
            output[axis] = sig.tolist()
        return output


class ThermalImageGenerator:
    def __init__(self, width: int = 64, height: int = 48):
        self.width = width
        self.height = height

    def generate(self, camera_id: str,
                 base_temp: float = 18.0,
                 has_hotspot: bool = False,
                 hotspot_severity: float = 0.0) -> Dict:
        rng = np.random.RandomState(hash(camera_id) % (2 ** 31))
        temp_matrix = base_temp + rng.randn(self.height, self.width) * 0.5
        hotspot_regions = []
        if has_hotspot:
            n_hotspots = max(1, int(hotspot_severity * 5))
            for _ in range(n_hotspots):
                cx = int(rng.rand() * self.width)
                cy = int(rng.rand() * self.height)
                r = 3 + int(hotspot_severity * 10)
                max_delta = 3.0 + hotspot_severity * 8.0
                for y in range(max(0, cy - r), min(self.height, cy + r + 1)):
                    for x in range(max(0, cx - r), min(self.width, cx + r + 1)):
                        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                        if dist <= r:
                            delta = max_delta * (1 - dist / r) ** 2
                            temp_matrix[y, x] += delta
                hotspot_regions.append({
                    "x": cx, "y": cy, "radius": r,
                    "max_temp": float(base_temp + max_delta),
                    "severity": float(hotspot_severity),
                })
        return {
            "camera_id": camera_id,
            "max_temp": float(np.max(temp_matrix)),
            "min_temp": float(np.min(temp_matrix)),
            "avg_temp": float(np.mean(temp_matrix)),
            "temperature_matrix": temp_matrix.tolist(),
            "hotspot_regions": hotspot_regions,
        }


class DelaminationInjector:
    """支持动态注入：可通过环境变量或 HTTP 接口触发某区域剥离扩展"""

    def __init__(self, config: FiveGSimulatorConfig):
        self.config = config
        self.manual_severity_overrides: Dict[str, float] = {}
        self._load_env_overrides()

    def _load_env_overrides(self):
        override_str = os.environ.get("MOGAO_DELAMINATION_INJECT", "")
        if not override_str:
            return
        try:
            for part in override_str.split(","):
                if "=" in part:
                    surface, sev = part.split("=", 1)
                    self.manual_severity_overrides[surface.strip()] = float(sev.strip())
            logger.info("从环境变量加载剥离注入: {overrides}", overrides=self.manual_severity_overrides)
        except Exception as e:
            logger.warning("环境变量剥离注入解析失败: {e}", e=e)

    def inject(self, surface_id: str, severity: float):
        self.manual_severity_overrides[surface_id] = np.clip(severity, 0, 1.0)
        logger.warning("注入剥离: {sid} → sev={sev:.2f}", sid=surface_id, sev=severity)
        m.DELAMINATIONS_DETECTED.labels("5g_simulator", "injected").inc()

    def get_effective_severity(self, sensor_id: str, base_severity: float) -> float:
        surface = self.config.SURFACE_MAP.get(sensor_id, "")
        if surface in self.manual_severity_overrides:
            return self.manual_severity_overrides[surface]
        return base_severity


class FiveGDataSimulator:
    def __init__(self, config: FiveGSimulatorConfig):
        self.config = config
        self.interval = config.REPORT_INTERVAL_SECONDS
        self.vibration_gen = VibrationSignalGenerator(
            n_samples=config.SAMPLES_PER_BATCH, fs=config.SAMPLING_RATE_HZ,
        )
        self.thermal_gen = ThermalImageGenerator()
        self.injector = DelaminationInjector(config)

        self.vib_severity_map: Dict[str, float] = {}
        for sid in config.VIBRATION_SENSORS:
            surface = config.SURFACE_MAP.get(sid, "")
            if surface in config.DELAMINATION_SURFACES:
                self.vib_severity_map[sid] = random.uniform(0.15, 0.65)
            else:
                self.vib_severity_map[sid] = random.uniform(0.0, 0.08)

        self.thermal_hotspot_map: Dict[str, float] = {}
        for cid in config.THERMAL_CAMERAS:
            if random.random() < 0.35:
                self.thermal_hotspot_map[cid] = random.uniform(0.1, 0.5)

    async def send_vibration_batch(self, session: aiohttp.ClientSession, batch_idx: int) -> bool:
        timestamp = datetime.now(timezone.utc).isoformat()
        sensors_data = {}
        surface_stats: Dict[str, int] = {}

        for sensor_id in self.config.VIBRATION_SENSORS:
            base_severity = self.vib_severity_map[sensor_id]
            drift_factor = 1.0 + 0.0001 * batch_idx * random.uniform(-1, 1)
            base_severity = float(np.clip(base_severity * drift_factor, 0, 0.95))
            self.vib_severity_map[sensor_id] = base_severity
            severity = self.injector.get_effective_severity(sensor_id, base_severity)

            seed_base = hash(sensor_id + str(batch_idx)) % (2 ** 31)
            if severity > 0.08:
                signal = self.vibration_gen.generate_delaminated(severity=severity, seed=seed_base)
            else:
                signal = self.vibration_gen.generate_healthy(seed=seed_base)
            sensors_data[sensor_id] = signal
            surface = self.config.SURFACE_MAP.get(sensor_id, "?")
            surface_stats[surface] = surface_stats.get(surface, 0) + 1

        payload = {"timestamp": timestamp, "sensors": sensors_data}
        latency = random.uniform(8, 35)
        await asyncio.sleep(latency / 1000.0)

        t0 = time.perf_counter()
        try:
            async with session.post(
                self.config.VIBRATION_ENDPOINT,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
                headers={"X-5G-Edge-Node": "DUNHUANG-MOGAO-EDGE-01"},
            ) as resp:
                duration = time.perf_counter() - t0
                if resp.status == 200:
                    body = await resp.json()
                    m.VIBRATION_BATCHES_INGESTED.labels("5g_simulator").inc(body.get("records_stored", 0))
                    m.API_REQUEST_DURATION.labels("5g_simulator", "POST", "vibration_ingest", "200").observe(duration)
                    logger.info(
                        "[批次#{bid:04d}] 振动数据上报成功 | 5G延迟={lat:.1f}ms | 传感器={n}台 | {stats}",
                        bid=batch_idx, lat=latency,
                        n=body.get("records_stored", 0), stats=surface_stats,
                    )
                    return True
                else:
                    logger.warning(
                        "[批次#{bid:04d}] 振动数据上报失败: HTTP {status}",
                        bid=batch_idx, status=resp.status,
                    )
                    return False
        except Exception as e:
            logger.opt(exception=True).error(f"[批次#{batch_idx:04d}] 振动数据上报异常")
            return False

    async def send_thermal_batch(self, session: aiohttp.ClientSession, batch_idx: int) -> int:
        timestamp = datetime.now(timezone.utc).isoformat()
        success = 0
        for ci, camera_id in enumerate(self.config.THERMAL_CAMERAS):
            severity = self.thermal_hotspot_map.get(camera_id, 0.0)
            base_temp = 18.0 + random.uniform(-2, 5)
            img = self.thermal_gen.generate(
                camera_id=camera_id, base_temp=base_temp,
                has_hotspot=severity > 0, hotspot_severity=severity,
            )
            img["timestamp"] = timestamp
            packet_loss = random.random() < 0.005
            if packet_loss:
                logger.info("模拟5G丢包: 热成像相机 {cid}", cid=camera_id)
                continue
            try:
                async with session.post(
                    self.config.THERMAL_ENDPOINT,
                    json=img,
                    timeout=aiohttp.ClientTimeout(total=60),
                    headers={"X-5G-Edge-Node": "DUNHUANG-MOGAO-EDGE-01"},
                ) as resp:
                    if resp.status == 200:
                        success += 1
                        m.THERMAL_IMAGES_INGESTED.labels("5g_simulator").inc()
            except Exception as e:
                logger.warning("热成像 {cid} 上报异常: {e}", cid=camera_id, e=e)
            await asyncio.sleep(random.uniform(2, 8) / 1000.0)
        logger.info(
            "[批次#{bid:04d}] 热成像上报完成 {s}/{n} 台",
            bid=batch_idx, s=success, n=len(self.config.THERMAL_CAMERAS),
        )
        return success

    async def run(self, num_batches: Optional[int] = None):
        logger.info("=" * 70)
        logger.info("5G数据上报模拟器启动")
        logger.info("振动传感器: {n} 台", n=len(self.config.VIBRATION_SENSORS))
        logger.info("热成像相机: {n} 台", n=len(self.config.THERMAL_CAMERAS))
        logger.info("总设备数: {n} 台", n=self.config.TOTAL_DEVICES)
        logger.info("采样频率: {n} Hz", n=self.config.SAMPLING_RATE_HZ)
        logger.info("窗口长度: {w}s ({n} 采样点)", w=self.config.WINDOW_SECONDS, n=self.config.SAMPLES_PER_BATCH)
        logger.info("上报间隔: {s}s ({mode})",
                    s=self.interval, mode="快速模式" if self.interval < 60 else "实际模式")
        logger.info("上报端点: {url}", url=self.config.VIBRATION_ENDPOINT)
        logger.info("剥离模拟墙面: {surfaces}", surfaces=self.config.DELAMINATION_SURFACES)
        if self.injector.manual_severity_overrides:
            logger.warning("手动剥离注入: {overrides}", overrides=self.injector.manual_severity_overrides)
        logger.info("=" * 70)

        batch_idx = 0
        timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=120, sock_connect=30)
        connector = aiohttp.TCPConnector(limit=50, force_close=False)

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            while True:
                try:
                    await self.send_vibration_batch(session, batch_idx)
                    await self.send_thermal_batch(session, batch_idx)
                except Exception as e:
                    logger.opt(exception=True).error("批次上报异常")

                batch_idx += 1
                if num_batches is not None and batch_idx >= num_batches:
                    logger.info("达到指定批次数量 {n}, 模拟器退出", n=num_batches)
                    break
                logger.info("等待 {s}s 后上报下一批数据...", s=self.interval)
                await asyncio.sleep(self.interval)


def parse_args():
    parser = argparse.ArgumentParser(description="5G边缘节点数据上报模拟器")
    parser.add_argument("--api-base", type=str, default=None, help="数据接入服务Base URL")
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "fast", "debug"], help="运行模式")
    parser.add_argument("--interval-seconds", type=int, default=None,
                        help="自定义上报间隔秒数(覆盖mode)")
    parser.add_argument("--batches", type=int, default=None, help="运行指定批次后退出")
    parser.add_argument("--devices", type=int, default=80, help="总设备数(振动+热成像,默认80)")
    parser.add_argument("--vibration-sensors", type=int, default=60, help="振动传感器台数")
    parser.add_argument("--thermal-cameras", type=int, default=20, help="热成像相机台数")
    parser.add_argument("--inject-delamination", type=str, default=None,
                        help="注入剥离扩展,格式: C096-N=0.7,C257-S=0.5")
    return parser.parse_args()


def main():
    args = parse_args()
    n_vib = args.vibration_sensors
    n_therm = args.thermal_cameras
    if args.devices != 80:
        total = args.devices
        n_vib = int(total * 0.75)
        n_therm = total - n_vib

    interval = args.interval_seconds
    if interval is None:
        interval = {"standard": 3600, "fast": 10, "debug": 5}[args.mode]

    api_base = args.api_base or os.environ.get("MOGAO_INGEST_URL")

    config = FiveGSimulatorConfig(
        api_base=api_base,
        n_vibration_sensors=n_vib,
        n_thermal_cameras=n_therm,
        interval_seconds=interval,
    )
    simulator = FiveGDataSimulator(config)

    if args.inject_delamination:
        for pair in args.inject_delamination.split(","):
            if "=" in pair:
                sid, sev = pair.split("=", 1)
                simulator.injector.inject(sid.strip(), float(sev.strip()))

    asyncio.run(simulator.run(num_batches=args.batches))


if __name__ == "__main__":
    main()
