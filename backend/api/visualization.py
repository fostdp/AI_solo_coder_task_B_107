from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, and_, text
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from backend.models import get_db
from backend.models.database import (
    Cave, WallSurface, VibrationSensor, ThermalCamera,
    DelaminationRegion, ModalAnalysisResult, GroutingTask,
    GroutingDiffusion, Alert, VibrationRawData, ThermalImage,
)

router = APIRouter(prefix="/visualization", tags=["三维可视化数据"])


@router.get("/cave-3d/{cave_id}")
async def get_cave_3d_data(
    cave_id: str,
    include_sensors: bool = True,
    include_cameras: bool = True,
    include_delamination: bool = True,
    include_grouting: bool = True,
    db: AsyncSession = Depends(get_db),
):
    cave_stmt = select(Cave).where(Cave.cave_id == cave_id)
    cave_res = await db.execute(cave_stmt)
    cave = cave_res.scalar_one_or_none()
    if not cave:
        raise HTTPException(status_code=404, detail="洞窟不存在")

    surfaces_stmt = select(WallSurface).where(WallSurface.cave_id == cave_id)
    surfaces_res = await db.execute(surfaces_stmt)
    surfaces = surfaces_res.scalars().all()

    result: Dict[str, Any] = {
        "cave": {
            "cave_id": cave.cave_id,
            "cave_name": cave.cave_name,
            "dynasty": cave.dynasty,
            "description": cave.description,
            "dimensions": cave.dimensions if cave.dimensions else {},
        },
        "walls": [],
    }

    for surf in surfaces:
        wall_data = {
            "surface_id": surf.surface_id,
            "wall_type": surf.wall_type,
            "area_sqm": float(surf.area_sqm) if surf.area_sqm else 0,
            "bounding_box_3d": surf.bounding_box_3d if surf.bounding_box_3d else {},
        }

        if include_sensors:
            vs_stmt = select(VibrationSensor).where(VibrationSensor.surface_id == surf.surface_id)
            vs_res = await db.execute(vs_stmt)
            wall_data["vibration_sensors"] = [
                {
                    "sensor_id": s.sensor_id,
                    "location_3d": s.location_3d,
                    "status": s.status,
                    "sampling_rate_hz": s.sampling_rate_hz,
                }
                for s in vs_res.scalars().all()
            ]

        if include_cameras:
            tc_stmt = select(ThermalCamera).where(ThermalCamera.surface_id == surf.surface_id)
            tc_res = await db.execute(tc_stmt)
            wall_data["thermal_cameras"] = [
                {
                    "camera_id": c.camera_id,
                    "location_3d": c.location_3d,
                    "status": c.status,
                }
                for c in tc_res.scalars().all()
            ]

        if include_delamination:
            dl_stmt = (
                select(DelaminationRegion)
                .where(
                    and_(
                        DelaminationRegion.surface_id == surf.surface_id,
                        DelaminationRegion.is_active == True,
                    )
                )
                .order_by(desc(DelaminationRegion.time))
                .distinct(DelaminationRegion.region_id)
            )
            dl_res = await db.execute(dl_stmt)
            regions = dl_res.scalars().all()
            seen_ids = set()
            unique_regions = []
            for r in regions:
                if r.region_id not in seen_ids:
                    seen_ids.add(r.region_id)
                    unique_regions.append({
                        "region_id": r.region_id,
                        "time": r.time.isoformat() if r.time else None,
                        "bounding_polygon_3d": r.bounding_polygon_3d,
                        "area_sqm": float(r.area_sqm) if r.area_sqm else 0,
                        "depth_mm": float(r.depth_mm) if r.depth_mm else 0,
                        "severity_score": float(r.severity_score) if r.severity_score else 0,
                        "confidence": float(r.confidence) if r.confidence else 0,
                        "frequency_drop_pct": float(r.frequency_drop_pct) if r.frequency_drop_pct else 0,
                    })
            wall_data["delamination_regions"] = unique_regions

        if include_grouting:
            gt_stmt = select(GroutingTask).where(GroutingTask.surface_id == surf.surface_id)
            gt_res = await db.execute(gt_stmt)
            tasks = gt_res.scalars().all()
            wall_data["grouting_tasks"] = []
            for t in tasks:
                gd_stmt = (
                    select(GroutingDiffusion)
                    .where(GroutingDiffusion.task_id == t.task_id)
                    .order_by(desc(GroutingDiffusion.time))
                    .limit(10)
                )
                gd_res = await db.execute(gd_stmt)
                diffusions = gd_res.scalars().all()

                wall_data["grouting_tasks"].append({
                    "task_id": t.task_id,
                    "status": t.status,
                    "start_time": t.start_time.isoformat() if t.start_time else None,
                    "material_type": t.material_type,
                    "operator": t.operator,
                    "injection_points": t.injection_points if t.injection_points else [],
                    "pressure_kpa": float(t.pressure_kpa) if t.pressure_kpa else None,
                    "latest_diffusions": [
                        {
                            "injection_point_id": d.injection_point_id,
                            "predicted_radius_mm": float(d.predicted_radius_mm) if d.predicted_radius_mm else 0,
                            "penetration_depth_mm": float(d.penetration_depth_mm) if d.penetration_depth_mm else 0,
                            "elapsed_seconds": d.elapsed_seconds,
                            "diffusion_front": d.diffusion_front if d.diffusion_front else [],
                            "particle_pathlines": d.particle_pathlines if d.particle_pathlines else [],
                        }
                        for d in diffusions
                    ],
                })

        result["walls"].append(wall_data)

    return result


@router.get("/realtime/sensor-readings/{cave_id}")
async def get_realtime_sensor_data(
    cave_id: str,
    hours: int = Query(24, ge=1, le=72),
    db: AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(hours=hours)

    surf_stmt = select(WallSurface.surface_id).where(WallSurface.cave_id == cave_id)
    surf_res = await db.execute(surf_stmt)
    surface_ids = [r[0] for r in surf_res.all()]

    v_summary_stmt = (
        select(
            VibrationSensor.sensor_id,
            VibrationSensor.location_3d,
            VibrationSensor.surface_id,
            func.max(VibrationRawData.time).label("last_time"),
            func.count(VibrationRawData.time).label("records_count"),
        )
        .select_from(VibrationSensor)
        .outerjoin(
            VibrationRawData,
            and_(
                VibrationRawData.sensor_id == VibrationSensor.sensor_id,
                VibrationRawData.time >= since,
            ),
        )
        .where(VibrationSensor.surface_id.in_(surface_ids))
        .group_by(VibrationSensor.sensor_id)
    )
    v_res = await db.execute(v_summary_stmt)

    t_summary_stmt = (
        select(
            ThermalCamera.camera_id,
            ThermalCamera.location_3d,
            ThermalCamera.surface_id,
            func.max(ThermalImage.time).label("last_time"),
            func.max(ThermalImage.max_temp).label("max_temp_recent"),
            func.avg(ThermalImage.avg_temp).label("avg_temp_recent"),
        )
        .select_from(ThermalCamera)
        .outerjoin(
            ThermalImage,
            and_(
                ThermalImage.camera_id == ThermalCamera.camera_id,
                ThermalImage.time >= since,
            ),
        )
        .where(ThermalCamera.surface_id.in_(surface_ids))
        .group_by(ThermalCamera.camera_id)
    )
    t_res = await db.execute(t_summary_stmt)

    freq_stmt = (
        select(
            ModalAnalysisResult.surface_id,
            ModalAnalysisResult.natural_frequencies,
            ModalAnalysisResult.time,
        )
        .select_from(ModalAnalysisResult)
        .where(
            and_(
                ModalAnalysisResult.surface_id.in_(surface_ids),
                ModalAnalysisResult.time >= since,
            )
        )
        .order_by(desc(ModalAnalysisResult.time))
    )
    f_res = await db.execute(freq_stmt)
    freq_map: Dict[str, Any] = {}
    for r in f_res.all():
        if r.surface_id not in freq_map and r.natural_frequencies:
            freq_map[r.surface_id] = {
                "frequencies": [float(x) for x in r.natural_frequencies],
                "time": r.time.isoformat() if r.time else None,
            }

    return {
        "cave_id": cave_id,
        "time_window_hours": hours,
        "vibration_sensors": [
            {
                "sensor_id": r.sensor_id,
                "surface_id": r.surface_id,
                "location_3d": r.location_3d,
                "last_reading_time": r.last_time.isoformat() if r.last_time else None,
                "records_count": int(r.records_count or 0),
            }
            for r in v_res.all()
        ],
        "thermal_cameras": [
            {
                "camera_id": r.camera_id,
                "surface_id": r.surface_id,
                "location_3d": r.location_3d,
                "last_reading_time": r.last_time.isoformat() if r.last_time else None,
                "max_temp_recent": float(r.max_temp_recent) if r.max_temp_recent else None,
                "avg_temp_recent": float(r.avg_temp_recent) if r.avg_temp_recent else None,
            }
            for r in t_res.all()
        ],
        "modal_by_surface": freq_map,
    }
