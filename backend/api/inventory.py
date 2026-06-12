from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import uuid

from backend.models.database import (
    Cave, WallSurface, VibrationSensor, ThermalCamera,
    DelaminationRegion, ModalAnalysisResult, GroutingTask,
    Alert, GroutingDiffusion, ReinforcementEffectiveness,
    VisitorFlowStats, CavePriorityRanking, BondStrengthAssessment,
)
from backend.models import get_db
from backend.schemas.models import (
    CaveResponse, WallSurfaceResponse, VibrationSensorResponse,
    ThermalCameraResponse, DelaminationRegionResponse,
    GroutingTaskResponse, AlertResponse, SystemStatusResponse,
    CavePriorityRankingRequest, CavePriorityRankingResponse,
    CavePriorityRankingItem, VisitorFlowStatsRequest, VisitorFlowStatsResponse,
)
from backend.algorithms.priority_ranking import (
    CaveProtectionPriorityRanking,
    CaveConditionData,
    generate_simulated_visitor_flow,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inventory", tags=["基础数据管理"])


@router.get("/caves", response_model=List[CaveResponse])
async def list_caves(db: AsyncSession = Depends(get_db)):
    stmt = select(Cave).order_by(Cave.cave_id)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/caves/{cave_id}", response_model=CaveResponse)
async def get_cave(cave_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Cave).where(Cave.cave_id == cave_id)
    result = await db.execute(stmt)
    cave = result.scalar_one_or_none()
    if not cave:
        raise HTTPException(status_code=404, detail="洞窟不存在")
    return cave


@router.get("/caves/{cave_id}/surfaces", response_model=List[WallSurfaceResponse])
async def list_cave_surfaces(cave_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(WallSurface).where(WallSurface.cave_id == cave_id)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/surfaces", response_model=List[WallSurfaceResponse])
async def list_surfaces(
    cave_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(WallSurface)
    if cave_id:
        stmt = stmt.where(WallSurface.cave_id == cave_id)
    stmt = stmt.order_by(WallSurface.surface_id)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/vibration-sensors", response_model=List[VibrationSensorResponse])
async def list_vibration_sensors(
    cave_id: Optional[str] = None,
    surface_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(VibrationSensor)
    if cave_id:
        stmt = stmt.where(VibrationSensor.cave_id == cave_id)
    if surface_id:
        stmt = stmt.where(VibrationSensor.surface_id == surface_id)
    stmt = stmt.order_by(VibrationSensor.sensor_id)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/thermal-cameras", response_model=List[ThermalCameraResponse])
async def list_thermal_cameras(
    cave_id: Optional[str] = None,
    surface_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ThermalCamera)
    if cave_id:
        stmt = stmt.where(ThermalCamera.cave_id == cave_id)
    if surface_id:
        stmt = stmt.where(ThermalCamera.surface_id == surface_id)
    stmt = stmt.order_by(ThermalCamera.camera_id)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/grouting-tasks", response_model=List[GroutingTaskResponse])
async def list_grouting_tasks(
    cave_id: Optional[str] = None,
    surface_id: Optional[str] = None,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(GroutingTask)
    if cave_id:
        stmt = stmt.where(GroutingTask.cave_id == cave_id)
    if surface_id:
        stmt = stmt.where(GroutingTask.surface_id == surface_id)
    if status:
        stmt = stmt.where(GroutingTask.status == status)
    stmt = stmt.order_by(desc(GroutingTask.start_time))
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/grouting-tasks/{task_id}", response_model=GroutingTaskResponse)
async def get_grouting_task(task_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(GroutingTask).where(GroutingTask.task_id == task_id)
    result = await db.execute(stmt)
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="灌浆任务不存在")
    return task


@router.get("/alerts", response_model=List[AlertResponse])
async def list_alerts(
    cave_id: Optional[str] = None,
    surface_id: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = "active",
    days: int = Query(7, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Alert)
    if cave_id:
        stmt = stmt.where(Alert.cave_id == cave_id)
    if surface_id:
        stmt = stmt.where(Alert.surface_id == surface_id)
    if severity:
        stmt = stmt.where(Alert.severity == severity)
    if status:
        stmt = stmt.where(Alert.status == status)
    stmt = stmt.where(Alert.created_at >= datetime.utcnow() - timedelta(days=days))
    stmt = stmt.order_by(desc(Alert.created_at))
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/status", response_model=SystemStatusResponse)
async def system_status(db: AsyncSession = Depends(get_db)):
    v_stmt = select(func.count(VibrationSensor.sensor_id)).where(VibrationSensor.status == "active")
    v_count = (await db.execute(v_stmt)).scalar() or 0

    t_stmt = select(func.count(ThermalCamera.camera_id)).where(ThermalCamera.status == "active")
    t_count = (await db.execute(t_stmt)).scalar() or 0

    a_stmt = select(func.count(Alert.alert_id)).where(Alert.status == "active")
    a_count = (await db.execute(a_stmt)).scalar() or 0

    g_stmt = select(func.count(GroutingTask.task_id)).where(GroutingTask.status == "in_progress")
    g_count = (await db.execute(g_stmt)).scalar() or 0

    last_stmt = select(func.max(ModalAnalysisResult.time))
    last_time = (await db.execute(last_stmt)).scalar()

    return {
        "status": "ok",
        "vibration_sensors": v_count,
        "thermal_cameras": t_count,
        "active_alerts": a_count,
        "active_grouting_tasks": g_count,
        "last_processing_time": last_time.isoformat() if last_time else None,
    }


@router.post("/simulate-visitor-flow", response_model=VisitorFlowStatsResponse)
async def simulate_visitor_flow(
    req: VisitorFlowStatsRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        result = generate_simulated_visitor_flow(
            cave_id=req.cave_id,
            days=req.days,
            base_daily=req.base_daily_visitors,
        )
        
        from datetime import datetime as dt
        for date_str, count in zip(result["dates"], result["daily_visitors"]):
            d = dt.fromisoformat(date_str)
            is_weekend = d.weekday() >= 5
            is_peak_season = 5 <= d.month <= 10
            is_holiday = (d.month == 10 and 1 <= d.day <= 7) or (d.month == 5 and 1 <= d.day <= 5)
            
            db_record = VisitorFlowStats(
                date=d,
                cave_id=req.cave_id,
                daily_visitors=count,
                is_weekend=is_weekend,
                is_peak_season=is_peak_season,
                is_holiday=is_holiday,
            )
            db.add(db_record)
        await db.commit()
        
        return VisitorFlowStatsResponse(
            cave_id=result["cave_id"],
            avg_daily=result["avg_daily"],
            peak_daily=result["peak_daily"],
            min_daily=result["min_daily"],
            total_annual=result["total_annual"],
            peak_season_avg=result["peak_season_avg"],
            off_peak_avg=result["off_peak_avg"],
            daily_visitors=result["daily_visitors"],
            dates=result["dates"],
        )
    except Exception as e:
        logger.error(f"游客流量模拟失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visitor-flow/{cave_id}")
async def get_visitor_flow(
    cave_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    try:
        stmt = select(VisitorFlowStats).where(VisitorFlowStats.cave_id == cave_id)
        
        if start_date:
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            stmt = stmt.where(VisitorFlowStats.date >= start_dt)
        if end_date:
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            stmt = stmt.where(VisitorFlowStats.date <= end_dt)
        
        stmt = stmt.order_by(desc(VisitorFlowStats.date))
        result = await db.execute(stmt)
        rows = result.scalars().all()
        
        if not rows:
            return {
                "cave_id": cave_id,
                "has_data": False,
                "records": [],
                "summary": None,
            }
        
        visitor_counts = [r.daily_visitors for r in rows]
        summary = {
            "avg_daily": round(float(sum(visitor_counts) / len(visitor_counts)), 2),
            "peak_daily": max(visitor_counts),
            "min_daily": min(visitor_counts),
            "total": sum(visitor_counts),
            "record_count": len(rows),
        }
        
        return {
            "cave_id": cave_id,
            "has_data": True,
            "records": [
                {
                    "date": r.date.isoformat() if r.date else None,
                    "daily_visitors": r.daily_visitors,
                    "is_weekend": r.is_weekend,
                    "is_peak_season": r.is_peak_season,
                    "is_holiday": r.is_holiday,
                    "temperature_c": float(r.temperature_c) if r.temperature_c else None,
                    "humidity_pct": float(r.humidity_pct) if r.humidity_pct else None,
                    "special_event": r.special_event,
                }
                for r in rows
            ],
            "summary": summary,
        }
    except Exception as e:
        logger.error(f"获取游客流量数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rank-caves-priority", response_model=CavePriorityRankingResponse)
async def rank_caves_priority(
    req: CavePriorityRankingRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        caves_data = []
        for cave in req.caves:
            caves_data.append(CaveConditionData(
                cave_id=cave.cave_id,
                cave_name=cave.cave_name,
                total_delamination_area_sqm=cave.total_delamination_area_sqm,
                max_severity_score=cave.max_severity_score,
                historical_repair_count=cave.historical_repair_count,
                last_repair_years_ago=cave.last_repair_years_ago,
                avg_visitor_flow_daily=cave.avg_visitor_flow_daily,
                peak_visitor_flow_daily=cave.peak_visitor_flow_daily,
                cultural_significance_score=cave.cultural_significance_score,
                structural_importance=cave.structural_importance,
                avg_bond_strength_remaining_mpa=cave.avg_bond_strength_remaining_mpa,
                active_alerts_count=cave.active_alerts_count,
                wall_surfaces_count=cave.wall_surfaces_count,
            ))
        
        ranker = CaveProtectionPriorityRanking(weights=req.weights)
        results = ranker.rank_caves(caves_data, use_nsga_ii=req.use_nsga_ii)
        
        ranking_id = str(uuid.uuid4())
        pareto_front_size = None
        
        for res in results:
            nsga_summary = res.get("nsga_ii_summary")
            if nsga_summary and pareto_front_size is None:
                pareto_front_size = nsga_summary.get("pareto_front_size")
            
            db_record = CavePriorityRanking(
                time=datetime.utcnow(),
                ranking_id=ranking_id,
                cave_id=res["cave_id"],
                priority_rank=res["priority_rank"],
                total_caves=len(results),
                urgency_score=res["urgency_score"],
                cost_efficiency_score=res["cost_efficiency_score"],
                cultural_impact_score=res["cultural_impact_score"],
                aggregated_score=res["aggregated_score"],
                weights_used=res["weights_used"],
                method=res["method"],
                pareto_rank=res.get("pareto_rank"),
                crowding_distance=res.get("crowding_distance"),
                pareto_front_size=pareto_front_size,
                nsga_ii_summary=res.get("nsga_ii_summary"),
                input_metrics=res.get("detail_metrics"),
            )
            db.add(db_record)
        await db.commit()
        
        ranking_items = []
        for res in results:
            ranking_items.append(CavePriorityRankingItem(
                cave_id=res["cave_id"],
                cave_name=res["cave_name"],
                priority_rank=res["priority_rank"],
                urgency_score=res["urgency_score"],
                cost_efficiency_score=res["cost_efficiency_score"],
                cultural_impact_score=res["cultural_impact_score"],
                aggregated_score=res["aggregated_score"],
                weights_used=res["weights_used"],
                method=res["method"],
                pareto_rank=res.get("pareto_rank"),
                crowding_distance=res.get("crowding_distance"),
                nsga_ii_summary=res.get("nsga_ii_summary"),
                detail_metrics=res.get("detail_metrics", {}),
            ))
        
        return CavePriorityRankingResponse(
            ranking=ranking_items,
            method=results[0]["method"] if results else "unknown",
            total_caves=len(results),
            timestamp=datetime.utcnow().isoformat() + "Z",
            pareto_front_size=pareto_front_size,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"窟室优先级排序失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rank-caves-priority-from-db")
async def rank_caves_priority_from_db(
    use_nsga_ii: bool = True,
    db: AsyncSession = Depends(get_db),
):
    try:
        caves_stmt = select(Cave).order_by(Cave.cave_id)
        caves_result = await db.execute(caves_stmt)
        caves = caves_result.scalars().all()
        
        if not caves:
            raise HTTPException(status_code=404, detail="数据库中无洞窟数据")
        
        caves_data = []
        for cave in caves:
            surfaces_stmt = select(WallSurface).where(WallSurface.cave_id == cave.cave_id)
            surfaces_res = await db.execute(surfaces_stmt)
            surfaces = surfaces_res.scalars().all()
            
            total_area = 0.0
            max_severity = 0.0
            surface_ids = []
            for s in surfaces:
                surface_ids.append(s.surface_id)
                delam_stmt = (
                    select(DelaminationRegion)
                    .where(
                        and_(
                            DelaminationRegion.surface_id == s.surface_id,
                            DelaminationRegion.is_active == True,
                            DelaminationRegion.time >= datetime.utcnow() - timedelta(days=30),
                        )
                    )
                    .order_by(desc(DelaminationRegion.time))
                    .limit(5)
                )
                delam_res = await db.execute(delam_stmt)
                delams = delam_res.scalars().all()
                for d in delams:
                    total_area += float(d.area_sqm) if d.area_sqm else 0.0
                    if d.severity_score and float(d.severity_score) > max_severity:
                        max_severity = float(d.severity_score)
            
            repair_stmt = (
                select(func.count(GroutingTask.task_id))
                .where(GroutingTask.cave_id == cave.cave_id)
            )
            repair_count = (await db.execute(repair_stmt)).scalar() or 0
            
            last_repair_stmt = (
                select(func.max(GroutingTask.end_time))
                .where(GroutingTask.cave_id == cave.cave_id)
            )
            last_repair = (await db.execute(last_repair_stmt)).scalar()
            last_repair_years = 10.0
            if last_repair:
                last_repair_years = (datetime.utcnow() - last_repair).days / 365.0
            
            visitor_stmt = (
                select(func.avg(VisitorFlowStats.daily_visitors))
                .where(
                    and_(
                        VisitorFlowStats.cave_id == cave.cave_id,
                        VisitorFlowStats.date >= datetime.utcnow() - timedelta(days=90),
                    )
                )
            )
            avg_visitors = (await db.execute(visitor_stmt)).scalar() or 500.0
            
            peak_visitor_stmt = (
                select(func.max(VisitorFlowStats.daily_visitors))
                .where(
                    and_(
                        VisitorFlowStats.cave_id == cave.cave_id,
                        VisitorFlowStats.date >= datetime.utcnow() - timedelta(days=365),
                    )
                )
            )
            peak_visitors = (await db.execute(peak_visitor_stmt)).scalar() or 2000.0
            
            avg_bond_strength = 0.8
            if surface_ids:
                bond_stmt = (
                    select(BondStrengthAssessment)
                    .where(BondStrengthAssessment.surface_id.in_(surface_ids))
                    .order_by(desc(BondStrengthAssessment.time))
                    .limit(5)
                )
                bond_res = await db.execute(bond_stmt)
                bond_rows = bond_res.scalars().all()
                if bond_rows:
                    strengths = [float(r.remaining_bond_strength_mpa) for r in bond_rows if r.remaining_bond_strength_mpa]
                    if strengths:
                        avg_bond_strength = sum(strengths) / len(strengths)
            
            alerts_stmt = (
                select(func.count(Alert.alert_id))
                .where(
                    and_(
                        Alert.cave_id == cave.cave_id,
                        Alert.status == "active",
                    )
                )
            )
            active_alerts = (await db.execute(alerts_stmt)).scalar() or 0
            
            if cave.cave_id == "C096":
                cultural_significance = 1.0
                structural_importance = 0.95
            elif cave.cave_id == "C257":
                cultural_significance = 0.95
                structural_importance = 0.9
            elif cave.cave_id == "C285":
                cultural_significance = 1.0
                structural_importance = 0.85
            else:
                cultural_significance = 0.8
                structural_importance = 0.8
            
            caves_data.append(CaveConditionData(
                cave_id=cave.cave_id,
                cave_name=cave.cave_name,
                total_delamination_area_sqm=round(total_area, 4),
                max_severity_score=round(max_severity, 2),
                historical_repair_count=int(repair_count),
                last_repair_years_ago=round(last_repair_years, 2),
                avg_visitor_flow_daily=float(avg_visitors),
                peak_visitor_flow_daily=float(peak_visitors),
                cultural_significance_score=cultural_significance,
                structural_importance=structural_importance,
                avg_bond_strength_remaining_mpa=round(avg_bond_strength, 6),
                active_alerts_count=int(active_alerts),
                wall_surfaces_count=len(surface_ids) if surface_ids else 5,
            ))
        
        ranker = CaveProtectionPriorityRanking()
        results = ranker.rank_caves(caves_data, use_nsga_ii=use_nsga_ii)
        
        ranking_id = str(uuid.uuid4())
        pareto_front_size = None
        
        for res in results:
            nsga_summary = res.get("nsga_ii_summary")
            if nsga_summary and pareto_front_size is None:
                pareto_front_size = nsga_summary.get("pareto_front_size")
            
            db_record = CavePriorityRanking(
                time=datetime.utcnow(),
                ranking_id=ranking_id,
                cave_id=res["cave_id"],
                priority_rank=res["priority_rank"],
                total_caves=len(results),
                urgency_score=res["urgency_score"],
                cost_efficiency_score=res["cost_efficiency_score"],
                cultural_impact_score=res["cultural_impact_score"],
                aggregated_score=res["aggregated_score"],
                weights_used=res["weights_used"],
                method=res["method"],
                pareto_rank=res.get("pareto_rank"),
                crowding_distance=res.get("crowding_distance"),
                pareto_front_size=pareto_front_size,
                nsga_ii_summary=res.get("nsga_ii_summary"),
                input_metrics=res.get("detail_metrics"),
            )
            db.add(db_record)
        await db.commit()
        
        return {
            "ranking_id": ranking_id,
            "ranking": results,
            "method": results[0]["method"] if results else "unknown",
            "total_caves": len(results),
            "pareto_front_size": pareto_front_size,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"基于数据库的窟室优先级排序失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/priority-ranking-history")
async def get_priority_ranking_history(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    try:
        stmt = (
            select(CavePriorityRanking)
            .where(CavePriorityRanking.time >= datetime.utcnow() - timedelta(days=days))
            .order_by(desc(CavePriorityRanking.time))
        )
        result = await db.execute(stmt)
        rows = result.scalars().all()
        
        grouped = {}
        for r in rows:
            rid = r.ranking_id
            if rid not in grouped:
                grouped[rid] = {
                    "ranking_id": rid,
                    "time": r.time.isoformat() if r.time else None,
                    "method": r.method,
                    "total_caves": r.total_caves,
                    "pareto_front_size": r.pareto_front_size,
                    "results": [],
                }
            grouped[rid]["results"].append({
                "cave_id": r.cave_id,
                "priority_rank": r.priority_rank,
                "aggregated_score": float(r.aggregated_score) if r.aggregated_score else None,
                "urgency_score": float(r.urgency_score) if r.urgency_score else None,
                "cost_efficiency_score": float(r.cost_efficiency_score) if r.cost_efficiency_score else None,
                "cultural_impact_score": float(r.cultural_impact_score) if r.cultural_impact_score else None,
                "weights_used": r.weights_used,
                "pareto_rank": r.pareto_rank,
                "crowding_distance": float(r.crowding_distance) if r.crowding_distance else None,
            })
        
        return sorted(grouped.values(), key=lambda x: x["time"], reverse=True)
    except Exception as e:
        logger.error(f"获取优先级排序历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
