from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from backend.models import get_db
from backend.schemas.models import (
    DiffusionSimulateRequest, DiffusionResponse, EffectivenessAssessmentResponse,
    InjectionRateOptimizationRequest, InjectionRateOptimizationResponse,
    PressureFlowCurvePoint, DryingShrinkagePredictionRequest, DryingShrinkagePredictionResponse,
    ShrinkageTimeCurve, FormulationComparisonResponse, FormulationComparisonItem,
    CuringScheduleOptimizationRequest, CuringScheduleOptimizationResponse,
    OptimalCuringSchedule,
)
from backend.services.data_processing import GroutingAnalysisService
from backend.algorithms.grouting_diffusion import (
    generate_simulated_pressure_flow_data,
    PressureFlowPolynomialRegressor,
    PressureFlowDataPoint,
)
from backend.algorithms.drying_shrinkage import (
    AHTDryingShrinkageModel,
    compare_formulations,
    optimize_curing_schedule,
    GROUT_FORMULATIONS,
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/grouting", tags=["灌浆分析与加固评估"])

grouting_service = GroutingAnalysisService()


@router.post("/simulate-diffusion", response_model=List[DiffusionResponse])
async def simulate_grouting_diffusion(
    req: DiffusionSimulateRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        results = await grouting_service.simulate_diffusion(
            db, req.task_id, req.elapsed_seconds
        )
        return [DiffusionResponse(**r) for r in results]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"灌浆扩散模拟失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assess-effectiveness/{task_id}", response_model=EffectivenessAssessmentResponse)
async def assess_reinforcement(
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    try:
        result = await grouting_service.assess_effectiveness(db, task_id)
        return EffectivenessAssessmentResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"加固效果评估失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/effectiveness-history/{surface_id}")
async def get_effectiveness_history(
    surface_id: str,
    db: AsyncSession = Depends(get_db),
):
    from backend.models.database import ReinforcementEffectiveness
    from sqlalchemy import select, desc

    stmt = (
        select(ReinforcementEffectiveness)
        .where(ReinforcementEffectiveness.surface_id == surface_id)
        .order_by(desc(ReinforcementEffectiveness.time))
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        {
            "time": r.time.isoformat() if r.time else None,
            "task_id": r.task_id,
            "frequency_recovery_pct": float(r.frequency_recovery_pct) if r.frequency_recovery_pct else None,
            "delamination_area_reduction_pct": float(r.delamination_area_reduction_pct) if r.delamination_area_reduction_pct else None,
            "bonding_strength_mpa": float(r.bonding_strength_mpa) if r.bonding_strength_mpa else None,
            "overall_score": float(r.overall_score) if r.overall_score else None,
            "assessment_notes": r.assessment_notes,
        }
        for r in rows
    ]


@router.post("/optimize-rate", response_model=InjectionRateOptimizationResponse)
async def optimize_injection_rate(
    req: InjectionRateOptimizationRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        if req.measured_data and len(req.measured_data) > 0:
            data_points = [
                PressureFlowDataPoint(
                    pressure_kpa=dp.pressure_kpa,
                    flow_rate_mls=dp.flow_rate_mls,
                    elapsed_seconds=dp.elapsed_seconds,
                    temperature_c=dp.temperature_c,
                    is_reliable=dp.is_reliable,
                )
                for dp in req.measured_data
            ]
            data_source = "measured"
        elif req.use_simulated_data:
            data_points = generate_simulated_pressure_flow_data(
                n_points=req.simulation_point_count,
            )
            data_source = "simulated"
        else:
            raise HTTPException(
                status_code=400,
                detail="必须提供实测数据或启用模拟数据"
            )
        
        regressor = PressureFlowPolynomialRegressor(
            max_degree=req.max_polynomial_degree,
            delamination_threshold_pressure_kpa=req.delamination_threshold_pressure_kpa,
            max_flow_rate_mls=req.max_flow_rate_mls,
        )
        regressor.fit(data_points)
        
        pressure_range = (100.0, 500.0)
        if req.pressure_range_kpa and len(req.pressure_range_kpa) >= 2:
            pressure_range = (float(req.pressure_range_kpa[0]), float(req.pressure_range_kpa[1]))
        
        result = regressor.optimize_injection_rate(
            pressure_range=pressure_range,
            target_radius_mm=req.target_radius_mm,
            max_allowable_risk_pct=req.max_allowable_risk_pct,
        )
        
        from backend.models.database import GroutingPressureFlowData
        db_record = GroutingPressureFlowData(
            time=datetime.utcnow(),
            measurement_id=str(uuid.uuid4()),
            task_id=req.grouting_task_id,
            surface_id=req.surface_id,
            pressure_kpa=result.optimal_pressure_kpa,
            flow_rate_mls=result.optimal_flow_rate_mls,
            temperature_c=data_points[0].temperature_c if data_points else None,
            is_reliable=True,
            data_source=data_source,
            polynomial_degree=result.polynomial_degree,
            r_squared=result.r_squared,
            optimal_pressure_kpa=result.optimal_pressure_kpa,
            optimal_flow_rate_mls=result.optimal_flow_rate_mls,
            secondary_delamination_risk_pct=result.secondary_delamination_risk_pct,
        )
        db.add(db_record)
        await db.commit()
        
        return InjectionRateOptimizationResponse(
            surface_id=req.surface_id,
            grouting_task_id=req.grouting_task_id,
            optimal_pressure_kpa=result.optimal_pressure_kpa,
            optimal_flow_rate_mls=result.optimal_flow_rate_mls,
            polynomial_degree=result.polynomial_degree,
            polynomial_coefficients=result.polynomial_coefficients,
            r_squared=result.r_squared,
            secondary_delamination_risk_pct=result.secondary_delamination_risk_pct,
            recommended_max_flow_mls=result.recommended_max_flow_mls,
            pressure_flow_curve=[
                PressureFlowCurvePoint(
                    pressure_kpa=p["pressure_kpa"],
                    flow_rate_mls=p["flow_rate_mls"],
                    delamination_risk_pct=p["delamination_risk_pct"],
                )
                for p in result.pressure_flow_curve
            ],
            warning_message=result.warning_message,
            data_source=data_source,
            data_points_count=len(data_points),
            processing_timestamp=datetime.utcnow().isoformat() + "Z",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"注浆速率优化失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-shrinkage", response_model=DryingShrinkagePredictionResponse)
async def predict_drying_shrinkage(
    req: DryingShrinkagePredictionRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        if req.formulation_id not in GROUT_FORMULATIONS:
            raise HTTPException(
                status_code=400,
                detail=f"未知配方ID: {req.formulation_id}, 可用配方: {list(GROUT_FORMULATIONS.keys())}"
            )
        
        model = AHTDryingShrinkageModel(
            formulation_id=req.formulation_id,
            initial_curing_days=req.initial_curing_days,
            prediction_horizon_days=req.prediction_horizon_days,
        )
        
        result = model.predict(
            ambient_temperature_c=req.ambient_temperature_c,
            ambient_humidity_pct=req.ambient_humidity_pct,
            constraint_factor=req.constraint_factor,
            wall_thickness_mm=req.wall_thickness_mm,
        )
        
        from backend.models.database import DryingShrinkagePrediction
        db_record = DryingShrinkagePrediction(
            time=datetime.utcnow(),
            prediction_id=str(uuid.uuid4()),
            formulation_id=result["formulation_id"],
            formulation_name=result["formulation_name"],
            ambient_temperature_c=result["ambient_temperature_c"],
            ambient_humidity_pct=result["ambient_humidity_pct"],
            constraint_factor=result["constraint_factor"],
            wall_thickness_mm=result["wall_thickness_mm"],
            final_shrinkage_strain=result["final_shrinkage_strain"],
            final_tensile_stress_mpa=result["final_tensile_stress_mpa"],
            crack_risk_index=result["crack_risk_index"],
            predicted_crack_width_mm=result["predicted_crack_width_mm"],
            crack_risk_level=result["crack_risk_level"],
            critical_period_days=result["critical_period_days"],
            tensile_strength_mpa=result["tensile_strength_mpa"],
            elastic_modulus_mpa=result["elastic_modulus_mpa"],
            shrinkage_time_curve=result["shrinkage_time_curve"],
            recommendations=result["recommendations"],
            prediction_horizon_days=req.prediction_horizon_days,
        )
        db.add(db_record)
        await db.commit()
        
        return DryingShrinkagePredictionResponse(
            formulation_id=result["formulation_id"],
            formulation_name=result["formulation_name"],
            ambient_temperature_c=result["ambient_temperature_c"],
            ambient_humidity_pct=result["ambient_humidity_pct"],
            constraint_factor=result["constraint_factor"],
            wall_thickness_mm=result["wall_thickness_mm"],
            final_shrinkage_strain=result["final_shrinkage_strain"],
            final_tensile_stress_mpa=result["final_tensile_stress_mpa"],
            crack_risk_index=result["crack_risk_index"],
            predicted_crack_width_mm=result["predicted_crack_width_mm"],
            crack_risk_level=result["crack_risk_level"],
            critical_period_days=result["critical_period_days"],
            tensile_strength_mpa=result["tensile_strength_mpa"],
            elastic_modulus_mpa=result["elastic_modulus_mpa"],
            shrinkage_time_curve=ShrinkageTimeCurve(
                days=result["shrinkage_time_curve"]["days"],
                shrinkage_strain=result["shrinkage_time_curve"]["shrinkage_strain"],
                tensile_stress_mpa=result["shrinkage_time_curve"]["tensile_stress_mpa"],
                crack_risk_index=result["shrinkage_time_curve"]["crack_risk_index"],
                hardening_degree=result["shrinkage_time_curve"]["hardening_degree"],
            ),
            recommendations=result["recommendations"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"干燥收缩预测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/formulations", response_model=List[Dict])
async def list_grout_formulations():
    try:
        return [
            {
                "formulation_id": fid,
                "name": data["name"],
                "water_binder_ratio": data["water_binder_ratio"],
                "ps_concentration_pct": data["ps_concentration_pct"],
                "fly_ash_pct": data["fly_ash_pct"],
                "tensile_strength_mpa": data["tensile_strength_mpa"],
                "elastic_modulus_mpa": data["elastic_modulus_mpa"],
                "final_shrinkage_strain": data["final_shrinkage_strain"],
            }
            for fid, data in GROUT_FORMULATIONS.items()
        ]
    except Exception as e:
        logger.error(f"获取配方列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare-formulations", response_model=FormulationComparisonResponse)
async def compare_grout_formulations(
    ambient_temperature_c: float = 20.0,
    ambient_humidity_pct: float = 60.0,
    constraint_factor: float = 0.7,
):
    try:
        results = compare_formulations(
            ambient_temperature_c=ambient_temperature_c,
            ambient_humidity_pct=ambient_humidity_pct,
            constraint_factor=constraint_factor,
        )
        
        return FormulationComparisonResponse(
            ambient_temperature_c=ambient_temperature_c,
            ambient_humidity_pct=ambient_humidity_pct,
            formulations=[
                FormulationComparisonItem(
                    formulation_id=r["formulation_id"],
                    formulation_name=r["formulation_name"],
                    shrinkage_strain=r["shrinkage_strain"],
                    crack_risk_index=r["crack_risk_index"],
                    crack_risk_level=r["crack_risk_level"],
                    tensile_strength_mpa=r["tensile_strength_mpa"],
                )
                for r in results
            ],
        )
    except Exception as e:
        logger.error(f"配方对比失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-curing-schedule", response_model=CuringScheduleOptimizationResponse)
async def optimize_curing_schedule_endpoint(
    req: CuringScheduleOptimizationRequest,
):
    try:
        if req.formulation_id not in GROUT_FORMULATIONS:
            raise HTTPException(
                status_code=400,
                detail=f"未知配方ID: {req.formulation_id}, 可用配方: {list(GROUT_FORMULATIONS.keys())}"
            )
        
        result = optimize_curing_schedule(
            ambient_temperature_c=req.ambient_temperature_c,
            ambient_humidity_pct=req.ambient_humidity_pct,
            formulation_id=req.formulation_id,
            target_risk_index=req.target_risk_index,
        )
        
        optimal_schedule = None
        if result.get("optimal_schedule"):
            opt = result["optimal_schedule"]
            optimal_schedule = OptimalCuringSchedule(
                curing_humidity_pct=opt["curing_humidity_pct"],
                curing_duration_days=opt["curing_duration_days"],
                achieved_risk_index=opt["achieved_risk_index"],
                predicted_crack_width_mm=opt["predicted_crack_width_mm"],
                note=opt.get("note"),
            )
        
        return CuringScheduleOptimizationResponse(
            optimization_needed=result["optimization_needed"],
            baseline_risk=result["baseline_risk"],
            target_risk=result["target_risk"],
            recommendation=result["recommendation"],
            optimal_schedule=optimal_schedule,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"养护制度优化失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pressure-flow-history/{task_id}")
async def get_pressure_flow_history(
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    from backend.models.database import GroutingPressureFlowData
    from sqlalchemy import select, desc

    stmt = (
        select(GroutingPressureFlowData)
        .where(GroutingPressureFlowData.task_id == task_id)
        .order_by(desc(GroutingPressureFlowData.time))
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        {
            "time": r.time.isoformat() if r.time else None,
            "measurement_id": r.measurement_id,
            "surface_id": r.surface_id,
            "pressure_kpa": float(r.pressure_kpa),
            "flow_rate_mls": float(r.flow_rate_mls),
            "elapsed_seconds": float(r.elapsed_seconds) if r.elapsed_seconds else None,
            "temperature_c": float(r.temperature_c) if r.temperature_c else None,
            "is_reliable": r.is_reliable,
            "data_source": r.data_source,
            "optimal_pressure_kpa": float(r.optimal_pressure_kpa) if r.optimal_pressure_kpa else None,
            "optimal_flow_rate_mls": float(r.optimal_flow_rate_mls) if r.optimal_flow_rate_mls else None,
            "secondary_delamination_risk_pct": float(r.secondary_delamination_risk_pct) if r.secondary_delamination_risk_pct else None,
        }
        for r in rows
    ]


@router.get("/shrinkage-history/{surface_id}")
async def get_shrinkage_history(
    surface_id: str,
    db: AsyncSession = Depends(get_db),
):
    from backend.models.database import DryingShrinkagePrediction
    from sqlalchemy import select, desc

    stmt = (
        select(DryingShrinkagePrediction)
        .where(DryingShrinkagePrediction.surface_id == surface_id)
        .order_by(desc(DryingShrinkagePrediction.time))
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        {
            "time": r.time.isoformat() if r.time else None,
            "prediction_id": r.prediction_id,
            "task_id": r.task_id,
            "formulation_id": r.formulation_id,
            "formulation_name": r.formulation_name,
            "ambient_temperature_c": float(r.ambient_temperature_c),
            "ambient_humidity_pct": float(r.ambient_humidity_pct),
            "final_shrinkage_strain": float(r.final_shrinkage_strain),
            "crack_risk_index": float(r.crack_risk_index),
            "predicted_crack_width_mm": float(r.predicted_crack_width_mm),
            "crack_risk_level": r.crack_risk_level,
            "critical_period_days": float(r.critical_period_days) if r.critical_period_days else None,
        }
        for r in rows
    ]
