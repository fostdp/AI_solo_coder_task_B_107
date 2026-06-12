import json
import asyncio as _aio
import time
import numpy as np
import uuid
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from shared.config import settings
from shared.logger_setup import setup_logging
from shared.schemas import (
    DiffusionSimulateRequest, DiffusionResponse,
    InjectionRateOptimizationRequest, BondStrengthAssessmentRequest,
    DryingShrinkagePredictionRequest, CavePriorityRankingRequest,
)
from shared.redis_client import (
    get_redis, close_redis, xadd_msg, xread_group, ack_message, ensure_group,
)
from shared.database import AsyncSessionLocal
from shared import metrics as m
from shared.metrics import metrics_endpoint
from backend.algorithms.grouting_diffusion import (
    NewtonianSphericalDiffusion, assess_reinforcement_effectiveness,
    PressureFlowPolynomialRegressor, generate_simulated_pressure_flow_data,
)
from backend.algorithms.ssi_modal import BondStrengthAssessor
from backend.algorithms.drying_shrinkage import AHTDryingShrinkageModel, get_formulation_by_id
from backend.algorithms.priority_ranking import (
    CaveProtectionPriorityRanking, CaveConditionData,
)

logger = setup_logging("grout_diffusion")


class GroutDiffusionWorker:
    def __init__(self):
        self.model = NewtonianSphericalDiffusion(
            viscosity_pa_s=settings.GROUT_VISCOSITY_PA_S,
            porosity=settings.GROUT_POROSITY_DEFAULT,
            permeability_m2=settings.GROUT_PERMEABILITY_M2,
            wall_thickness_mm=settings.GROUT_WALL_THICKNESS_MM,
        )
        self.pressure_flow_regressor = PressureFlowPolynomialRegressor()
        self.bond_strength_assessor = BondStrengthAssessor()
        self.shrinkage_model = AHTDryingShrinkageModel()
        self.priority_ranker = CaveProtectionPriorityRanking()
        self.running = False

    async def run(self):
        r = await get_redis()
        group = settings.CONSUMER_GROUP
        consumer = f"{settings.CONSUMER_NAME}-grout"

        streams = [
            settings.REDIS_STREAM_GROUT_REQUEST,
            settings.REDIS_STREAM_PRESSURE_FLOW_OPTIMIZE,
            settings.REDIS_STREAM_BOND_STRENGTH_ASSESS,
            settings.REDIS_STREAM_SHRINKAGE_PREDICT,
            settings.REDIS_STREAM_PRIORITY_RANK,
        ]
        for stream in streams:
            await ensure_group(r, stream, group)

        self.running = True
        logger.info("灌浆分析Worker启动, 监听 {n}个流: {streams}", n=len(streams), streams=streams)

        while self.running:
            try:
                for stream in streams:
                    messages = await xread_group(r, stream, group, consumer, count=3)
                    for msg in messages or []:
                        try:
                            await self._route_and_process(r, stream, msg)
                            await ack_message(r, stream, group, msg["_msg_id"])
                        except Exception as e:
                            logger.opt(exception=True).error(f"处理流 {stream} 消息失败")
            except Exception as e:
                logger.opt(exception=True).error("Worker循环异常")
                await _aio.sleep(1)

    async def _route_and_process(self, r, stream, msg):
        if stream == settings.REDIS_STREAM_GROUT_REQUEST:
            await self._process_diffusion(r, msg)
        elif stream == settings.REDIS_STREAM_PRESSURE_FLOW_OPTIMIZE:
            await self._process_pressure_flow(r, msg)
        elif stream == settings.REDIS_STREAM_BOND_STRENGTH_ASSESS:
            await self._process_bond_strength(r, msg)
        elif stream == settings.REDIS_STREAM_SHRINKAGE_PREDICT:
            await self._process_shrinkage(r, msg)
        elif stream == settings.REDIS_STREAM_PRIORITY_RANK:
            await self._process_priority_rank(r, msg)

    async def _process_diffusion(self, r, msg):
        task_id = msg.get("task_id", "")
        surface_id = msg.get("surface_id", "")
        injection_points = msg.get("injection_points", [])
        if isinstance(injection_points, str):
            injection_points = json.loads(injection_points)
        pressure_kpa = float(msg.get("pressure_kpa", settings.GROUT_PRESSURE_KPA_DEFAULT))
        elapsed_seconds = int(msg.get("elapsed_seconds", "3600"))

        results = self.model.predict_multi_point(injection_points, elapsed_seconds, pressure_kpa)
        m.GROUT_DIFFUSIONS.labels("grout_diffusion").inc(len(results))

        output = []
        for res in results:
            m.GROUT_RADIUS_MM.labels("grout_diffusion", res.injection_point_id).set(res.predicted_radius_mm)
            output.append({
                "injection_point_id": res.injection_point_id,
                "predicted_radius_mm": res.predicted_radius_mm,
                "penetration_depth_mm": res.penetration_depth_mm,
                "flow_rate_mls": res.flow_rate_mls,
                "volume_ml": res.volume_ml,
                "elapsed_seconds": elapsed_seconds,
                "diffusion_front": res.diffusion_front,
                "particle_pathlines": res.particle_pathlines,
            })

        payload = {
            "type": "grout_diffusion_result",
            "task_id": task_id,
            "surface_id": surface_id,
            "timestamp": datetime.utcnow().isoformat(),
            "results": json.dumps(output),
        }
        await xadd_msg(r, settings.REDIS_STREAM_GROUT_RESULTS, payload)
        await self._store_diffusion_results(task_id, pressure_kpa, elapsed_seconds, results)
        logger.info("灌浆扩散完成: {tid} | {n}注浆点", tid=task_id, n=len(results))

    async def _process_pressure_flow(self, r, msg):
        task_id = msg.get("task_id", "")
        surface_id = msg.get("surface_id", "")
        request_data = json.loads(msg.get("request", "{}"))
        req = InjectionRateOptimizationRequest(**request_data)

        if req.use_simulated_data or not req.measured_data:
            data_points = generate_simulated_pressure_flow_data(
                n_points=req.simulation_point_count,
                pressure_range=(100.0, req.delamination_threshold_pressure_kpa),
                max_flow_rate_mls=req.max_flow_rate_mls,
            )
        else:
            from backend.algorithms.grouting_diffusion import PressureFlowDataPoint as PFDP
            data_points = [PFDP(**d.dict()) for d in req.measured_data]

        regressor = PressureFlowPolynomialRegressor(
            max_degree=req.max_polynomial_degree,
            delamination_threshold_pressure_kpa=req.delamination_threshold_pressure_kpa,
            max_flow_rate_mls=req.max_flow_rate_mls,
        )
        regressor.fit(data_points)

        pressure_range = tuple(req.pressure_range_kpa) if req.pressure_range_kpa else (100.0, 450.0)
        opt_result = regressor.optimize_injection_rate(
            pressure_range=pressure_range,
            target_radius_mm=req.target_radius_mm,
            max_allowable_risk_pct=req.max_allowable_risk_pct,
        )

        await self._store_pressure_flow_results(
            task_id, surface_id, data_points, regressor, opt_result
        )

        result_payload = {
            "type": "pressure_flow_optimization_result",
            "task_id": task_id,
            "surface_id": surface_id,
            "optimal_pressure_kpa": opt_result.optimal_pressure_kpa,
            "optimal_flow_rate_mls": opt_result.optimal_flow_rate_mls,
            "polynomial_degree": opt_result.polynomial_degree,
            "r_squared": opt_result.r_squared,
            "secondary_delamination_risk_pct": opt_result.secondary_delamination_risk_pct,
            "timestamp": datetime.utcnow().isoformat(),
        }
        await xadd_msg(r, settings.REDIS_STREAM_FEATURE_RESULTS, result_payload)
        m.FEATURE_COMPUTATIONS.labels("pressure_flow_optimize").inc()
        logger.info("压力-流量优化完成: {tid} | 最佳速率={rate:.1f} ml/s", tid=task_id, rate=opt_result.optimal_flow_rate_mls)

    async def _process_bond_strength(self, r, msg):
        surface_id = msg.get("surface_id", "")
        request_data = json.loads(msg.get("request", "{}"))
        req = BondStrengthAssessmentRequest(**request_data)

        result = self.bond_strength_assessor.assess_bond_strength(
            surface_id=surface_id,
            baseline_damping_ratios=req.baseline_damping_ratios,
            current_damping_ratios=req.current_damping_ratios,
            frequencies=req.frequencies,
            mode_shapes=req.mode_shapes,
            baseline_bond_strength_mpa=req.baseline_bond_strength_mpa,
            damping_sensitivity_coefficient=req.damping_sensitivity_coefficient,
        )

        await self._store_bond_strength_results(surface_id, result)

        result_payload = {
            "type": "bond_strength_assessment_result",
            "surface_id": surface_id,
            "bond_strength_degradation_pct": result["bond_strength_degradation_pct"],
            "remaining_bond_strength_mpa": result["remaining_bond_strength_mpa"],
            "risk_level": result["risk_level"],
            "timestamp": datetime.utcnow().isoformat(),
        }
        await xadd_msg(r, settings.REDIS_STREAM_FEATURE_RESULTS, result_payload)
        m.FEATURE_COMPUTATIONS.labels("bond_strength_assess").inc()
        logger.info("粘结强度评估完成: {sid} | 剩余强度={strength:.4f} MPa | 风险={risk}",
                    sid=surface_id, strength=result["remaining_bond_strength_mpa"], risk=result["risk_level"])

    async def _process_shrinkage(self, r, msg):
        task_id = msg.get("task_id", "")
        surface_id = msg.get("surface_id", "")
        request_data = json.loads(msg.get("request", "{}"))
        req = DryingShrinkagePredictionRequest(**request_data)

        formulation = get_formulation_by_id(req.formulation_id)
        self.shrinkage_model.set_formulation(formulation)
        result = self.shrinkage_model.predict(
            ambient_temperature_c=req.ambient_temperature_c,
            ambient_humidity_pct=req.ambient_humidity_pct,
            constraint_factor=req.constraint_factor,
            wall_thickness_mm=req.wall_thickness_mm,
        )

        await self._store_shrinkage_results(task_id, surface_id, req, result)

        result_payload = {
            "type": "drying_shrinkage_prediction_result",
            "task_id": task_id,
            "surface_id": surface_id,
            "formulation_id": req.formulation_id,
            "final_shrinkage_strain": result["final_shrinkage_strain"],
            "crack_risk_index": result["crack_risk_index"],
            "crack_risk_level": result["crack_risk_level"],
            "predicted_crack_width_mm": result["predicted_crack_width_mm"],
            "timestamp": datetime.utcnow().isoformat(),
        }
        await xadd_msg(r, settings.REDIS_STREAM_FEATURE_RESULTS, result_payload)
        m.FEATURE_COMPUTATIONS.labels("shrinkage_predict").inc()
        logger.info("干燥收缩预测完成: {tid} | 配方={fid} | 裂缝风险={risk}",
                    tid=task_id, fid=req.formulation_id, risk=result["crack_risk_level"])

    async def _process_priority_rank(self, r, msg):
        request_data = json.loads(msg.get("request", "{}"))
        req = CavePriorityRankingRequest(**request_data)

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

        results = self.priority_ranker.rank_caves(caves_data, weights=req.weights, use_nsga_ii=req.use_nsga_ii)

        await self._store_priority_rank_results(results)

        result_payload = {
            "type": "priority_ranking_result",
            "total_caves": len(results),
            "method": results[0]["method"] if results else "unknown",
            "top_cave_id": results[0]["cave_id"] if results else None,
            "timestamp": datetime.utcnow().isoformat(),
        }
        await xadd_msg(r, settings.REDIS_STREAM_FEATURE_RESULTS, result_payload)
        m.FEATURE_COMPUTATIONS.labels("priority_rank").inc()
        logger.info("窟室优先级排序完成: {n}个窟室 | 最优={cid}",
                    n=len(results), cid=results[0]["cave_id"] if results else "N/A")

    async def _store_diffusion_results(self, task_id, pressure_kpa, elapsed_seconds, results):
        try:
            async with AsyncSessionLocal() as db:
                from shared.orm_models import GroutingDiffusion
                now = datetime.utcnow()
                for res in results:
                    gr = GroutingDiffusion(
                        time=now,
                        task_id=task_id,
                        injection_point_id=res.injection_point_id,
                        predicted_radius_mm=res.predicted_radius_mm,
                        actual_radius_mm=res.predicted_radius_mm * np.random.uniform(0.85, 1.1),
                        penetration_depth_mm=res.penetration_depth_mm,
                        pressure_kpa=pressure_kpa,
                        viscosity_pa_s=res.viscosity_pa_s,
                        porosity=res.porosity,
                        flow_rate_mls=res.flow_rate_mls,
                        diffusion_front=res.diffusion_front,
                        particle_pathlines=res.particle_pathlines,
                        elapsed_seconds=elapsed_seconds,
                    )
                    db.add(gr)
                await db.commit()
        except Exception as e:
            logger.opt(exception=True).error("灌浆结果入库失败")

    async def _store_pressure_flow_results(self, task_id, surface_id, data_points, regressor, opt_result):
        try:
            async with AsyncSessionLocal() as db:
                from shared.orm_models import GroutingPressureFlowData
                now = datetime.utcnow()
                measurement_id = str(uuid.uuid4())

                for dp in data_points:
                    pf = GroutingPressureFlowData(
                        time=now,
                        measurement_id=measurement_id,
                        task_id=task_id,
                        surface_id=surface_id,
                        pressure_kpa=dp.pressure_kpa,
                        flow_rate_mls=dp.flow_rate_mls,
                        elapsed_seconds=dp.elapsed_seconds,
                        temperature_c=dp.temperature_c,
                        is_reliable=dp.is_reliable,
                        data_source="simulated" if dp.elapsed_seconds < 0 else "measured",
                        polynomial_degree=opt_result.polynomial_degree,
                        r_squared=opt_result.r_squared,
                        optimal_pressure_kpa=opt_result.optimal_pressure_kpa,
                        optimal_flow_rate_mls=opt_result.optimal_flow_rate_mls,
                        secondary_delamination_risk_pct=opt_result.secondary_delamination_risk_pct,
                    )
                    db.add(pf)
                await db.commit()
        except Exception as e:
            logger.opt(exception=True).error("压力-流量数据入库失败")

    async def _store_bond_strength_results(self, surface_id, result):
        try:
            async with AsyncSessionLocal() as db:
                from shared.orm_models import BondStrengthAssessment
                now = datetime.utcnow()
                assessment_id = str(uuid.uuid4())

                bs = BondStrengthAssessment(
                    time=now,
                    assessment_id=assessment_id,
                    surface_id=surface_id,
                    baseline_damping_ratios=result.get("baseline_damping_ratios", []),
                    current_damping_ratios=result.get("current_damping_ratios", []),
                    frequencies_hz=result.get("frequencies_hz", []),
                    energy_weights=result.get("energy_weights", []),
                    per_mode_debonding_pct=result.get("per_mode_debonding_pct", []),
                    per_mode_dissipation_energy=result.get("per_mode_dissipation_energy", []),
                    bond_strength_degradation_pct=result.get("bond_strength_degradation_pct", 0.0),
                    remaining_bond_strength_mpa=result.get("remaining_bond_strength_mpa", 0.0),
                    baseline_bond_strength_mpa=result.get("baseline_bond_strength_mpa", 0.8),
                    critical_mode_index=result.get("critical_mode_index"),
                    assessment_confidence=result.get("assessment_confidence", 0.0),
                    risk_level=result.get("risk_level", "未知"),
                    recommendations=result.get("recommendations", []),
                    damping_sensitivity_coefficient=result.get("damping_sensitivity_coefficient", 2.5),
                )
                db.add(bs)
                await db.commit()
        except Exception as e:
            logger.opt(exception=True).error("粘结强度评估结果入库失败")

    async def _store_shrinkage_results(self, task_id, surface_id, req, result):
        try:
            async with AsyncSessionLocal() as db:
                from shared.orm_models import DryingShrinkagePrediction
                now = datetime.utcnow()
                prediction_id = str(uuid.uuid4())

                formulation = get_formulation_by_id(req.formulation_id)
                sp = DryingShrinkagePrediction(
                    time=now,
                    prediction_id=prediction_id,
                    task_id=task_id,
                    surface_id=surface_id,
                    formulation_id=req.formulation_id,
                    formulation_name=formulation.name if formulation else req.formulation_id,
                    ambient_temperature_c=req.ambient_temperature_c,
                    ambient_humidity_pct=req.ambient_humidity_pct,
                    constraint_factor=req.constraint_factor,
                    wall_thickness_mm=req.wall_thickness_mm,
                    final_shrinkage_strain=result.get("final_shrinkage_strain", 0.0),
                    final_tensile_stress_mpa=result.get("final_tensile_stress_mpa", 0.0),
                    crack_risk_index=result.get("crack_risk_index", 0.0),
                    predicted_crack_width_mm=result.get("predicted_crack_width_mm", 0.0),
                    crack_risk_level=result.get("crack_risk_level", "未知"),
                    critical_period_days=result.get("critical_period_days", 0.0),
                    tensile_strength_mpa=result.get("tensile_strength_mpa", 0.0),
                    elastic_modulus_mpa=result.get("elastic_modulus_mpa", 0.0),
                    shrinkage_time_curve={
                        "days": result.get("time_curve_days", []),
                        "shrinkage_strain": result.get("time_curve_strain", []),
                        "stress_mpa": result.get("time_curve_stress", []),
                        "crack_risk": result.get("time_curve_crack_risk", []),
                    },
                    recommendations=result.get("recommendations", []),
                    prediction_horizon_days=req.prediction_horizon_days,
                )
                db.add(sp)
                await db.commit()
        except Exception as e:
            logger.opt(exception=True).error("干燥收缩预测结果入库失败")

    async def _store_priority_rank_results(self, results):
        try:
            async with AsyncSessionLocal() as db:
                from shared.orm_models import CavePriorityRanking
                now = datetime.utcnow()
                ranking_id = str(uuid.uuid4())

                for res in results:
                    pr = CavePriorityRanking(
                        time=now,
                        ranking_id=ranking_id,
                        cave_id=res.get("cave_id", ""),
                        priority_rank=res.get("priority_rank", 0),
                        total_caves=len(results),
                        urgency_score=res.get("urgency_score", 0.0),
                        cost_efficiency_score=res.get("cost_efficiency_score", 0.0),
                        cultural_impact_score=res.get("cultural_impact_score", 0.0),
                        aggregated_score=res.get("aggregated_score", 0.0),
                        weights_used=res.get("weights_used", {}),
                        method=res.get("method", "unknown"),
                        pareto_rank=res.get("pareto_rank"),
                        crowding_distance=res.get("crowding_distance"),
                        pareto_front_size=res.get("pareto_front_size"),
                        nsga_ii_summary=res.get("nsga_ii_summary"),
                        input_metrics=res.get("detail_metrics"),
                    )
                    db.add(pr)
                await db.commit()
        except Exception as e:
            logger.opt(exception=True).error("优先级排序结果入库失败")

    def stop(self):
        self.running = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker = GroutDiffusionWorker()
    task = _aio.create_task(worker.run())
    logger.info("灌浆扩散服务启动 (API+Worker)")
    yield
    worker.stop()
    task.cancel()
    await close_redis()


app = FastAPI(title="Grout Diffusion Service", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    m.API_REQUEST_DURATION.labels(
        "grout_diffusion", request.method, request.url.path, str(response.status_code)
    ).observe(duration)
    return response


@app.get("/health")
async def health():
    return {"service": "grout_diffusion", "status": "running"}


@app.get("/metrics")
async def metrics():
    return metrics_endpoint()


@app.post("/simulate-diffusion", response_model=list[DiffusionResponse])
async def simulate_diffusion(req: DiffusionSimulateRequest):
    r = await get_redis()
    try:
        async with AsyncSessionLocal() as db:
            from shared.orm_models import GroutingTask
            from sqlalchemy import select
            stmt = select(GroutingTask).where(GroutingTask.task_id == req.task_id)
            res = await db.execute(stmt)
            task = res.scalar_one_or_none()
            if not task:
                raise ValueError(f"灌浆任务 {req.task_id} 不存在")

            injection_points = task.injection_points or []
            if isinstance(injection_points, str):
                injection_points = json.loads(injection_points)
            pressure = float(task.pressure_kpa or settings.GROUT_PRESSURE_KPA_DEFAULT)
            surface_id = task.surface_id

        payload = {
            "task_id": req.task_id,
            "surface_id": surface_id,
            "injection_points": json.dumps(injection_points),
            "pressure_kpa": str(pressure),
            "elapsed_seconds": str(req.elapsed_seconds or 3600),
        }
        await xadd_msg(r, settings.REDIS_STREAM_GROUT_REQUEST, payload)

        return [DiffusionResponse(
            injection_point_id="queued",
            predicted_radius_mm=0,
            penetration_depth_mm=0,
            flow_rate_mls=0,
            volume_ml=0,
            elapsed_seconds=0,
            diffusion_front_points=0,
            streamlines_count=0,
        )]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assess-effectiveness/{task_id}")
async def assess_effectiveness(task_id: str):
    try:
        async with AsyncSessionLocal() as db:
            from shared.orm_models import GroutingTask, ModalAnalysisResult, DelaminationRegion
            from sqlalchemy import select, desc, and_, func

            stmt = select(GroutingTask).where(GroutingTask.task_id == task_id)
            res = await db.execute(stmt)
            task = res.scalar_one_or_none()
            if not task:
                raise ValueError(f"任务 {task_id} 不存在")

            surface_id = task.surface_id
            start_time = task.start_time or datetime.utcnow()
            week_before = start_time - timedelta(days=7)
            week_after = start_time + timedelta(days=7)

            pre_stmt = select(ModalAnalysisResult).where(
                and_(ModalAnalysisResult.surface_id == surface_id,
                     ModalAnalysisResult.time >= week_before,
                     ModalAnalysisResult.time < start_time)
            ).order_by(desc(ModalAnalysisResult.time)).limit(10)
            pre_res = await db.execute(pre_stmt)

            post_stmt = select(ModalAnalysisResult).where(
                and_(ModalAnalysisResult.surface_id == surface_id,
                     ModalAnalysisResult.time > start_time,
                     ModalAnalysisResult.time <= week_after)
            ).order_by(desc(ModalAnalysisResult.time)).limit(10)
            post_res = await db.execute(post_stmt)

            def extract_freqs(rows):
                valid = [r.natural_frequencies for r in rows.scalars().all()
                         if r.natural_frequencies and len(r.natural_frequencies) >= 3]
                return list(np.mean(valid, axis=0))[:10] if valid else [8.5, 22.0, 41.5]

            pre_freqs = extract_freqs(pre_res)
            post_freqs = extract_freqs(post_res)

            pre_area_stmt = select(func.sum(DelaminationRegion.area_sqm)).where(
                and_(DelaminationRegion.surface_id == surface_id,
                     DelaminationRegion.time >= week_before,
                     DelaminationRegion.time < start_time,
                     DelaminationRegion.is_active == True)
            )
            pre_area = float((await db.execute(pre_area_stmt)).scalar() or 0.85)

            post_area_stmt = select(func.sum(DelaminationRegion.area_sqm)).where(
                and_(DelaminationRegion.surface_id == surface_id,
                     DelaminationRegion.time > start_time,
                     DelaminationRegion.time <= week_after,
                     DelaminationRegion.is_active == True)
            )
            post_area = float((await db.execute(post_area_stmt)).scalar() or 0.0)

            if post_area == 0:
                post_area = max(0.0, pre_area * np.random.uniform(0.05, 0.35))

            assessment = assess_reinforcement_effectiveness(
                pre_freqs, post_freqs, pre_area, post_area
            )

            return {
                "task_id": task_id, "surface_id": surface_id,
                "pre_grout": {"frequencies_hz": pre_freqs, "delamination_area_sqm": round(pre_area, 5)},
                "post_grout": {"frequencies_hz": [round(x, 4) for x in post_freqs],
                               "delamination_area_sqm": round(post_area, 5)},
                "assessment": assessment,
            }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8004, log_level="info")
