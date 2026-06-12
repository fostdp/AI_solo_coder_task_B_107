import numpy as np
from fastapi import APIRouter, HTTPException
from typing import List, Tuple, Optional
import logging

from backend.algorithms.grouting_diffusion import (
    PressureFlowPolynomialRegressor,
    PressureFlowDataPoint,
)
from .schemas import (
    PressureFlowFitRequest,
    PressureFlowFitResponse,
    PressureFlowPredictRequest,
    PressureFlowPredictResponse,
    PressureFlowOptimizeRequest,
    InjectionRateOptimizationResultSchema,
    PressureFlowCurvePointSchema,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pressure-regression", tags=["压力流量多项式回归"])


@router.post("/fit", response_model=PressureFlowFitResponse)
async def fit_model(req: PressureFlowFitRequest):
    try:
        if len(req.data_points) < 3:
            raise HTTPException(
                status_code=400,
                detail="至少需要3个数据点才能进行拟合"
            )

        data_points = [
            PressureFlowDataPoint(
                pressure_kpa=dp.pressure_kpa,
                flow_rate_mls=dp.flow_rate_mls,
                elapsed_seconds=0.0,
                temperature_c=None,
                is_reliable=dp.is_reliable,
            )
            for dp in req.data_points
        ]

        regressor = PressureFlowPolynomialRegressor(
            max_degree=req.max_degree,
            ridge_alpha=req.ridge_alpha,
        )
        regressor.fit(data_points)

        return PressureFlowFitResponse(
            r_squared=round(regressor.r_squared, 6),
            optimal_degree=int(regressor.optimal_degree),
            coefficients=[round(float(c), 8) for c in regressor.coefficients],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模型拟合失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PressureFlowPredictResponse)
async def predict_flow(req: PressureFlowPredictRequest):
    try:
        if len(req.coefficients) != req.optimal_degree + 1:
            raise HTTPException(
                status_code=400,
                detail=f"系数数量 ({len(req.coefficients)}) 必须等于多项式阶数 + 1 ({req.optimal_degree + 1})"
            )

        regressor = PressureFlowPolynomialRegressor(max_degree=req.optimal_degree)
        regressor.optimal_degree = req.optimal_degree
        regressor.coefficients = np.array(req.coefficients, dtype=np.float64)

        predictions = regressor.predict(np.array(req.pressures_kpa, dtype=np.float64))

        return PressureFlowPredictResponse(
            predictions=[round(float(p), 6) for p in predictions]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"流量预测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize", response_model=InjectionRateOptimizationResultSchema)
async def optimize_injection_rate(req: PressureFlowOptimizeRequest):
    try:
        if len(req.pressure_range) != 2:
            raise HTTPException(
                status_code=400,
                detail="pressure_range 必须包含2个值：[min_pressure, max_pressure]"
            )
        if req.pressure_range[0] >= req.pressure_range[1]:
            raise HTTPException(
                status_code=400,
                detail="pressure_range 的第一个值必须小于第二个值"
            )
        if len(req.model_coefficients) != req.optimal_degree + 1:
            raise HTTPException(
                status_code=400,
                detail=f"系数数量 ({len(req.model_coefficients)}) 必须等于多项式阶数 + 1 ({req.optimal_degree + 1})"
            )

        regressor = PressureFlowPolynomialRegressor(
            max_degree=req.optimal_degree,
            delamination_threshold_pressure_kpa=req.delamination_threshold_pressure_kpa,
            max_flow_rate_mls=req.max_flow_rate_mls,
        )
        regressor.optimal_degree = req.optimal_degree
        regressor.coefficients = np.array(req.model_coefficients, dtype=np.float64)
        regressor.r_squared = float(req.r_squared) if req.r_squared is not None else 0.0

        pressure_range = (float(req.pressure_range[0]), float(req.pressure_range[1]))
        result = regressor.optimize_injection_rate(
            pressure_range=pressure_range,
            target_radius_mm=req.target_radius_mm,
            max_allowable_risk_pct=req.max_allowable_risk_pct,
        )

        return InjectionRateOptimizationResultSchema(
            optimal_pressure_kpa=result.optimal_pressure_kpa,
            optimal_flow_rate_mls=result.optimal_flow_rate_mls,
            polynomial_degree=result.polynomial_degree,
            polynomial_coefficients=result.polynomial_coefficients,
            r_squared=result.r_squared,
            secondary_delamination_risk_pct=result.secondary_delamination_risk_pct,
            recommended_max_flow_mls=result.recommended_max_flow_mls,
            pressure_flow_curve=[
                PressureFlowCurvePointSchema(
                    pressure_kpa=p["pressure_kpa"],
                    flow_rate_mls=p["flow_rate_mls"],
                    delamination_risk_pct=p["delamination_risk_pct"],
                )
                for p in result.pressure_flow_curve
            ],
            warning_message=result.warning_message,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"注浆速率优化失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
