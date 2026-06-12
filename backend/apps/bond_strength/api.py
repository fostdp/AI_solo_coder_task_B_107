import logging
import numpy as np
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException

from backend.algorithms.ssi_modal import BondStrengthAssessor
from backend.apps.bond_strength.schemas import (
    BondStrengthAssessRequest,
    BondStrengthAssessResponse,
    TemperatureCalibrateRequest,
    TemperatureCalibrateResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/bond_strength", tags=["粘结强度评估"])

assessor = BondStrengthAssessor()


@router.post("/assess", response_model=BondStrengthAssessResponse)
async def assess_bond_strength_endpoint(req: BondStrengthAssessRequest):
    try:
        if len(req.baseline_damping_ratios) != len(req.current_damping_ratios):
            raise HTTPException(
                status_code=400,
                detail=f"基线阻尼比数量({len(req.baseline_damping_ratios)})与当前阻尼比数量({len(req.current_damping_ratios)})不匹配"
            )

        if len(req.baseline_damping_ratios) == 0:
            raise HTTPException(
                status_code=400,
                detail="阻尼比列表不能为空"
            )

        if req.frequencies is not None and len(req.frequencies) != len(req.baseline_damping_ratios):
            raise HTTPException(
                status_code=400,
                detail=f"频率列表长度({len(req.frequencies)})与阻尼比列表长度({len(req.baseline_damping_ratios)})不匹配"
            )

        mode_shapes = None
        if hasattr(req, "mode_shapes") and req.mode_shapes is not None:
            mode_shapes = np.array(req.mode_shapes, dtype=np.float64)

        result = assessor.assess_bond_strength(
            surface_id=req.surface_id,
            baseline_damping_ratios=req.baseline_damping_ratios,
            current_damping_ratios=req.current_damping_ratios,
            frequencies=req.frequencies,
            mode_shapes=mode_shapes,
            timestamp=req.timestamp,
            ambient_temperature_c=req.ambient_temperature_c,
        )

        return BondStrengthAssessResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"粘结强度评估失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/temperature_calibrate", response_model=TemperatureCalibrateResponse)
async def temperature_calibrate_endpoint(req: TemperatureCalibrateRequest):
    try:
        if len(req.damping_ratios) == 0:
            raise HTTPException(
                status_code=400,
                detail="阻尼比列表不能为空"
            )

        temp_assessor = BondStrengthAssessor(
            reference_temperature_c=req.reference_temperature_c,
        )

        correction_factor = temp_assessor._temperature_correction_factor(req.temperature_c)

        corrected_damping_ratios = [
            temp_assessor._correct_damping_for_temperature(float(dr), req.temperature_c)
            for dr in req.damping_ratios
        ]

        return TemperatureCalibrateResponse(
            corrected_damping_ratios=[round(float(x), 6) for x in corrected_damping_ratios],
            correction_factor=round(correction_factor, 6),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"温度修正计算失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
