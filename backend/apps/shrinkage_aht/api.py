import logging
import numpy as np
from datetime import datetime
from typing import Dict, List

from fastapi import APIRouter, HTTPException

from backend.apps.shrinkage_aht.schemas import (
    DryingShrinkagePredictionRequest,
    DryingShrinkagePredictionResponse,
    FormulationComparisonRequest,
    FormulationComparisonResponse,
    FormulationComparisonItem,
    ShrinkageTimeCurve,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/shrinkage_aht", tags=["干燥收缩预测"])

try:
    from .aht_core import (
        aht_final_shrinkage,
        humidity_diffusion_factor,
        arrhenius_temperature_factor,
        compute_alpha_curve,
    )
    USE_CYTHON = True
    logger.info("使用Cython加速的aht_core模块")
except ImportError:
    from .aht_python import (
        aht_final_shrinkage,
        humidity_diffusion_factor,
        arrhenius_temperature_factor,
        compute_alpha_curve,
    )
    USE_CYTHON = False
    logger.warning("Cython模块导入失败，使用纯Python版本aht_python")


GROUT_FORMULATIONS = {
    "sintered_stone_powder_ps_basic": {
        "name": "烧结石粉-PS基础配方",
        "water_binder_ratio": 0.45,
        "ps_concentration_pct": 5.0,
        "fly_ash_pct": 0.0,
        "shrinkage_coefficient": 3.5e-4,
        "tensile_strength_mpa": 0.8,
        "elastic_modulus_mpa": 1200.0,
        "poissons_ratio": 0.22,
        "cracking_tensile_strain": 800e-6,
        "drying_rate_factor": 1.0,
        "final_shrinkage_strain": 1200e-6,
    },
    "sintered_stone_powder_ps_modified": {
        "name": "烧结石粉-PS改性配方(含粉煤灰)",
        "water_binder_ratio": 0.42,
        "ps_concentration_pct": 7.0,
        "fly_ash_pct": 20.0,
        "shrinkage_coefficient": 2.8e-4,
        "tensile_strength_mpa": 1.0,
        "elastic_modulus_mpa": 1400.0,
        "poissons_ratio": 0.21,
        "cracking_tensile_strain": 950e-6,
        "drying_rate_factor": 0.85,
        "final_shrinkage_strain": 950e-6,
    },
    "high_modulus_low_shrinkage": {
        "name": "高模量低收缩配方",
        "water_binder_ratio": 0.38,
        "ps_concentration_pct": 8.0,
        "fly_ash_pct": 30.0,
        "shrinkage_coefficient": 2.2e-4,
        "tensile_strength_mpa": 1.2,
        "elastic_modulus_mpa": 1600.0,
        "poissons_ratio": 0.20,
        "cracking_tensile_strain": 1100e-6,
        "drying_rate_factor": 0.7,
        "final_shrinkage_strain": 800e-6,
    },
    "traditional_lime_mortar": {
        "name": "传统石灰砂浆对比配方",
        "water_binder_ratio": 0.55,
        "ps_concentration_pct": 0.0,
        "fly_ash_pct": 0.0,
        "shrinkage_coefficient": 5.0e-4,
        "tensile_strength_mpa": 0.3,
        "elastic_modulus_mpa": 500.0,
        "poissons_ratio": 0.25,
        "cracking_tensile_strain": 600e-6,
        "drying_rate_factor": 1.3,
        "final_shrinkage_strain": 1800e-6,
    },
}


AHT_BETA = 0.15
AHT_GAMMA = 0.6
HUMIDITY_SENSITIVITY = 0.8
TEMPERATURE_ACTIVATION_ENERGY = 35000.0
R_GAS_CONSTANT = 8.314
T_REF_K = 293.15
RH_REF = 0.6


def _shrinkage_strain(alpha: float, final_shrinkage_strain: float, constraint_factor: float) -> float:
    return final_shrinkage_strain * alpha * constraint_factor


def _tensile_stress(epsilon_shrink: float, alpha: float, elastic_modulus_mpa: float,
                    poissons_ratio: float, constraint_factor: float) -> float:
    E_t = elastic_modulus_mpa * (0.3 + 0.7 * alpha)
    nu = poissons_ratio
    biaxial_factor = 1.0 / (1.0 - nu)
    sigma_tensile = E_t * epsilon_shrink * constraint_factor * biaxial_factor
    return max(sigma_tensile, 0.0)


def _crack_risk_index(sigma_tensile: float, alpha: float, tensile_strength_mpa: float) -> float:
    f_t = tensile_strength_mpa * (0.2 + 0.8 * alpha)
    stress_strength_ratio = sigma_tensile / max(f_t, 0.01)
    return min(stress_strength_ratio, 1.0)


def _crack_width_estimation(crack_risk_index: float, epsilon_shrink: float,
                            panel_dimension_m: float = 2.0, rebar_spacing_m: float = 0.5) -> float:
    if crack_risk_index < 0.5:
        return 0.0
    crack_spacing = 0.66 * rebar_spacing_m * (1.0 + 0.15 / max(crack_risk_index, 0.1))
    crack_width = 2.0 * epsilon_shrink * crack_spacing * (crack_risk_index ** 0.5)
    return max(crack_width * 1000.0, 0.0)


def _generate_recommendations(risk_level: str, temperature_c: float, humidity_pct: float,
                              formulation: Dict) -> List[str]:
    recs = []
    
    if risk_level in ["高", "极高"]:
        recs.append("建议采用分级加荷养护制度，前7天保持90%以上湿度，避免快速干燥")
        recs.append(f"考虑改用低收缩配方: {GROUT_FORMULATIONS['high_modulus_low_shrinkage']['name']}")
        recs.append("增加PS材料掺量至7-8%，提高浆体极限拉伸值")
        recs.append("在灌浆后14天内避免游客参观，减少振动扰动")
        if temperature_c > 28:
            recs.append("当前环境温度偏高，建议在灌浆区域设置遮阳降温措施")
        if humidity_pct < 40:
            recs.append("当前环境湿度过低，建议安装加湿器保持环境湿度在60%以上")
    elif risk_level == "中":
        recs.append("加强前14天的湿养护，每日喷雾保湿不少于3次")
        recs.append("考虑在浆体中掺入15-20%粉煤灰，降低干燥收缩")
        recs.append("在灌浆后7天内限制该区域的游客流量")
        if temperature_c > 25:
            recs.append("环境温度略高，建议调整灌浆时间至早间或傍晚温度较低时段")
        if humidity_pct < 50:
            recs.append("环境湿度偏低，建议在灌浆后覆盖保湿膜养护")
    elif risk_level == "低":
        recs.append("按常规湿养护制度执行即可，前3天保持表面湿润")
        recs.append("持续监测环境温湿度，避免突发干热天气影响")
    else:
        recs.append("收缩风险极低，可按正常养护流程操作")
    
    if formulation["water_binder_ratio"] > 0.45:
        recs.append(f"当前配方水胶比({formulation['water_binder_ratio']:.2f})偏高，建议降低至0.40-0.42以减少收缩")
    
    if formulation["fly_ash_pct"] == 0:
        recs.append("建议掺加15-30%粉煤灰，利用火山灰效应改善浆体微观结构，降低干燥收缩")
    
    return recs


def predict_drying_shrinkage(
    formulation_id: str,
    ambient_temperature_c: float,
    ambient_humidity_pct: float,
    constraint_factor: float,
    wall_thickness_mm: float,
    prediction_horizon_days: float,
    initial_curing_days: float,
) -> Dict:
    if formulation_id not in GROUT_FORMULATIONS:
        raise ValueError(f"未知配方ID: {formulation_id}")
    
    formulation = GROUT_FORMULATIONS[formulation_id]
    
    n_points = 100
    time_days = np.linspace(0, prediction_horizon_days, n_points, dtype=np.float64)
    
    alpha_curve = compute_alpha_curve(
        time_days,
        ambient_temperature_c,
        ambient_humidity_pct,
        initial_curing_days,
        formulation["drying_rate_factor"],
    )
    
    epsilon_curve = np.array([
        _shrinkage_strain(a, formulation["final_shrinkage_strain"], constraint_factor)
        for a in alpha_curve
    ])
    
    stress_curve = np.array([
        _tensile_stress(e, a, formulation["elastic_modulus_mpa"], 
                        formulation["poissons_ratio"], constraint_factor)
        for e, a in zip(epsilon_curve, alpha_curve)
    ])
    
    risk_curve = np.array([
        _crack_risk_index(s, a, formulation["tensile_strength_mpa"])
        for s, a in zip(stress_curve, alpha_curve)
    ])
    
    final_alpha = float(alpha_curve[-1])
    final_shrinkage = float(epsilon_curve[-1])
    final_stress = float(stress_curve[-1])
    final_risk = float(risk_curve[-1])
    
    thickness_factor = min(wall_thickness_mm / 50.0, 2.0)
    adjusted_shrinkage = final_shrinkage * (1.0 + 0.3 * (thickness_factor - 1.0))
    adjusted_risk = final_risk * (1.0 + 0.2 * (thickness_factor - 1.0))
    
    crack_width = _crack_width_estimation(adjusted_risk, adjusted_shrinkage)
    
    if adjusted_risk >= 0.9:
        risk_level = "极高"
    elif adjusted_risk >= 0.7:
        risk_level = "高"
    elif adjusted_risk >= 0.4:
        risk_level = "中"
    elif adjusted_risk >= 0.2:
        risk_level = "低"
    else:
        risk_level = "无"
    
    critical_period_idx = np.argmax(risk_curve >= 0.5) if np.any(risk_curve >= 0.5) else n_points - 1
    critical_period_days = float(time_days[critical_period_idx])
    
    recommendations = _generate_recommendations(
        risk_level=risk_level,
        temperature_c=ambient_temperature_c,
        humidity_pct=ambient_humidity_pct,
        formulation=formulation,
    )
    
    return {
        "formulation_id": formulation_id,
        "formulation_name": formulation["name"],
        "ambient_temperature_c": ambient_temperature_c,
        "ambient_humidity_pct": ambient_humidity_pct,
        "constraint_factor": constraint_factor,
        "wall_thickness_mm": wall_thickness_mm,
        "final_shrinkage_strain": round(adjusted_shrinkage, 8),
        "final_tensile_stress_mpa": round(final_stress, 6),
        "crack_risk_index": round(adjusted_risk, 4),
        "predicted_crack_width_mm": round(crack_width, 4),
        "crack_risk_level": risk_level,
        "critical_period_days": round(critical_period_days, 2),
        "tensile_strength_mpa": formulation["tensile_strength_mpa"],
        "elastic_modulus_mpa": formulation["elastic_modulus_mpa"],
        "shrinkage_time_curve": ShrinkageTimeCurve(
            days=[round(float(t), 2) for t in time_days],
            shrinkage_strain=[round(float(e), 8) for e in epsilon_curve],
            tensile_stress_mpa=[round(float(s), 6) for s in stress_curve],
            crack_risk_index=[round(float(r), 4) for r in risk_curve],
            hardening_degree=[round(float(a), 4) for a in alpha_curve],
        ),
        "recommendations": recommendations,
    }


@router.post("/predict", response_model=DryingShrinkagePredictionResponse)
async def predict_endpoint(req: DryingShrinkagePredictionRequest):
    try:
        result = predict_drying_shrinkage(
            formulation_id=req.formulation_id,
            ambient_temperature_c=req.ambient_temperature_c,
            ambient_humidity_pct=req.ambient_humidity_pct,
            constraint_factor=req.constraint_factor,
            wall_thickness_mm=req.wall_thickness_mm,
            prediction_horizon_days=req.prediction_horizon_days,
            initial_curing_days=req.initial_curing_days,
        )
        return DryingShrinkagePredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"干燥收缩预测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare_formulations", response_model=FormulationComparisonResponse)
async def compare_formulations_endpoint(req: FormulationComparisonRequest):
    try:
        formulation_ids = list(GROUT_FORMULATIONS.keys())
        if len(formulation_ids) != 4:
            raise HTTPException(status_code=500, detail="配方数量不正确，需要恰好4种配方")
        
        results = []
        for form_id in formulation_ids:
            pred = predict_drying_shrinkage(
                formulation_id=form_id,
                ambient_temperature_c=req.ambient_temperature_c,
                ambient_humidity_pct=req.ambient_humidity_pct,
                constraint_factor=req.constraint_factor,
                wall_thickness_mm=50.0,
                prediction_horizon_days=90.0,
                initial_curing_days=3.0,
            )
            results.append(FormulationComparisonItem(
                formulation_id=form_id,
                formulation_name=pred["formulation_name"],
                shrinkage_strain=pred["final_shrinkage_strain"],
                crack_risk_index=pred["crack_risk_index"],
                crack_risk_level=pred["crack_risk_level"],
                tensile_strength_mpa=pred["tensile_strength_mpa"],
            ))
        
        results.sort(key=lambda x: x.crack_risk_index)
        
        return FormulationComparisonResponse(
            ambient_temperature_c=req.ambient_temperature_c,
            ambient_humidity_pct=req.ambient_humidity_pct,
            formulations=results,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"配方比较失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
