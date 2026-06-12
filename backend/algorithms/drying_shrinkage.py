import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta


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


@dataclass
class GroutFormulation:
    formulation_id: str
    name: str
    water_binder_ratio: float
    ps_concentration_pct: float
    fly_ash_pct: float
    shrinkage_coefficient: float
    tensile_strength_mpa: float
    elastic_modulus_mpa: float
    poissons_ratio: float
    cracking_tensile_strain: float
    drying_rate_factor: float
    final_shrinkage_strain: float

    @property
    def id(self):
        return self.formulation_id


class AHTDryingShrinkageModel:
    def __init__(
        self,
        formulation_id: str = "sintered_stone_powder_ps_basic",
        reference_temperature_c: float = 20.0,
        reference_humidity_pct: float = 60.0,
        initial_curing_days: float = 3.0,
        prediction_horizon_days: float = 180.0,
    ):
        if formulation_id not in GROUT_FORMULATIONS:
            raise ValueError(f"未知配方ID: {formulation_id}")
        
        self.formulation = GroutFormulation(
            formulation_id=formulation_id,
            **GROUT_FORMULATIONS[formulation_id],
        )
        
        self.T_ref = reference_temperature_c + 273.15
        self.RH_ref = reference_humidity_pct / 100.0
        self.initial_curing_days = initial_curing_days
        self.prediction_horizon = prediction_horizon_days

        self.AHT_beta = 0.15
        self.AHT_gamma = 0.6
        self.humidity_sensitivity = 0.8
        self.temperature_activation_energy = 35000.0
        self.R_gas_constant = 8.314

    def set_formulation(self, formulation: GroutFormulation):
        if not isinstance(formulation, GroutFormulation):
            raise ValueError("formulation 必须是 GroutFormulation 类型")
        self.formulation = formulation

    def _arrhenius_temperature_factor(
        self,
        temperature_c: float,
    ) -> float:
        T = temperature_c + 273.15
        factor = np.exp(
            self.temperature_activation_energy / self.R_gas_constant
            * (1.0 / self.T_ref - 1.0 / T)
        )
        return float(factor)

    def _humidity_diffusion_factor(
        self,
        humidity_pct: float,
    ) -> float:
        RH = humidity_pct / 100.0
        if RH >= self.RH_ref:
            factor = 1.0 - self.humidity_sensitivity * (RH - self.RH_ref)
        else:
            factor = 1.0 + 2.0 * self.humidity_sensitivity * (self.RH_ref - RH)
        return float(max(factor, 0.1))

    def _age_hardening_function(
        self,
        t_days: float,
        temperature_c: float,
        humidity_pct: float,
    ) -> float:
        t_eq = max(t_days - self.initial_curing_days, 0.0)
        T_factor = self._arrhenius_temperature_factor(temperature_c)
        RH_factor = self._humidity_diffusion_factor(humidity_pct)
        
        t_eff = t_eq * T_factor * RH_factor * self.formulation.drying_rate_factor
        
        if t_eff <= 0:
            return 0.0
        
        alpha = 1.0 - np.exp(-self.AHT_beta * (t_eff ** self.AHT_gamma))
        return float(alpha)

    def _shrinkage_strain(
        self,
        alpha: float,
        constraint_factor: float = 0.7,
    ) -> float:
        epsilon_shrink = self.formulation.final_shrinkage_strain * alpha * constraint_factor
        return float(epsilon_shrink)

    def _tensile_stress(
        self,
        epsilon_shrink: float,
        alpha: float,
        constraint_factor: float = 0.7,
    ) -> float:
        E_t = self.formulation.elastic_modulus_mpa * (0.3 + 0.7 * alpha)
        nu = self.formulation.poissons_ratio
        biaxial_factor = 1.0 / (1.0 - nu)
        
        sigma_tensile = E_t * epsilon_shrink * constraint_factor * biaxial_factor
        return float(max(sigma_tensile, 0.0))

    def _crack_risk_index(
        self,
        sigma_tensile: float,
        alpha: float,
    ) -> float:
        f_t = self.formulation.tensile_strength_mpa * (0.2 + 0.8 * alpha)
        stress_strength_ratio = sigma_tensile / max(f_t, 0.01)
        raw_risk = stress_strength_ratio * 0.7
        return float(min(raw_risk, 0.99))

    def _crack_width_estimation(
        self,
        crack_risk_index: float,
        epsilon_shrink: float,
        panel_dimension_m: float = 2.0,
        rebar_spacing_m: float = 0.5,
    ) -> float:
        if crack_risk_index < 0.5:
            return 0.0
        
        crack_spacing = 0.66 * rebar_spacing_m * (1.0 + 0.15 / max(crack_risk_index, 0.1))
        crack_width = 2.0 * epsilon_shrink * crack_spacing * (crack_risk_index ** 0.5)
        
        return float(max(crack_width * 1000.0, 0.0))

    def predict(
        self,
        ambient_temperature_c: float,
        ambient_humidity_pct: float,
        constraint_factor: float = 0.7,
        wall_thickness_mm: float = 50.0,
    ) -> Dict:
        n_points = 100
        time_days = np.linspace(0, self.prediction_horizon, n_points)
        
        alpha_curve = np.array([
            self._age_hardening_function(t, ambient_temperature_c, ambient_humidity_pct)
            for t in time_days
        ])
        
        epsilon_curve = np.array([
            self._shrinkage_strain(a, constraint_factor) for a in alpha_curve
        ])
        
        stress_curve = np.array([
            self._tensile_stress(e, a, constraint_factor)
            for e, a in zip(epsilon_curve, alpha_curve)
        ])
        
        risk_curve = np.array([
            self._crack_risk_index(s, a)
            for s, a in zip(stress_curve, alpha_curve)
        ])
        
        final_alpha = alpha_curve[-1]
        final_shrinkage = epsilon_curve[-1]
        final_stress = stress_curve[-1]
        final_risk = risk_curve[-1]
        
        thickness_factor = min(wall_thickness_mm / 50.0, 2.0)
        adjusted_shrinkage = final_shrinkage * (1.0 + 0.3 * (thickness_factor - 1.0))
        adjusted_risk = final_risk * (1.0 + 0.2 * (thickness_factor - 1.0))
        
        crack_width = self._crack_width_estimation(adjusted_risk, adjusted_shrinkage)
        
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
        
        recommendations = self._generate_recommendations(
            risk_level=risk_level,
            temperature_c=ambient_temperature_c,
            humidity_pct=ambient_humidity_pct,
            formulation=self.formulation,
        )
        
        return {
            "formulation_id": self.formulation.formulation_id,
            "formulation_name": self.formulation.name,
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
            "tensile_strength_mpa": self.formulation.tensile_strength_mpa,
            "elastic_modulus_mpa": self.formulation.elastic_modulus_mpa,
            "time_curve_days": [round(float(t), 2) for t in time_days],
            "time_curve_strain": [round(float(e), 8) for e in epsilon_curve],
            "shrinkage_time_curve": {
                "days": [round(float(t), 2) for t in time_days],
                "shrinkage_strain": [round(float(e), 8) for e in epsilon_curve],
                "tensile_stress_mpa": [round(float(s), 6) for s in stress_curve],
                "crack_risk_index": [round(float(r), 4) for r in risk_curve],
                "hardening_degree": [round(float(a), 4) for a in alpha_curve],
            },
            "recommendations": recommendations,
        }

    def _generate_recommendations(
        self,
        risk_level: str,
        temperature_c: float,
        humidity_pct: float,
        formulation: GroutFormulation,
    ) -> List[str]:
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
        
        if formulation.water_binder_ratio > 0.45:
            recs.append(f"当前配方水胶比({formulation.water_binder_ratio:.2f})偏高，建议降低至0.40-0.42以减少收缩")
        
        if formulation.fly_ash_pct == 0:
            recs.append("建议掺加15-30%粉煤灰，利用火山灰效应改善浆体微观结构，降低干燥收缩")
        
        return recs


def get_formulation_by_id(formulation_id: str) -> Optional[GroutFormulation]:
    if formulation_id not in GROUT_FORMULATIONS:
        return None
    return GroutFormulation(
        formulation_id=formulation_id,
        **GROUT_FORMULATIONS[formulation_id],
    )


def compare_formulations(
    ambient_temperature_c: float,
    ambient_humidity_pct: float,
    constraint_factor: float = 0.7,
) -> List[Dict]:
    results = []
    for form_id in GROUT_FORMULATIONS.keys():
        model = AHTDryingShrinkageModel(
            formulation_id=form_id,
            prediction_horizon_days=90.0,
        )
        pred = model.predict(
            ambient_temperature_c=ambient_temperature_c,
            ambient_humidity_pct=ambient_humidity_pct,
            constraint_factor=constraint_factor,
        )
        results.append({
            "formulation_id": form_id,
            "formulation_name": pred["formulation_name"],
            "shrinkage_strain": pred["final_shrinkage_strain"],
            "crack_risk_index": pred["crack_risk_index"],
            "crack_risk_level": pred["crack_risk_level"],
            "tensile_strength_mpa": pred["tensile_strength_mpa"],
        })
    
    results.sort(key=lambda x: x["crack_risk_index"])
    return results


def optimize_curing_schedule(
    ambient_temperature_c: float,
    ambient_humidity_pct: float,
    formulation_id: str = "sintered_stone_powder_ps_basic",
    target_risk_index: float = 0.5,
) -> Dict:
    base_model = AHTDryingShrinkageModel(
        formulation_id=formulation_id,
        prediction_horizon_days=28.0,
    )
    
    base_pred = base_model.predict(
        ambient_temperature_c=ambient_temperature_c,
        ambient_humidity_pct=ambient_humidity_pct,
    )
    
    if base_pred["crack_risk_index"] <= target_risk_index:
        return {
            "optimization_needed": False,
            "baseline_risk": base_pred["crack_risk_index"],
            "target_risk": target_risk_index,
            "recommendation": "当前环境条件下无需特殊养护优化",
            "optimal_schedule": None,
        }
    
    humidity_options = [60, 70, 80, 90, 95]
    curing_days_options = [3, 7, 14, 21, 28]
    
    best_schedule = None
    best_risk = float("inf")
    
    for hum in humidity_options:
        for days in curing_days_options:
            model = AHTDryingShrinkageModel(
                formulation_id=formulation_id,
                initial_curing_days=days,
                prediction_horizon_days=28.0,
            )
            
            pred = model.predict(
                ambient_temperature_c=ambient_temperature_c,
                ambient_humidity_pct=hum,
            )
            
            if pred["crack_risk_index"] <= target_risk_index and pred["crack_risk_index"] < best_risk:
                best_risk = pred["crack_risk_index"]
                best_schedule = {
                    "curing_humidity_pct": hum,
                    "curing_duration_days": days,
                    "achieved_risk_index": pred["crack_risk_index"],
                    "predicted_crack_width_mm": pred["predicted_crack_width_mm"],
                }
                break
        if best_schedule is not None:
            break
    
    if best_schedule is None:
        best_schedule = {
            "curing_humidity_pct": 95,
            "curing_duration_days": 28,
            "achieved_risk_index": best_risk if best_risk < float("inf") else base_pred["crack_risk_index"],
            "note": "即使在最优养护条件下，仍存在一定收缩风险，建议配合使用膨胀剂或调整配方",
        }
    
    return {
        "optimization_needed": True,
        "baseline_risk": base_pred["crack_risk_index"],
        "target_risk": target_risk_index,
        "recommendation": f"需要优化养护制度以控制收缩风险",
        "optimal_schedule": best_schedule,
    }
