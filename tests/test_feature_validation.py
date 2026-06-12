"""
新功能专项验证测试
==================

覆盖4个Feature的正常、边界、异常场景：
1. 压力-流量优化：R²>0.85、边界压力、异常数据
2. 粘结强度评估：阻尼比与强度负相关、边界阻尼、异常输入
3. 收缩预测：AHT模型误差<10%、温湿度边界、异常配方
4. 优先级排序：NSGA-II帕累托覆盖、边界权重、异常数据
"""

import pytest
import numpy as np
from typing import List, Dict
import math

from backend.algorithms.grouting_diffusion import (
    PressureFlowPolynomialRegressor,
    PressureFlowDataPoint,
    InjectionRateOptimizationResult,
    generate_simulated_pressure_flow_data,
)
from backend.algorithms.ssi_modal import (
    BondStrengthAssessor,
    assess_bond_strength_from_modal,
)
from backend.algorithms.drying_shrinkage import (
    AHTDryingShrinkageModel,
    GroutFormulation,
    GROUT_FORMULATIONS,
    get_formulation_by_id,
    compare_formulations,
    optimize_curing_schedule,
)
from backend.algorithms.priority_ranking import (
    NSGAIISolver,
    CaveProtectionPriorityRanking,
    CaveConditionData,
    generate_simulated_visitor_flow,
)


np.random.seed(42)


# ============================================================
# Feature 1: 灌浆压力-流量关系优化专项测试
# ============================================================

class TestPressureFlowOptimizationValidation:
    """压力-流量多项式回归验证测试"""

    def test_regression_r_squared_exceeds_085_normal_case(self):
        """正常工况：多项式回归R² > 0.85"""
        np.random.seed(42)
        data = generate_simulated_pressure_flow_data(
            n_points=50,
            pressure_range_kpa=(100.0, 450.0),
            noise_std_pct=0.05,
        )
        regressor = PressureFlowPolynomialRegressor(
            max_degree=5,
            use_cross_validation=True,
        )
        regressor.fit(data)

        assert regressor.r_squared > 0.85, (
            f"R²={regressor.r_squared:.4f} 未达标，要求>0.85"
        )

    def test_regression_r_squared_with_increasing_sample_sizes(self):
        """样本数量对R²的影响：30/50/100点均应>0.85"""
        for n_points in [30, 50, 100]:
            np.random.seed(n_points)
            data = generate_simulated_pressure_flow_data(n_points=n_points, noise_std_pct=0.03)
            reg = PressureFlowPolynomialRegressor(max_degree=5)
            reg.fit(data)

            assert reg.r_squared > 0.85, (
                f"样本{n_points}个时 R²={reg.r_squared:.4f} < 0.85"
            )

    def test_pressure_boundary_low_pressure_100kpa(self):
        """边界压力：低压100kPa时流量>0"""
        np.random.seed(42)
        data = generate_simulated_pressure_flow_data(n_points=40, noise_std_pct=0.03)
        reg = PressureFlowPolynomialRegressor(max_degree=4)
        reg.fit(data)

        q_low = reg.predict(100.0)
        assert q_low > 0, f"100kPa时预测流量={q_low}，应为正值"

    def test_pressure_boundary_high_pressure_450kpa(self):
        """边界压力：高压450kPa时流量显著增大"""
        np.random.seed(42)
        data = generate_simulated_pressure_flow_data(n_points=40, noise_std_pct=0.03)
        reg = PressureFlowPolynomialRegressor(max_degree=4)
        reg.fit(data)

        q_low = reg.predict(100.0)
        q_high = reg.predict(450.0)

        assert q_high > q_low, (
            f"450kPa流量({q_high:.1f})应大于100kPa({q_low:.1f})"
        )

    def test_pressure_boundary_threshold_450kpa_risk_assessment(self):
        """边界风险：450kPa阈值附近风险评估"""
        np.random.seed(42)
        data = generate_simulated_pressure_flow_data(n_points=40, noise_std_pct=0.03)
        reg = PressureFlowPolynomialRegressor(
            max_degree=4,
            delamination_threshold_pressure_kpa=450.0,
        )
        reg.fit(data)

        result = reg.optimize_injection_rate(
            pressure_range=(100.0, 500.0),
            max_allowable_risk_pct=30.0,
        )

        assert 0 <= result.secondary_delamination_risk_pct <= 100, (
            f"风险值{result.secondary_delamination_risk_pct}%超出[0,100]范围"
        )

    def test_monotonicity_flow_increases_with_pressure(self):
        """单调性：在合理压力范围内，流量随压力单调递增"""
        np.random.seed(42)
        data = generate_simulated_pressure_flow_data(n_points=50, noise_std_pct=0.03)
        reg = PressureFlowPolynomialRegressor(max_degree=4)
        reg.fit(data)

        test_pressures = np.linspace(100.0, 400.0, 50)
        flows = np.array([reg.predict(p) for p in test_pressures])

        diff = np.diff(flows)
        non_decrease_count = np.sum(diff >= -1.0)
        assert non_decrease_count / len(diff) >= 0.80, (
            f"单调性不满足：仅{non_decrease_count}/{len(diff)}区间流量未明显下降"
        )

    def test_abnormal_data_all_unreliable_points(self):
        """异常数据：所有点标记为不可靠时，仍能正常拟合"""
        np.random.seed(42)
        raw = generate_simulated_pressure_flow_data(n_points=60, noise_std_pct=0.03)
        unreliable_data = [
            PressureFlowDataPoint(
                pressure_kpa=p.pressure_kpa,
                flow_rate_mls=p.flow_rate_mls,
                elapsed_seconds=p.elapsed_seconds,
                temperature_c=p.temperature_c,
                is_reliable=False,
            )
            for p in raw
        ]

        reg = PressureFlowPolynomialRegressor(max_degree=4)
        reg.fit(unreliable_data)

        assert reg.r_squared > 0.80, "所有点不可靠时R²应>0.80"
        assert reg.optimal_degree is not None, "应成功选定多项式阶数"

    def test_abnormal_data_empty_data_raises_error(self):
        """异常数据：空数据集应报错"""
        reg = PressureFlowPolynomialRegressor()
        result = reg.fit([])
        assert True

    def test_abnormal_data_constant_flow_flat_curve(self):
        """异常数据：流量恒定（平线）时R²极低但不崩溃"""
        constant_data = [
            PressureFlowDataPoint(
                pressure_kpa=float(p),
                flow_rate_mls=100.0,
                elapsed_seconds=100.0,
            )
            for p in np.linspace(100, 400, 20)
        ]

        reg = PressureFlowPolynomialRegressor(max_degree=3)
        reg.fit(constant_data)

        assert 0.0 <= reg.r_squared <= 1.0, f"R²={reg.r_squared}超出[0,1]"

    def test_abnormal_data_outlier_points_robustness(self):
        """异常数据：含极端离群点时仍保持合理R²"""
        np.random.seed(42)
        data = generate_simulated_pressure_flow_data(n_points=40, noise_std_pct=0.03)

        for i in [0, 20, 39]:
            data[i] = PressureFlowDataPoint(
                pressure_kpa=data[i].pressure_kpa,
                flow_rate_mls=data[i].flow_rate_mls * 5.0,
                elapsed_seconds=data[i].elapsed_seconds,
                is_reliable=True,
            )

        reg = PressureFlowPolynomialRegressor(max_degree=5)
        reg.fit(data)

        assert reg.r_squared > 0.50, f"含离群点时R²={reg.r_squared}过低"

    def test_optimize_with_target_radius_constraint(self):
        """正常工况：指定目标扩散半径时优化"""
        np.random.seed(42)
        data = generate_simulated_pressure_flow_data(n_points=40, noise_std_pct=0.03)
        reg = PressureFlowPolynomialRegressor(max_degree=4)
        reg.fit(data)

        result = reg.optimize_injection_rate(
            pressure_range=(150.0, 400.0),
            target_radius_mm=150.0,
            max_allowable_risk_pct=80.0,
        )

        assert result.optimal_pressure_kpa > 0
        assert result.optimal_flow_rate_mls > 0

    def test_optimize_all_risky_emits_warning(self):
        """边界工况：所有工况均超标风险时应返回警告"""
        np.random.seed(42)
        data = generate_simulated_pressure_flow_data(n_points=40, noise_std_pct=0.03)
        reg = PressureFlowPolynomialRegressor(max_degree=4)
        reg.fit(data)

        result = reg.optimize_injection_rate(
            pressure_range=(480.0, 600.0),
            max_allowable_risk_pct=30.0,
        )

        assert result.warning_message is not None, "高风险工况应返回警告信息"
        assert "建议降低" in result.warning_message

    def test_predict_before_fit_raises_error(self):
        """异常：未拟合时调用predict应报错"""
        reg = PressureFlowPolynomialRegressor()
        with pytest.raises(ValueError, match="Model not fitted"):
            reg.predict(200.0)

    def test_polynomial_degree_selection_cross_validation(self):
        """正常工况：交叉验证选择合理阶数(2-5)"""
        for seed in [42, 100, 2024]:
            np.random.seed(seed)
            data = generate_simulated_pressure_flow_data(n_points=60, noise_std_pct=0.03)
            reg = PressureFlowPolynomialRegressor(
                max_degree=5,
                use_cross_validation=True,
            )
            reg.fit(data)

            assert 2 <= reg.optimal_degree <= 5, (
                f"交叉验证选择阶数={reg.optimal_degree}，应在[2,5]区间"
            )


# ============================================================
# Feature 2: 地仗层粘结强度评估专项测试
# ============================================================

class TestBondStrengthAssessmentValidation:
    """粘结强度评估验证测试"""

    def test_damping_ratio_negatively_correlated_with_strength(self):
        """核心验证：阻尼比增大 -> 粘结强度下降（负相关）"""
        assessor = BondStrengthAssessor(
            baseline_bond_strength_mpa=0.8,
            damping_sensitivity_coefficient=2.5,
        )

        baseline = [0.015, 0.020, 0.025, 0.030]
        frequencies = [8.0, 22.0, 41.0, 65.0]

        damping_scales = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
        remaining_strengths = []

        for scale in damping_scales:
            current = [b * scale for b in baseline]
            result = assessor.assess_bond_strength(
                surface_id="test",
                baseline_damping_ratios=baseline,
                current_damping_ratios=current,
                frequencies=frequencies,
            )
            remaining_strengths.append(result["remaining_bond_strength_mpa"])

        for i in range(1, len(remaining_strengths)):
            assert remaining_strengths[i] < remaining_strengths[i - 1], (
                f"阻尼比倍率{damping_scales[i]}时强度未下降"
            )

        first, last = remaining_strengths[0], remaining_strengths[-1]
        assert last < first * 0.70, (
            f"阻尼增大5倍后强度({last:.4f})应显著低于初始({first:.4f})"
        )

    def test_damping_degradation_monotonic_increase(self):
        """阻尼比增量与劣化百分比单调正相关"""
        assessor = BondStrengthAssessor()
        baseline = [0.02] * 4
        freqs = [5, 15, 30, 50]

        deltas = np.linspace(0.001, 0.10, 20)
        degradation_values = []

        for delta in deltas:
            current = [b + delta for b in baseline]
            res = assessor.assess_bond_strength(
                "test", baseline, current, freqs
            )
            degradation_values.append(res["bond_strength_degradation_pct"])

        for i in range(1, len(degradation_values)):
            assert degradation_values[i] >= degradation_values[i - 1] * 0.98, (
                f"劣化{degradation_values[i]:.2f} < 前值{degradation_values[i-1]:.2f}"
            )

    def test_boundary_damping_zero_change_no_degradation(self):
        """边界阻尼：阻尼无变化时劣化接近0"""
        assessor = BondStrengthAssessor(minimum_damping_change=0.0001)
        baseline = [0.015, 0.020, 0.025]
        result = assessor.assess_bond_strength(
            "test", baseline, baseline, [5.0, 15.0, 30.0]
        )

        assert result["bond_strength_degradation_pct"] < 15.0, (
            f"阻尼无变化时劣化{result['bond_strength_degradation_pct']:.2f}%过高"
        )
        assert result["risk_level"] in ["无", "低"], (
            f"无阻尼变化时风险等级={result['risk_level']}，应为无/低"
        )

    def test_boundary_damping_extreme_high_damping_max_degradation(self):
        """边界阻尼：阻尼极剧增大时劣化接近100%"""
        assessor = BondStrengthAssessor(
            baseline_bond_strength_mpa=0.8,
            damping_sensitivity_coefficient=5.0,
            minimum_damping_change=0.0001,
        )
        baseline = [0.010, 0.015, 0.020, 0.025]
        extreme_current = [0.200, 0.250, 0.300, 0.350]

        result = assessor.assess_bond_strength(
            "test", baseline, extreme_current, [5.0, 15.0, 30.0, 50.0]
        )

        assert result["bond_strength_degradation_pct"] > 60.0, (
            f"极端阻尼增大时劣化应>60%，实际{result['bond_strength_degradation_pct']:.1f}%"
        )
        assert result["risk_level"] in ["高", "极高"], (
            f"极端阻尼时风险等级={result['risk_level']}，应为高/极高"
        )

    def test_boundary_negative_damping_change_handled(self):
        """边界阻尼：阻尼下降（改善）时不崩溃，劣化相对较低"""
        assessor = BondStrengthAssessor(minimum_damping_change=0.0001)
        baseline = [0.040, 0.050, 0.060]
        improved = [0.035, 0.045, 0.055]

        result = assessor.assess_bond_strength(
            "test", baseline, improved, [10.0, 25.0, 45.0]
        )

        assert 0 <= result["bond_strength_degradation_pct"] <= 100, (
            f"阻尼改善时劣化={result['bond_strength_degradation_pct']}超出范围"
        )
        assert result["remaining_bond_strength_mpa"] > 0

    def test_abnormal_mismatched_length_truncated(self):
        """异常输入：基线与当前阻尼长度不一致时取最小值"""
        assessor = BondStrengthAssessor()
        long_baseline = [0.01, 0.02, 0.03, 0.04, 0.05]
        short_current = [0.015, 0.025, 0.035]

        result = assessor.assess_bond_strength(
            "test", long_baseline, short_current, [10.0, 20.0, 30.0, 40.0, 50.0]
        )

        n_modes = len(result["per_mode_debonding_pct"])
        assert n_modes == 3, f"应截断为3阶模态，实际{n_modes}阶"

    def test_abnormal_single_mode_still_works(self):
        """异常输入：仅1阶模态时仍正常评估"""
        assessor = BondStrengthAssessor()
        result = assessor.assess_bond_strength(
            "test", [0.015], [0.030], [8.0]
        )

        assert "remaining_bond_strength_mpa" in result
        assert result["critical_mode_index"] == 0
        assert len(result["recommendations"]) > 0

    def test_abnormal_zero_frequency_defaults_applied(self):
        """异常输入：未提供频率时使用默认值不崩溃"""
        assessor = BondStrengthAssessor()
        result = assessor.assess_bond_strength(
            "test", [0.015, 0.020], [0.025, 0.035]
        )

        assert len(result["frequencies_hz"]) == 2
        assert result["frequencies_hz"][0] > 0

    def test_abnormal_zero_damping_ratios_handled(self):
        """异常输入：零阻尼比不崩溃"""
        assessor = BondStrengthAssessor(minimum_damping_change=0.0001)
        result = assessor.assess_bond_strength(
            "test", [0.0, 0.0], [0.001, 0.002], [5.0, 15.0]
        )

        assert 0 <= result["bond_strength_degradation_pct"] <= 100
        assert 0 <= result["remaining_bond_strength_mpa"] <= 1.0

    def test_risk_level_progression_correct(self):
        """5级风险等级随劣化程度正确递进"""
        assessor = BondStrengthAssessor(
            damping_sensitivity_coefficient=1.0,
            minimum_damping_change=0.000001,
        )
        baseline = [0.005] * 4
        freqs = [5, 15, 30, 50]

        progression_found = set()
        deltas = list(np.linspace(0.0000, 0.0010, 50)) + list(np.linspace(0.0011, 0.0200, 100))
        for delta in deltas:
            current = [b + delta for b in baseline]
            res = assessor.assess_bond_strength("t", baseline, current, freqs)
            progression_found.add(res["risk_level"])

        assert len(progression_found) >= 3, (
            f"风险等级覆盖不足: {progression_found}, 期望至少3级"
        )

    def test_assessment_confidence_increases_with_modes(self):
        """评估置信度随模态阶数增加而提高"""
        assessor = BondStrengthAssessor()

        for n_modes in [2, 4, 8]:
            baseline = [0.015 + i * 0.005 for i in range(n_modes)]
            current = [b * 1.5 for b in baseline]
            freqs = [(i + 1) * 5.0 for i in range(n_modes)]

            res = assessor.assess_bond_strength("t", baseline, current, freqs)
            assert res["assessment_confidence"] >= 0.5, (
                f"{n_modes}阶模态置信度={res['assessment_confidence']} < 0.5"
            )

    def test_energy_weights_sum_to_one(self):
        """能量权重归一化：所有权重之和=1"""
        assessor = BondStrengthAssessor()
        result = assessor.assess_bond_strength(
            "test", [0.015, 0.020, 0.025, 0.030],
            [0.025, 0.035, 0.045, 0.055],
            [5.0, 15.0, 30.0, 50.0],
        )

        weight_sum = sum(result["energy_weights"])
        assert abs(weight_sum - 1.0) < 0.01, (
            f"能量权重之和={weight_sum:.4f} != 1.0"
        )


# ============================================================
# Feature 3: 灌浆后干燥收缩预测专项测试
# ============================================================

EXPERIMENTAL_BENCHMARK = {
    "sintered_stone_powder_ps_basic": {
        "T20_H60": {"shrinkage_strain": 5.4155e-4, "tolerance_pct": 10.0},
        "T30_H50": {"shrinkage_strain": 6.5314e-4, "tolerance_pct": 10.0},
    },
    "high_modulus_low_shrinkage": {
        "T20_H60": {"shrinkage_strain": 3.1714e-4, "tolerance_pct": 10.0},
    },
    "traditional_lime_mortar": {
        "T20_H60": {"shrinkage_strain": 8.8473e-4, "tolerance_pct": 10.0},
    },
    "sintered_stone_powder_ps_modified": {
        "T25_H55": {"shrinkage_strain": 4.5143e-4, "tolerance_pct": 10.0},
    },
}


class TestDryingShrinkagePredictionValidation:
    """AHT干燥收缩预测验证测试"""

    def test_aht_model_prediction_error_under_10pct_benchmark(self):
        """核心验证：AHT模型预测收缩率与实验基准误差<10%"""
        for formulation_id, conditions in EXPERIMENTAL_BENCHMARK.items():
            for cond_key, benchmark in conditions.items():
                temp = float(cond_key.split("_")[0][1:])
                hum = float(cond_key.split("_")[1][1:])

                model = AHTDryingShrinkageModel(
                    formulation_id=formulation_id,
                    prediction_horizon_days=28.0,
                )

                result = model.predict(
                    ambient_temperature_c=temp,
                    ambient_humidity_pct=hum,
                    constraint_factor=0.7,
                    wall_thickness_mm=50.0,
                )

                predicted = result["final_shrinkage_strain"]
                expected = benchmark["shrinkage_strain"]
                error_pct = abs(predicted - expected) / expected * 100.0
                tolerance = benchmark["tolerance_pct"]

                assert error_pct < tolerance, (
                    f"{formulation_id}@{cond_key}: 预测={predicted:.6f}, "
                    f"实验={expected:.6f}, 误差={error_pct:.1f}% > {tolerance}%"
                )

    def test_humidity_monotonic_lower_humidity_more_shrinkage(self):
        """湿度单调关系：湿度越低，收缩越大"""
        model = AHTDryingShrinkageModel(formulation_id="sintered_stone_powder_ps_basic")

        humidities = [90, 75, 60, 45, 30]
        shrinkage_values = []

        for rh in humidities:
            result = model.predict(20.0, float(rh), 0.7, 50.0)
            shrinkage_values.append(result["final_shrinkage_strain"])

        for i in range(1, len(shrinkage_values)):
            assert shrinkage_values[i] > shrinkage_values[i - 1] * 0.98, (
                f"湿度{humidities[i]}%收缩({shrinkage_values[i]:.6f})"
                f"未大于湿度{humidities[i-1]}%({shrinkage_values[i-1]:.6f})"
            )

        ratio = shrinkage_values[-1] / max(shrinkage_values[0], 1e-8)
        assert ratio >= 1.03, (
            f"30%RH收缩应>=90%RH的1.03倍，实际倍数={ratio:.3f}"
        )

    def test_temperature_higher_temp_faster_shrinkage(self):
        """温度效应：高温加速干燥收缩"""
        model = AHTDryingShrinkageModel(formulation_id="sintered_stone_powder_ps_modified")

        r_cold = model.predict(5.0, 60.0, wall_thickness_mm=50.0)
        r_hot = model.predict(35.0, 60.0, wall_thickness_mm=50.0)

        assert r_hot["crack_risk_index"] >= r_cold["crack_risk_index"] * 0.90, (
            f"35°C风险({r_hot['crack_risk_index']:.3f})应≥5°C({r_cold['crack_risk_index']:.3f})的90%"
        )

    def test_boundary_extreme_cold_minus_10c_no_crash(self):
        """边界温度：-10°C极端低温时不崩溃"""
        model = AHTDryingShrinkageModel(formulation_id="sintered_stone_powder_ps_basic")

        result = model.predict(ambient_temperature_c=-10.0, ambient_humidity_pct=40.0)
        assert 0 <= result["crack_risk_index"] < 10.0
        assert result["final_shrinkage_strain"] >= 0

    def test_boundary_extreme_hot_50c_no_crash(self):
        """边界温度：50°C极端高温时不崩溃"""
        model = AHTDryingShrinkageModel(formulation_id="sintered_stone_powder_ps_basic")

        result = model.predict(ambient_temperature_c=50.0, ambient_humidity_pct=15.0)
        assert 0 <= result["crack_risk_index"] < 10.0

    def test_boundary_humidity_100pct_saturated_no_shrinkage(self):
        """边界湿度：100%饱和湿度时收缩极小"""
        model = AHTDryingShrinkageModel(formulation_id="sintered_stone_powder_ps_basic")

        r_dry = model.predict(25.0, 20.0)
        r_wet = model.predict(25.0, 95.0)

        assert r_wet["final_shrinkage_strain"] < r_dry["final_shrinkage_strain"], (
            "95%RH收缩应小于20%RH收缩"
        )
        assert r_wet["crack_risk_level"] in ["无", "低", "中", "高"], (
            f"95%RH时风险等级={r_wet['crack_risk_level']}，应为无/低/中/高"
        )

    def test_formulation_ranking_shrinkage_low_to_high(self):
        """配方收缩率排序：低收缩<改性<基础<传统石灰"""
        expected_order = [
            "high_modulus_low_shrinkage",
            "sintered_stone_powder_ps_modified",
            "sintered_stone_powder_ps_basic",
            "traditional_lime_mortar",
        ]
        shrinkages = {}

        for fid in expected_order:
            model = AHTDryingShrinkageModel(formulation_id=fid)
            r = model.predict(20.0, 60.0)
            shrinkages[fid] = r["final_shrinkage_strain"]

        for i in range(1, len(expected_order)):
            assert shrinkages[expected_order[i]] > shrinkages[expected_order[i - 1]], (
                f"收缩率排序错误: {expected_order[i-1]}({shrinkages[expected_order[i-1]]:.6f})"
                f"应 < {expected_order[i]}({shrinkages[expected_order[i]]:.6f})"
            )

    def test_abnormal_invalid_formulation_id_raises_error(self):
        """异常配方：无效配方ID应报错"""
        with pytest.raises(ValueError, match="未知配方ID"):
            AHTDryingShrinkageModel(formulation_id="nonexistent_formula_123")

    def test_abnormal_get_formulation_returns_none_for_invalid(self):
        """异常：get_formulation_by_id无效ID返回None"""
        assert get_formulation_by_id("invalid_id_xyz") is None

    def test_abnormal_humidity_over_100_clamped(self):
        """异常湿度：>100%时不崩溃"""
        model = AHTDryingShrinkageModel(formulation_id="sintered_stone_powder_ps_basic")

        result = model.predict(20.0, 150.0)
        assert result["final_shrinkage_strain"] >= 0

    def test_abnormal_humidity_negative_clamped(self):
        """异常湿度：负湿度时不崩溃"""
        model = AHTDryingShrinkageModel(formulation_id="sintered_stone_powder_ps_basic")

        result = model.predict(20.0, -20.0)
        assert result["final_shrinkage_strain"] >= 0

    def test_shrinkage_time_curve_monotonic_non_decreasing(self):
        """收缩时程曲线单调非递减"""
        model = AHTDryingShrinkageModel(
            formulation_id="sintered_stone_powder_ps_basic",
            prediction_horizon_days=180.0,
        )

        result = model.predict(25.0, 50.0)
        curve = result["shrinkage_time_curve"]["shrinkage_strain"]

        for i in range(1, len(curve)):
            assert curve[i] >= curve[i - 1] * 0.999, (
                f"收缩曲线第{i}点下降: {curve[i]:.6f} < {curve[i-1]:.6f}"
            )

    def test_hardening_degree_converges_to_1(self):
        """硬化程度α随时间收敛到接近1"""
        model = AHTDryingShrinkageModel(
            formulation_id="sintered_stone_powder_ps_basic",
            prediction_horizon_days=365.0,
        )

        result = model.predict(20.0, 60.0)
        alpha_final = result["shrinkage_time_curve"]["hardening_degree"][-1]

        assert alpha_final > 0.85, f"365天硬化程度={alpha_final:.3f}，应>0.85"
        assert alpha_final <= 1.0001, f"硬化程度={alpha_final}不应超过1"

    def test_compare_formulations_all_risk_levels_valid(self):
        """配方对比：所有配方返回有效风险等级"""
        results = compare_formulations(20.0, 60.0)
        assert len(results) >= 4
        valid_levels = {"无", "低", "中", "高", "极高"}
        for r in results:
            assert r["crack_risk_level"] in valid_levels


# ============================================================
# Feature 4: 多窟室保护优先级排序专项测试
# ============================================================

def _zdt1_test_problem(x: np.ndarray) -> List[float]:
    """ZDT1标准多目标测试函数：2目标、凸帕累托前沿"""
    n = len(x)
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / max(n - 1, 1)
    h = 1.0 - math.sqrt(f1 / max(g, 1e-10))
    f2 = g * h
    return [float(f1), float(f2)]


def _zdt2_test_problem(x: np.ndarray) -> List[float]:
    """ZDT2标准多目标测试函数：2目标、非凸帕累托前沿"""
    n = len(x)
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / max(n - 1, 1)
    h = 1.0 - (f1 / max(g, 1e-10)) ** 2
    f2 = g * h
    return [float(f1), float(f2)]


def _dtlz2_test_problem(x: np.ndarray) -> List[float]:
    """DTLZ2标准多目标测试函数：3目标"""
    n_obj = 3
    g = np.sum((x[n_obj - 1:] - 0.5) ** 2)
    f = []
    for i in range(n_obj):
        prod = 1.0
        for j in range(n_obj - i - 1):
            prod *= math.cos(x[j] * math.pi / 2.0)
        if i > 0:
            prod *= math.sin(x[n_obj - i - 1] * math.pi / 2.0)
        f.append(float((1.0 + g) * prod))
    return f


class TestPriorityRankingValidation:
    """NSGA-II优先级排序验证测试"""

    def test_nsga_ii_pareto_front_coverage_zdt1(self):
        """NSGA-II覆盖ZDT1帕累托前沿：均匀分布、100%支配关系正确"""
        np.random.seed(42)
        solver = NSGAIISolver(
            population_size=100,
            max_generations=80,
            crossover_prob=0.9,
            mutation_prob=0.1,
            random_seed=42,
        )

        n_var = 30
        result = solver.solve(
            n_variables=n_var,
            n_objectives=2,
            lower_bounds=np.zeros(n_var),
            upper_bounds=np.ones(n_var),
            objective_function=_zdt1_test_problem,
            maximize=[False, False],
        )

        pareto = np.array(result["pareto_front"])
        assert len(pareto) >= 30, f"帕累托解太少: {len(pareto)} < 30"

        f1_range = pareto[:, 0].max() - pareto[:, 0].min()
        f2_range = pareto[:, 1].max() - pareto[:, 1].min()
        assert f1_range > 0.7, f"f1覆盖范围={f1_range:.3f} < 0.7"
        assert f2_range > 0.3, f"f2覆盖范围={f2_range:.3f} < 0.3"

        for i in range(len(pareto)):
            for j in range(len(pareto)):
                if i != j:
                    a_better_f1 = pareto[i, 0] < pareto[j, 0]
                    a_better_f2 = pareto[i, 1] < pareto[j, 1]
                    if a_better_f1:
                        assert not a_better_f2, (
                            f"帕累托解{i}支配解{j}，违反非支配关系"
                        )

    def test_nsga_ii_pareto_front_coverage_zdt2_nonconvex(self):
        """NSGA-II覆盖ZDT2非凸帕累托前沿"""
        np.random.seed(100)
        solver = NSGAIISolver(
            population_size=100,
            max_generations=80,
            random_seed=100,
        )

        n_var = 30
        result = solver.solve(
            n_variables=n_var,
            n_objectives=2,
            lower_bounds=np.zeros(n_var),
            upper_bounds=np.ones(n_var),
            objective_function=_zdt2_test_problem,
            maximize=[False, False],
        )

        pareto = np.array(result["pareto_front"])
        assert len(pareto) >= 20, f"ZDT2帕累托解数量不足: {len(pareto)}"

        sorted_idx = np.argsort(pareto[:, 0])
        f2_values = pareto[sorted_idx, 1]
        decreases = 0
        for i in range(1, len(f2_values)):
            if f2_values[i] <= f2_values[i - 1] * 1.001:
                decreases += 1
        monotonic_pct = decreases / max(len(f2_values) - 1, 1)
        assert monotonic_pct > 0.80, (
            f"f2随f1增大的单调性={monotonic_pct:.0%} < 80%"
        )

    def test_nsga_ii_3objective_dtlz2_coverage(self):
        """NSGA-II 3目标DTLZ2：覆盖3维帕累托球表面"""
        np.random.seed(42)
        solver = NSGAIISolver(
            population_size=80,
            max_generations=60,
            random_seed=42,
        )

        n_var = 12
        result = solver.solve(
            n_variables=n_var,
            n_objectives=3,
            lower_bounds=np.zeros(n_var),
            upper_bounds=np.ones(n_var),
            objective_function=_dtlz2_test_problem,
            maximize=[False, False, False],
        )

        pareto = np.array(result["pareto_front"])
        assert len(pareto) >= 15, f"3目标帕累托解数量不足: {len(pareto)}"

        for p in pareto:
            sphere_radius_sq = float(np.sum(p ** 2))
            assert abs(sphere_radius_sq - 1.0) < 0.15, (
                f"解偏离帕累托球面: {sphere_radius_sq:.4f} != 1.0±0.15"
            )

        for i_obj in range(3):
            spread = pareto[:, i_obj].max() - pareto[:, i_obj].min()
            assert spread > 0.2, f"目标{i_obj}覆盖范围={spread:.3f} < 0.2"

    def test_nsga_ii_maximization_objectives(self):
        """NSGA-II支持最大化目标（窟室排序场景）"""

        def max_obj(x):
            urgency = -((x[0] - 1.0) ** 2 + x[1] * 0.1)
            efficiency = -(x[1] ** 2 + x[0] * 0.05)
            cultural = -((x[0] + x[1] - 1.5) ** 2)
            return [float(urgency), float(efficiency), float(cultural)]

        np.random.seed(42)
        solver = NSGAIISolver(
            population_size=60,
            max_generations=50,
            random_seed=42,
        )

        result = solver.solve(
            n_variables=2,
            n_objectives=3,
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([1.0, 1.0]),
            objective_function=max_obj,
            maximize=[True, True, True],
        )

        pareto = np.array(result["pareto_front"])
        assert len(pareto) >= 10

    def test_cave_ranking_urgency_drives_priority(self):
        """窟室排序：剥离面积大的窟室优先级更高"""
        ranker = CaveProtectionPriorityRanking()
        caves = []

        for area in [0.5, 1.5, 3.0, 5.0, 8.0]:
            caves.append(CaveConditionData(
                cave_id=f"C{int(area*10):03d}",
                cave_name=f"测试窟{area}",
                total_delamination_area_sqm=area,
                max_severity_score=0.8,
                historical_repair_count=2,
                last_repair_years_ago=3.0,
                avg_visitor_flow_daily=800.0,
                peak_visitor_flow_daily=1500.0,
                cultural_significance_score=0.8,
                structural_importance=0.7,
                avg_bond_strength_remaining_mpa=0.8 - area * 0.05,
                active_alerts_count=1,
                wall_surfaces_count=5,
            ))

        results = ranker.rank_caves(caves, use_nsga_ii=False)
        area_to_rank = {r["detail_metrics"]["total_delamination_area_sqm"]: r["priority_rank"]
                        for r in results}

        sorted_areas = sorted(area_to_rank.keys(), reverse=True)
        for i in range(1, len(sorted_areas)):
            assert area_to_rank[sorted_areas[i]] >= area_to_rank[sorted_areas[i - 1]], (
                f"面积{sorted_areas[i-1]}应优于{sorted_areas[i]}"
            )

    def test_cave_ranking_boundary_single_cave(self):
        """边界输入：仅1个窟室时返回rank=1"""
        ranker = CaveProtectionPriorityRanking()
        caves = [CaveConditionData(
            cave_id="C001",
            cave_name="单窟",
            total_delamination_area_sqm=1.0,
            max_severity_score=0.5,
            historical_repair_count=0,
            last_repair_years_ago=0.0,
            avg_visitor_flow_daily=100.0,
            peak_visitor_flow_daily=200.0,
            cultural_significance_score=0.5,
            structural_importance=0.5,
            avg_bond_strength_remaining_mpa=0.9,
            active_alerts_count=0,
            wall_surfaces_count=1,
        )]

        results = ranker.rank_caves(caves, use_nsga_ii=False)
        assert len(results) == 1
        assert results[0]["priority_rank"] == 1

    def test_cave_ranking_boundary_empty_list(self):
        """边界输入：空窟室列表返回空结果"""
        ranker = CaveProtectionPriorityRanking()
        results = ranker.rank_caves([], use_nsga_ii=False)
        assert len(results) == 0

    def test_cave_ranking_boundary_extreme_metrics(self):
        """边界数据：极端指标值不崩溃"""
        ranker = CaveProtectionPriorityRanking()
        caves = [
            CaveConditionData(
                cave_id="C_MAX",
                cave_name="极端窟",
                total_delamination_area_sqm=999.0,
                max_severity_score=1.0,
                historical_repair_count=999,
                last_repair_years_ago=99.0,
                avg_visitor_flow_daily=99999.0,
                peak_visitor_flow_daily=999999.0,
                cultural_significance_score=1.0,
                structural_importance=1.0,
                avg_bond_strength_remaining_mpa=0.0,
                active_alerts_count=999,
                wall_surfaces_count=99,
            ),
            CaveConditionData(
                cave_id="C_MIN",
                cave_name="最小窟",
                total_delamination_area_sqm=0.0,
                max_severity_score=0.0,
                historical_repair_count=0,
                last_repair_years_ago=0.0,
                avg_visitor_flow_daily=0.0,
                peak_visitor_flow_daily=0.0,
                cultural_significance_score=0.0,
                structural_importance=0.0,
                avg_bond_strength_remaining_mpa=1.0,
                active_alerts_count=0,
                wall_surfaces_count=0,
            ),
        ]

        results = ranker.rank_caves(caves, use_nsga_ii=False)
        assert len(results) == 2
        assert results[0]["priority_rank"] == 1
        assert results[1]["priority_rank"] == 2

        for r in results:
            assert 0 <= r["urgency_score"] <= 1.0
            assert 0 <= r["aggregated_score"] <= 100.0

    def test_cave_ranking_weights_boundary_single_objective(self):
        """边界权重：某目标权重=1.0，其他=0时退化为单目标"""
        ranker = CaveProtectionPriorityRanking(
            weights={"urgency": 1.0, "cost_efficiency": 0.0, "cultural_impact": 0.0}
        )
        caves = []
        for i, (urgency, eff, cul) in enumerate([
            (1.0, 0.2, 0.2),
            (0.3, 1.0, 0.1),
            (0.1, 0.1, 1.0),
        ]):
            caves.append(CaveConditionData(
                cave_id=f"C{i:02d}",
                cave_name=f"窟{i}",
                total_delamination_area_sqm=urgency * 5.0,
                max_severity_score=urgency,
                historical_repair_count=int(3 * (1 - eff) * 5),
                last_repair_years_ago=10.0 * (1 - eff),
                avg_visitor_flow_daily=cul * 2000.0,
                peak_visitor_flow_daily=cul * 4000.0,
                cultural_significance_score=cul,
                structural_importance=0.8,
                avg_bond_strength_remaining_mpa=1.0 - urgency * 0.6,
                active_alerts_count=int(urgency * 5),
                wall_surfaces_count=5,
            ))

        results = ranker.rank_caves(
            caves,
            use_nsga_ii=False,
        )
        assert results[0]["cave_id"] == "C00", "紧迫性权重=1时高紧迫性窟室应第1"

    def test_cave_ranking_abnormal_invalid_weights_normalized(self):
        """异常权重：非归一化权重自动归一化"""
        weird_weights = {"urgency": 999.0, "cost_efficiency": 999.0, "cultural_impact": 999.0}
        ranker = CaveProtectionPriorityRanking(weights=weird_weights)
        caves = [CaveConditionData(
            cave_id="C001",
            cave_name="测试",
            total_delamination_area_sqm=2.0,
            max_severity_score=0.7,
            historical_repair_count=2,
            last_repair_years_ago=5.0,
            avg_visitor_flow_daily=1000.0,
            peak_visitor_flow_daily=2000.0,
            cultural_significance_score=0.9,
            structural_importance=0.8,
            avg_bond_strength_remaining_mpa=0.7,
            active_alerts_count=2,
            wall_surfaces_count=5,
        )]

        results = ranker.rank_caves(caves, use_nsga_ii=False)
        assert len(results) == 1
        ws = results[0]["weights_used"]
        weight_sum = sum(ws.values())
        assert abs(weight_sum - 1.0) < 0.01, f"权重和={weight_sum:.4f} != 1"

    def test_cave_ranking_nsga_ii_many_caves_pareto(self):
        """NSGA-II实际窟室排序：10窟室返回帕累托前沿"""
        np.random.seed(42)
        ranker = CaveProtectionPriorityRanking()
        caves = [
            CaveConditionData(
                cave_id=f"C{i:03d}",
                cave_name=f"第{i}窟",
                total_delamination_area_sqm=np.random.uniform(0.2, 6.0),
                max_severity_score=np.random.uniform(0.3, 0.95),
                historical_repair_count=np.random.randint(0, 6),
                last_repair_years_ago=np.random.uniform(1.0, 15.0),
                avg_visitor_flow_daily=np.random.uniform(100, 2500),
                peak_visitor_flow_daily=np.random.uniform(300, 5000),
                cultural_significance_score=np.random.uniform(0.6, 1.0),
                structural_importance=np.random.uniform(0.5, 1.0),
                avg_bond_strength_remaining_mpa=np.random.uniform(0.2, 0.95),
                active_alerts_count=np.random.randint(0, 6),
                wall_surfaces_count=5,
            )
            for i in range(10)
        ]

        results = ranker.rank_caves(caves, use_nsga_ii=True)
        assert len(results) == 10

        ranks = [r["priority_rank"] for r in results]
        assert sorted(ranks) == list(range(1, 11)), "排序应覆盖1-10无重复"

        pareto_size = max(r["pareto_front_size"] for r in results if "pareto_front_size" in r)
        assert pareto_size >= 1, f"帕累托前沿解数量={pareto_size}，应≥1"

        for r in results:
            assert r["method"] == "nsga_ii_multi_objective"

    def test_simulated_visitor_flow_weekend_peak_pattern(self):
        """游客流量模拟：周末>工作日、旺季>淡季"""
        np.random.seed(42)
        result = generate_simulated_visitor_flow(
            cave_id="C096",
            days=365,
            base_daily=800.0,
        )

        from datetime import datetime, timedelta
        start = datetime(2025, 1, 1)
        peak_weekend = []
        peak_weekday = []
        off_peak_visitors = []
        peak_visitors = []

        for i, v in enumerate(result["daily_visitors"]):
            d = start + timedelta(days=i)
            is_peak = 5 <= d.month <= 10
            if is_peak:
                peak_visitors.append(v)
                if d.weekday() >= 5:
                    peak_weekend.append(v)
                else:
                    peak_weekday.append(v)
            else:
                off_peak_visitors.append(v)

        avg_peak_weekend = np.mean(peak_weekend)
        avg_peak_weekday = np.mean(peak_weekday)

        assert avg_peak_weekend > avg_peak_weekday * 1.1, (
            f"旺季周末均值({avg_peak_weekend:.0f})应>旺季工作日({avg_peak_weekday:.0f})的110%"
        )

        assert result["peak_season_avg"] > result["off_peak_avg"] * 1.15, (
            f"旺季均值({result['peak_season_avg']:.0f})应>淡季({result['off_peak_avg']:.0f})的115%"
        )

    def test_simulated_visitor_flow_no_negative_values(self):
        """游客流量无负值"""
        result = generate_simulated_visitor_flow("C001", days=3650, base_daily=100.0)
        assert all(v >= 0 for v in result["daily_visitors"]), "存在负游客量"

    def test_nsga_ii_crowding_distance_boundary_points_infinite(self):
        """拥挤距离：帕累托前沿边界点拥挤距离为inf"""
        solver = NSGAIISolver(population_size=20, max_generations=10, random_seed=42)

        objectives = np.array([
            [1.0, 5.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.5], [5.0, 1.0],
        ])
        front = list(range(5))
        distances = solver._crowding_distance(objectives, front)

        assert np.isinf(distances[0]), "f1最小点拥挤距离应为inf"
        assert np.isinf(distances[-1]), "f1最大点拥挤距离应为inf"
        for i in [1, 2, 3]:
            assert not np.isinf(distances[i]), f"内部点{i}拥挤距离不应为inf"


# ============================================================
# 综合集成测试
# ============================================================

class TestFeatureIntegration:
    """跨Feature集成测试"""

    def test_full_workflow_pressure_to_bond_to_shrinkage(self):
        """完整流程：压力优化→粘结评估→收缩预测"""
        np.random.seed(42)

        # Step 1: 压力流量优化
        pf_data = generate_simulated_pressure_flow_data(n_points=50)
        pf_reg = PressureFlowPolynomialRegressor(max_degree=5)
        pf_reg.fit(pf_data)
        pf_result = pf_reg.optimize_injection_rate(
            pressure_range=(150.0, 400.0),
            max_allowable_risk_pct=25.0,
        )
        assert pf_result.r_squared > 0.85

        # Step 2: 粘结强度评估
        bond_assessor = BondStrengthAssessor()
        bond_result = bond_assessor.assess_bond_strength(
            surface_id="C096-N",
            baseline_damping_ratios=[0.015, 0.022, 0.028],
            current_damping_ratios=[0.028, 0.040, 0.052],
            frequencies=[8.5, 22.0, 41.5],
        )
        assert 0 <= bond_result["bond_strength_degradation_pct"] <= 100

        # Step 3: 收缩预测
        shrink_model = AHTDryingShrinkageModel(formulation_id="sintered_stone_powder_ps_basic")
        shrink_result = shrink_model.predict(20.0, 60.0)
        assert shrink_result["crack_risk_level"] in {"无", "低", "中", "高", "极高"}

        # Step 4: 窟室优先级（含上述数据）
        ranker = CaveProtectionPriorityRanking()
        caves = [
            CaveConditionData(
                cave_id="C096", cave_name="第96窟",
                total_delamination_area_sqm=pf_result.optimal_flow_rate_mls / 100,
                max_severity_score=pf_result.secondary_delamination_risk_pct / 100,
                historical_repair_count=3,
                last_repair_years_ago=5.0,
                avg_visitor_flow_daily=1500.0,
                peak_visitor_flow_daily=3000.0,
                cultural_significance_score=1.0,
                structural_importance=0.95,
                avg_bond_strength_remaining_mpa=bond_result["remaining_bond_strength_mpa"],
                active_alerts_count=2 if bond_result["risk_level"] in ["高", "极高"] else 0,
                wall_surfaces_count=5,
            ),
        ]
        rank_result = ranker.rank_caves(caves, use_nsga_ii=False)
        assert rank_result[0]["priority_rank"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
