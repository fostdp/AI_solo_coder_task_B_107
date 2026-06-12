import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from backend.algorithms.grouting_diffusion import (
    PressureFlowPolynomialRegressor,
    PressureFlowDataPoint,
    generate_simulated_pressure_flow_data,
)


def test_data_generation():
    print("=" * 60)
    print("测试1: 生成模拟压力-流量数据")
    print("=" * 60)
    data = generate_simulated_pressure_flow_data(n_points=30)
    print(f"  生成数据点数量: {len(data)}")
    print(f"  前3个数据点:")
    for dp in data[:3]:
        print(f"    P={dp.pressure_kpa:.1f} kPa, Q={dp.flow_rate_mls:.2f} mL/s, reliable={dp.is_reliable}")
    assert len(data) == 30
    print("  ✓ 通过\n")
    return data


def test_model_fit(data):
    print("=" * 60)
    print("测试2: PressureFlowPolynomialRegressor.fit()")
    print("=" * 60)
    regressor = PressureFlowPolynomialRegressor(max_degree=5, ridge_alpha=1e-3)
    regressor.fit(data)
    print(f"  最优多项式阶数: {regressor.optimal_degree}")
    print(f"  R² 拟合优度: {regressor.r_squared:.6f}")
    print(f"  系数数量: {len(regressor.coefficients)} (应为阶数+1={regressor.optimal_degree + 1})")
    print(f"  系数值: {[round(float(c), 6) for c in regressor.coefficients]}")
    assert regressor.optimal_degree is not None
    assert regressor.coefficients is not None
    assert len(regressor.coefficients) == regressor.optimal_degree + 1
    assert 0.0 <= regressor.r_squared <= 1.0 or regressor.r_squared > 0.9
    print("  ✓ 通过\n")
    return regressor


def test_model_predict(regressor):
    print("=" * 60)
    print("测试3: PressureFlowPolynomialRegressor.predict()")
    print("=" * 60)
    test_pressures = [100.0, 200.0, 300.0, 400.0, 500.0]
    predictions = regressor.predict(np.array(test_pressures))
    print(f"  输入压力(kPa): {test_pressures}")
    print(f"  预测流量(mL/s): {[round(float(p), 4) for p in predictions]}")
    assert len(predictions) == len(test_pressures)
    assert all(p >= 0.0 for p in predictions)
    scalar_pred = regressor.predict(250.0)
    print(f"  标量输入 250.0 kPa 的预测: {scalar_pred:.4f} mL/s")
    assert isinstance(scalar_pred, float)
    assert scalar_pred >= 0.0
    print("  ✓ 通过\n")


def test_optimize_injection_rate(regressor):
    print("=" * 60)
    print("测试4: optimize_injection_rate()")
    print("=" * 60)
    result = regressor.optimize_injection_rate(
        pressure_range=(100.0, 500.0),
        target_radius_mm=None,
        max_allowable_risk_pct=30.0,
    )
    print(f"  最优压力: {result.optimal_pressure_kpa:.2f} kPa")
    print(f"  最优流量: {result.optimal_flow_rate_mls:.4f} mL/s")
    print(f"  多项式阶数: {result.polynomial_degree}")
    print(f"  R²: {result.r_squared:.4f}")
    print(f"  分层风险: {result.secondary_delamination_risk_pct:.2f}%")
    print(f"  推荐最大流量: {result.recommended_max_flow_mls:.4f} mL/s")
    print(f"  压力-流量曲线点数: {len(result.pressure_flow_curve)}")
    print(f"  警告信息: {result.warning_message}")
    assert result.optimal_pressure_kpa >= 100.0 and result.optimal_pressure_kpa <= 500.0
    assert result.optimal_flow_rate_mls >= 0.0
    assert len(result.pressure_flow_curve) > 0
    curve_pt = result.pressure_flow_curve[0]
    assert "pressure_kpa" in curve_pt
    assert "flow_rate_mls" in curve_pt
    assert "delamination_risk_pct" in curve_pt
    print("  ✓ 通过\n")
    return result


def test_predict_with_target_radius(regressor):
    print("=" * 60)
    print("测试5: 带目标半径的 optimize_injection_rate()")
    print("=" * 60)
    result = regressor.optimize_injection_rate(
        pressure_range=(100.0, 500.0),
        target_radius_mm=30.0,
        max_allowable_risk_pct=30.0,
    )
    target_flow = 30.0 * 8.0
    print(f"  目标半径: 30.0 mm → 目标流量约: {target_flow:.1f} mL/s")
    print(f"  最优压力: {result.optimal_pressure_kpa:.2f} kPa")
    print(f"  最优流量: {result.optimal_flow_rate_mls:.4f} mL/s")
    print(f"  与目标流量偏差: {abs(result.optimal_flow_rate_mls - target_flow):.4f} mL/s")
    assert result.optimal_flow_rate_mls >= 0.0
    print("  ✓ 通过\n")


def test_edge_case_min_points():
    print("=" * 60)
    print("测试6: 最小数据点拟合（边界情况）")
    print("=" * 60)
    min_data = [
        PressureFlowDataPoint(pressure_kpa=100.0, flow_rate_mls=50.0, elapsed_seconds=0.0, is_reliable=True),
        PressureFlowDataPoint(pressure_kpa=200.0, flow_rate_mls=120.0, elapsed_seconds=0.0, is_reliable=True),
        PressureFlowDataPoint(pressure_kpa=300.0, flow_rate_mls=200.0, elapsed_seconds=0.0, is_reliable=True),
    ]
    regressor = PressureFlowPolynomialRegressor(max_degree=2)
    regressor.fit(min_data)
    print(f"  3个数据点 → 阶数: {regressor.optimal_degree}, R²: {regressor.r_squared:.6f}")
    pred = regressor.predict(250.0)
    print(f"  250 kPa → 预测流量: {pred:.4f} mL/s")
    print("  ✓ 通过\n")


def test_unreliable_points():
    print("=" * 60)
    print("测试7: 含不可靠数据点的拟合")
    print("=" * 60)
    data = [
        PressureFlowDataPoint(pressure_kpa=100.0 * (i + 1), flow_rate_mls=50.0 * (i + 1) + np.random.normal(0, 5),
                              elapsed_seconds=float(i * 10), is_reliable=(i % 5 != 0))
        for i in range(20)
    ]
    unreliable_count = sum(1 for d in data if not d.is_reliable)
    print(f"  总数据点: {len(data)}, 不可靠点: {unreliable_count}")
    regressor = PressureFlowPolynomialRegressor(max_degree=4)
    regressor.fit(data)
    print(f"  最优阶数: {regressor.optimal_degree}, R²: {regressor.r_squared:.6f}")
    print("  ✓ 通过\n")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("#  压力流量多项式回归 - 核心算法功能测试")
    print("#" * 60 + "\n")

    try:
        data = test_data_generation()
        regressor = test_model_fit(data)
        test_model_predict(regressor)
        test_optimize_injection_rate(regressor)
        test_predict_with_target_radius(regressor)
        test_edge_case_min_points()
        test_unreliable_points()

        print("#" * 60)
        print("#  所有测试通过 ✓")
        print("#" * 60 + "\n")
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
