import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("粘结强度模块 - 完整检查与测试脚本")
print("=" * 60)

issues_found = []
fixes_applied = []

print("\n[1] 语法编译检查 (py_compile)")
print("-" * 40)
import py_compile
files = [
    "backend/apps/bond_strength/__init__.py",
    "backend/apps/bond_strength/schemas.py",
    "backend/apps/bond_strength/api.py",
    "backend/apps/bond_strength/main.py",
]
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"  PASS: {f}")
    except py_compile.PyCompileError as e:
        print(f"  FAIL: {f} - {e}")
        issues_found.append(f"语法错误: {f} - {e}")

print("\n[2] 基础库导入检查")
print("-" * 40)
basic_libs = ["numpy", "typing", "pydantic", "logging", "datetime"]
for lib in basic_libs:
    try:
        __import__(lib)
        print(f"  PASS: {lib}")
    except ImportError as e:
        print(f"  FAIL: {lib} - {e}")
        issues_found.append(f"缺少库: {lib}")

print("\n[3] 模块导入检查")
print("-" * 40)
try:
    from backend.apps.bond_strength import schemas
    print("  PASS: schemas.py 导入成功")
except Exception as e:
    print(f"  FAIL: schemas.py - {type(e).__name__}: {e}")
    issues_found.append(f"schemas导入失败: {e}")

try:
    from backend.apps.bond_strength import api
    print("  PASS: api.py 导入成功")
except Exception as e:
    print(f"  FAIL: api.py - {type(e).__name__}: {e}")
    issues_found.append(f"api导入失败: {e}")

try:
    from backend.apps.bond_strength import main
    print("  PASS: main.py 导入成功")
except Exception as e:
    print(f"  WARN: main.py - {type(e).__name__}: {e}")

print("\n[4] schemas.py 字段完整性检查")
print("-" * 40)
from backend.apps.bond_strength.schemas import BondStrengthAssessResponse

required_fields = [
    "surface_id",
    "timestamp",
    "baseline_damping_ratios",
    "current_damping_ratios",
    "raw_current_damping_ratios",
    "frequencies_hz",
    "energy_weights",
    "per_mode_debonding_pct",
    "per_mode_dissipation_energy",
    "bond_strength_degradation_pct",
    "remaining_bond_strength_mpa",
    "baseline_bond_strength_mpa",
    "critical_mode_index",
    "assessment_confidence",
    "risk_level",
    "ambient_temperature_c",
    "temperature_correction_factor",
    "recommendations",
]

actual_fields = list(BondStrengthAssessResponse.model_fields.keys())
print(f"  实际字段数: {len(actual_fields)}")
print(f"  要求字段数: {len(required_fields)}")

for field in required_fields:
    if field in actual_fields:
        print(f"  PASS: {field}")
    else:
        print(f"  FAIL: 缺少字段 {field}")
        issues_found.append(f"schemas缺少字段: {field}")

new_fields = ["raw_current_damping_ratios", "ambient_temperature_c", "temperature_correction_factor"]
print("\n  [新增字段检查]")
for f in new_fields:
    if f in actual_fields:
        print(f"  PASS: 新增字段 {f} 存在")
    else:
        print(f"  FAIL: 新增字段 {f} 缺失")
        issues_found.append(f"新增字段缺失: {f}")

print("\n[5] BondStrengthAssessor 类一致性检查")
print("-" * 40)
from backend.apps.bond_strength.api import BondStrengthAssessor as API_Assessor
from backend.algorithms.ssi_modal import BondStrengthAssessor as SSI_Assessor
import inspect

api_init_params = list(inspect.signature(API_Assessor.__init__).parameters.keys())
ssi_init_params = list(inspect.signature(SSI_Assessor.__init__).parameters.keys())

required_init_params = ["self", "baseline_bond_strength_mpa", "damping_sensitivity_coefficient",
                        "minimum_damping_change", "reference_damping_ratio",
                        "reference_temperature_c", "temperature_activation_factor"]

print(f"  api.py __init__ 参数: {api_init_params}")
print(f"  ssi_modal.py __init__ 参数: {ssi_init_params}")

for p in required_init_params:
    if p in api_init_params:
        print(f"  PASS: __init__ 参数 {p}")
    else:
        print(f"  FAIL: __init__ 缺少参数 {p}")
        issues_found.append(f"api.py __init__缺少参数: {p}")

required_methods = ["_temperature_correction_factor", "_correct_damping_for_temperature",
                    "_damping_energy_dissipation", "_interface_debonding_model",
                    "_modal_strain_energy_distribution", "assess_bond_strength"]

for m in required_methods:
    if hasattr(API_Assessor, m):
        print(f"  PASS: 方法 {m} 存在")
    else:
        print(f"  FAIL: 缺少方法 {m}")
        issues_found.append(f"api.py缺少方法: {m}")

assess_params = list(inspect.signature(API_Assessor.assess_bond_strength).parameters.keys())
print(f"  assess_bond_strength 参数: {assess_params}")
if "ambient_temperature_c" in assess_params:
    print("  PASS: assess_bond_strength 含 ambient_temperature_c 参数")
else:
    print("  FAIL: assess_bond_strength 缺少 ambient_temperature_c 参数")
    issues_found.append("assess_bond_strength缺少ambient_temperature_c参数")

if api_init_params == ssi_init_params:
    print("  PASS: __init__ 参数与 ssi_modal.py 一致")
else:
    print("  WARN: __init__ 参数与 ssi_modal.py 不一致")

print("\n[6] typing 导入完整性检查")
print("-" * 40)

import re

def check_typing_imports(filepath, name):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    typing_used = set()
    for t in ["List", "Dict", "Optional", "Tuple"]:
        if re.search(rf'\b{t}\b', content):
            typing_used.add(t)

    np_used = bool(re.search(r'np\.ndarray', content))

    import_match = re.search(r'from typing import ([^\n]+)', content)
    imported = set()
    if import_match:
        imported_str = import_match.group(1)
        imported = set([x.strip() for x in imported_str.split(',')])

    print(f"  [{name}]")
    print(f"    使用的 typing 类型: {typing_used if typing_used else '无'}")
    if np_used:
        print(f"    使用 np.ndarray: 是")
    print(f"    实际导入: {imported if imported else '无'}")

    missing = typing_used - imported
    if missing:
        print(f"    FAIL: 缺少导入 {missing}")
        issues_found.append(f"{name} 缺少typing导入: {missing}")
    else:
        print(f"    PASS: typing 导入完整")

check_typing_imports("backend/apps/bond_strength/schemas.py", "schemas.py")
check_typing_imports("backend/apps/bond_strength/api.py", "api.py")

print("\n" + "=" * 60)
print("[问题汇总]")
print("=" * 60)
if issues_found:
    print(f"发现 {len(issues_found)} 个问题:")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")
else:
    print("未发现问题！")

print("\n" + "=" * 60)
print("[核心逻辑功能测试]")
print("=" * 60)

from backend.apps.bond_strength.api import BondStrengthAssessor
import numpy as np

print("\n测试1: 创建评估器实例")
print("-" * 40)
try:
    assessor = BondStrengthAssessor(
        baseline_bond_strength_mpa=0.8,
        reference_temperature_c=20.0,
        temperature_activation_factor=0.025,
    )
    print(f"  PASS: 评估器创建成功")
    print(f"    T_ref_c = {assessor.T_ref_c}")
    print(f"    temp_activation = {assessor.temp_activation}")
except Exception as e:
    print(f"  FAIL: {e}")
    issues_found.append(f"评估器创建失败: {e}")

print("\n测试2: 温度修正因子计算")
print("-" * 40)
try:
    factor_high = assessor._temperature_correction_factor(35.0)
    factor_low = assessor._temperature_correction_factor(5.0)
    factor_ref = assessor._temperature_correction_factor(20.0)
    print(f"  35°C (高于参考15°C): 修正因子 = {factor_high:.4f}")
    print(f"  5°C (低于参考15°C): 修正因子 = {factor_low:.4f}")
    print(f"  20°C (参考温度): 修正因子 = {factor_ref:.4f}")
    assert factor_high > 1.0, "高温修正因子应>1"
    assert factor_low < 1.0, "低温修正因子应<1"
    assert abs(factor_ref - 1.0) < 1e-6, "参考温度修正因子应为1"
    print("  PASS: 温度修正因子计算正确")
except Exception as e:
    print(f"  FAIL: {e}")
    issues_found.append(f"温度修正因子测试失败: {e}")

print("\n测试3: 阻尼比温度修正")
print("-" * 40)
try:
    damping = 0.015
    corrected_high = assessor._correct_damping_for_temperature(damping, 35.0)
    corrected_low = assessor._correct_damping_for_temperature(damping, 5.0)
    print(f"  原始阻尼比: {damping}")
    print(f"  35°C修正后: {corrected_high:.6f}")
    print(f"  5°C修正后: {corrected_low:.6f}")
    assert corrected_high < damping, "高温修正后阻尼比应降低"
    assert corrected_low > damping, "低温修正后阻尼比应升高"
    print("  PASS: 阻尼比温度修正正确")
except Exception as e:
    print(f"  FAIL: {e}")
    issues_found.append(f"阻尼比温度修正测试失败: {e}")

print("\n测试4: 粘结强度评估 (无温度修正)")
print("-" * 40)
try:
    result = assessor.assess_bond_strength(
        surface_id="WALL-001",
        baseline_damping_ratios=[0.012, 0.008, 0.010],
        current_damping_ratios=[0.012, 0.008, 0.010],
        frequencies=[5.0, 12.0, 20.0],
        timestamp="2026-06-12T00:00:00Z",
    )
    print(f"  surface_id: {result['surface_id']}")
    print(f"  risk_level: {result['risk_level']}")
    print(f"  bond_strength_degradation_pct: {result['bond_strength_degradation_pct']:.2f}%")
    print(f"  remaining_bond_strength_mpa: {result['remaining_bond_strength_mpa']:.4f}")
    print(f"  ambient_temperature_c: {result.get('ambient_temperature_c')}")
    print(f"  temperature_correction_factor: {result.get('temperature_correction_factor')}")
    print(f"  raw_current_damping_ratios: {result['raw_current_damping_ratios']}")
    assert result["risk_level"] == "无", f"无劣化时风险等级应为'无', 实际为{result['risk_level']}"
    assert "raw_current_damping_ratios" in result, "缺少raw_current_damping_ratios字段"
    print("  PASS: 无劣化评估正确")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
    issues_found.append(f"无温度修正评估失败: {e}")

print("\n测试5: 粘结强度评估 (含温度修正)")
print("-" * 40)
try:
    result = assessor.assess_bond_strength(
        surface_id="WALL-002",
        baseline_damping_ratios=[0.012, 0.008, 0.010],
        current_damping_ratios=[0.020, 0.015, 0.018],
        frequencies=[5.0, 12.0, 20.0],
        timestamp="2026-06-12T00:00:00Z",
        ambient_temperature_c=35.0,
    )
    print(f"  surface_id: {result['surface_id']}")
    print(f"  risk_level: {result['risk_level']}")
    print(f"  bond_strength_degradation_pct: {result['bond_strength_degradation_pct']:.2f}%")
    print(f"  remaining_bond_strength_mpa: {result['remaining_bond_strength_mpa']:.4f}")
    print(f"  ambient_temperature_c: {result.get('ambient_temperature_c')}")
    print(f"  temperature_correction_factor: {result.get('temperature_correction_factor'):.4f}")
    print(f"  raw_current_damping_ratios: {result['raw_current_damping_ratios']}")
    print(f"  current_damping_ratios (修正后): {result['current_damping_ratios']}")
    assert result["ambient_temperature_c"] == 35.0, "ambient_temperature_c字段值错误"
    assert result["temperature_correction_factor"] is not None, "缺少temperature_correction_factor"
    assert result["temperature_correction_factor"] > 1.0, "高温时修正因子应>1"
    assert len(result["recommendations"]) > 0, "应有建议"
    print("  PASS: 温度修正评估正确")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
    issues_found.append(f"含温度修正评估失败: {e}")

print("\n测试6: Pydantic Schema 验证")
print("-" * 40)
try:
    from backend.apps.bond_strength.schemas import BondStrengthAssessResponse
    resp = BondStrengthAssessResponse(**result)
    print(f"  PASS: BondStrengthAssessResponse 验证成功")
    print(f"    risk_level: {resp.risk_level}")
    print(f"    ambient_temperature_c: {resp.ambient_temperature_c}")
    print(f"    temperature_correction_factor: {resp.temperature_correction_factor}")
    print(f"    raw_current_damping_ratios: {resp.raw_current_damping_ratios}")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
    issues_found.append(f"Schema验证失败: {e}")

print("\n" + "=" * 60)
print("[最终结果]")
print("=" * 60)
if issues_found:
    print(f"共发现 {len(issues_found)} 个问题:")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")
else:
    print("所有检查通过，未发现问题！")
    print("核心逻辑功能测试全部通过！")

print("\n测试完成。")
