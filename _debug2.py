import numpy as np
from backend.algorithms.drying_shrinkage import (
    AHTDryingShrinkageModel, compare_formulations, GROUT_FORMULATIONS
)

# 测试温湿度对收缩的影响
m = AHTDryingShrinkageModel('sintered_stone_powder_ps_basic', prediction_horizon_days=28)

print("=== 湿度影响 (20°C) ===")
for rh in [30, 40, 50, 60, 70, 80, 90, 95, 100]:
    r = m.predict(20, rh, 0.7)
    print(f"  RH={rh:3d}%: shrinkage={r['final_shrinkage_strain']:.6e}, "
          f"risk_idx={r['crack_risk_index']:.3f}, risk_lvl={r['crack_risk_level']}, "
          f"alpha={r['hardening_degree_alpha']:.3f}, humidity_factor={r['time_curve'][0]:.3f}"
          if 'time_curve' in r else '')
    if rh == 60:
        print(f'  Keys: {list(r.keys())}')

print("\n=== 温度影响 (60%RH) ===")
for t in [-10, 0, 10, 20, 30, 40, 50]:
    r = m.predict(t, 60, 0.7)
    print(f"  T={t:3d}°C: shrinkage={r['final_shrinkage_strain']:.6e}, risk_idx={r['crack_risk_index']:.3f}, risk_lvl={r['crack_risk_level']}")

print("\n=== 4种配方 (20°C 60%RH) ===")
form_ids = ['sintered_stone_powder_ps_basic', 'lime_cement_basic', 'silica_fume_modified', 'epoxy_nano_modified']
for fid in form_ids:
    m = AHTDryingShrinkageModel(fid, prediction_horizon_days=28)
    r = m.predict(20, 60, 0.7)
    print(f"  {fid}: shrinkage={r['final_shrinkage_strain']:.6e}, risk_idx={r['crack_risk_index']:.3f}, risk_lvl={r['crack_risk_level']}")

# 测试benchmark场景
print("\n=== Benchmark Scenarios ===")
scenarios = [
    ('sintered_stone_powder_ps_basic', 20, 60, 28),
    ('sintered_stone_powder_ps_basic', 30, 50, 28),
    ('lime_cement_basic', 20, 60, 28),
    ('silica_fume_modified', 20, 60, 28),
    ('epoxy_nano_modified', 20, 60, 28),
    ('sintered_stone_powder_ps_basic', 25, 55, 90),
]
for fid, T, RH, days in scenarios:
    m = AHTDryingShrinkageModel(fid, prediction_horizon_days=days)
    r = m.predict(T, RH, 0.7)
    tag = f"{fid}@T{T}_H{RH}_D{days}"
    print(f"  {tag}: pred={r['final_shrinkage_strain']:.6e}")

print("\n=== compare_formulations (20C, 60RH) ===")
cf = compare_formulations(20, 60, 0.7)
for c in cf:
    print(f"  {c['formulation_id']}: shrinkage={c['shrinkage_strain']:.6e}, risk_idx={c['crack_risk_index']:.3f}, risk_lvl={c['crack_risk_level']}")

# 粘结强度风险等级调试
print("\n=== 粘结强度风险等级 ===")
from backend.algorithms.ssi_modal import BondStrengthAssessor
for delta in [0.000, 0.005, 0.010, 0.020, 0.030, 0.050, 0.075, 0.100, 0.150, 0.200]:
    a = BondStrengthAssessor(minimum_damping_change=0.00001)
    baseline = [0.005, 0.007, 0.010]
    current = [b + delta for b in baseline]
    res = a.assess_bond_strength('t', baseline, current, [10, 25, 45])
    print(f"  delta={delta:.3f}: deg={res['bond_strength_degradation_pct']:6.2f}%, risk={res['risk_level']}, remain={res['remaining_bond_strength_mpa']:.4f}")

# optimize_injection_rate的全风险场景
print("\n=== optimize_injection_rate warning测试 ===")
from backend.algorithms.grouting_diffusion import generate_simulated_pressure_flow_data, PressureFlowPolynomialRegressor
np.random.seed(42)
data = generate_simulated_pressure_flow_data(n_points=40, noise_std_pct=0.03)
reg = PressureFlowPolynomialRegressor(max_degree=4)
reg.fit(data)
for p in range(300, 501, 20):
    q = reg.predict(p)
    r = reg._compute_delamination_risk(p, q)
    print(f"  P={p:3d}kPa, Q={q:7.2f}ml/s, Risk={r:5.1f}%")
