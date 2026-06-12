import numpy as np
from backend.algorithms.grouting_diffusion import (
    generate_simulated_pressure_flow_data, PressureFlowPolynomialRegressor
)

data = generate_simulated_pressure_flow_data(n_points=50, pressure_range_kpa=(50, 600), noise_std_pct=0.05, max_flow_rate_mls=500)
reg = PressureFlowPolynomialRegressor(max_degree=5)
reg.fit(data)
print(f'R2 = {reg.r_squared:.4f}, 最优阶数 = {reg.optimal_degree}')
print(f'100kPa 流量 = {reg.predict(100):.2f} ml/s')
print(f'450kPa 流量 = {reg.predict(450):.2f} ml/s')
print(f'300kPa 流量 = {reg.predict(300):.2f} ml/s')

for p in [300, 400, 450, 460, 500]:
    q = reg.predict(p)
    r = reg._compute_delamination_risk(p, q)
    print(f'P={p}kPa, Q={q:.1f}, Risk={r:.1f}%')

# 测试异常数据
bad_data = [generate_simulated_pressure_flow_data(1, (500,500), noise_std_pct=0.9)[0] for _ in range(20)]
bad_data_points = []
for p in np.linspace(50, 600, 20):
    dp = generate_simulated_pressure_flow_data(1, (p,p), noise_std_pct=2.0)[0]
    bad_data_points.append(dp)
try:
    reg2 = PressureFlowPolynomialRegressor()
    reg2.fit(bad_data_points)
    print(f'异常数据 R2 = {reg2.r_squared:.4f}')
except Exception as e:
    print(f'异常数据异常: {type(e).__name__}: {e}')

# 粘结强度
from backend.algorithms.bond_strength import BondStrengthAssessor
a = BondStrengthAssessor()
r = a.assess_bond_strength('t', [0.02,0.03], [0.04,0.06], [10,30])
print(f'BS 劣化={r["bond_strength_degradation_pct"]:.2f}%, 风险={r["risk_level"]}, 剩余={r["remaining_bond_strength_mpa"]:.3f}')
print(f'BS keys: {list(r.keys())}')

# AHT收缩
from backend.algorithms.drying_shrinkage import AHTDryingShrinkageModel, compare_formulations, get_formulation_by_id
m = AHTDryingShrinkageModel('sintered_stone_powder_ps_basic', prediction_horizon_days=28)
r = m.predict(20, 60, 0.7)
print(f'AHT shrinkage={r["final_shrinkage_strain"]:.6e}, risk={r["crack_risk_index"]:.3f}, alpha={r["hardening_degree_alpha"]:.3f}')
print(f'AHT keys: {list(r.keys())}')
if 'experiment_data' in r:
    print('AHT experiment_data exists')

form_ids = ['sintered_stone_powder_ps_basic', 'lime_cement_basic', 'silica_fume_modified', 'epoxy_nano_modified']
for fid in form_ids:
    m = AHTDryingShrinkageModel(fid, prediction_horizon_days=28)
    rr = m.predict(20, 60, 0.7)
    print(f'{fid}: shrinkage={rr["final_shrinkage_strain"]:.6e}')

cf = compare_formulations(20, 60, 0.7)
print(f'compare formulations len={len(cf)}, keys[0]={list(cf[0].keys())}')
for i,c in enumerate(cf):
    print(f'  {c["formulation_id"]}: shrinkage={c["shrinkage_strain"]:.6e}, risk_idx={c["crack_risk_index"]:.3f}')
