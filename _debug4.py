import sys
import numpy as np
from backend.algorithms.ssi_modal import BondStrengthAssessor

# 精细扫描各个delta
for sens in [1.0, 2.0, 3.0, 5.0]:
    assessor = BondStrengthAssessor(
        damping_sensitivity_coefficient=sens,
        minimum_damping_change=0.00001,
    )
    baseline = [0.005] * 4
    freqs = [5, 15, 30, 50]

    found = set()
    deg_samples = []
    for delta in np.linspace(0.000, 0.100, 50):
        current = [b + delta for b in baseline]
        res = assessor.assess_bond_strength("t", baseline, current, freqs)
        deg = res['bond_strength_degradation_pct']
        found.add(res['risk_level'])
        if len(deg_samples) < 15:
            deg_samples.append((delta, deg, res['risk_level']))

    sys.stdout.write(f"\n--- sens={sens} ---\n")
    for d, deg, rl in deg_samples:
        sys.stdout.write(f"  delta={d:.4f}: deg={deg:.2f}%, {rl}\n")
    sys.stdout.write(f"  FOUND levels: {found}\n")
    sys.stdout.flush()
