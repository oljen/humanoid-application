"""Pick where to add data, BEFORE collecting it.
For the 8 bowl positions 5-7 cm from the centre: how much do the 7 phone-trained policies disagree
on their first action, before moving? Top 3 = targets; 3 random others at the same distance = controls."""
import json, numpy as np
from policy_lib import *

env = make_env()
models = load_models("uniform", range(1, 8))
hard = json.load(open("hard_uniform.json"))
RING = [(-0.05, 0.0), (0.05, 0.0), (0.0, 0.05), (0.0, -0.05),
        (-0.05, -0.05), (0.05, -0.05), (-0.05, 0.05), (0.05, 0.05)]
rows = []
for dx, dy in RING:
    us = []
    for k in range(6):
        obs = reset_at(env, dx, dy, 40_000 + k)
        if obs is not None:
            us.append(disagreement(models, obs))
    succ = [x["success"] for x in hard if round(x["dx"], 2) == dx and round(x["dy"], 2) == dy]
    rows.append(dict(cell=[dx, dy], u0=float(np.mean(us)) if us else None, n_valid=len(us),
                     baseline_success=float(np.mean(succ)) if succ else None))
    print(f"bowl moved ({dx*100:+.0f},{dy*100:+.0f}) cm | disagreement before moving: "
          f"{rows[-1]['u0'] if rows[-1]['u0'] is None else round(rows[-1]['u0'], 4)} "
          f"({len(us)} valid layouts) | baseline success {rows[-1]['baseline_success']}")
valid = sorted([r for r in rows if r["u0"] is not None and r["n_valid"] >= 3], key=lambda r: -r["u0"])
targets = [r["cell"] for r in valid[:3]]
rest = [r["cell"] for r in valid[3:]]
controls = [rest[i] for i in np.random.default_rng(0).choice(len(rest), 3, replace=False)]
json.dump(dict(targets=targets, controls=controls, table=rows,
               rule="top-3 by initial-state disagreement of 7 phone-trained policies among the 8 cells "
                    "5-7 cm from centre; controls = 3 random other cells at the same distance. "
                    "Frozen before any new data was collected."),
          open("targets.json", "w"), indent=1)
print("\nTARGETS (most unsure):", targets)
print("CONTROLS (random, same distance):", controls)
