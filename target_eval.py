import os, sys, json
import numpy as np
from policy_lib import *

variant = sys.argv[1]
seeds = [int(s) for s in sys.argv[2].split(",")]
T = json.load(open("targets.json"))
JITTER = [(0.015, 0.015), (-0.015, 0.015), (0.015, -0.015), (-0.015, -0.015)]
EPS = 2
env = make_env()
out = []
for s in seeds:
    model = load_models(variant, [s])[0]
    for r_i, region in enumerate(["targets", "controls"]):
        wins = []
        for ci, (cx, cy) in enumerate(T[region]):
            for ji, (jx, jy) in enumerate(JITTER):
                for e in range(EPS):
                    obs = reset_at(env, cx + jx, cy + jy, 50_000 + r_i * 5000 + ci * 100 + ji * 10 + e)
                    if obs is None:
                        continue
                    ok = rollout(env, model, obs, rotate=(variant == "expert"))
                    wins.append(ok)
                    out.append(dict(variant=variant, seed=s, region=region, cell=[cx, cy],
                                    jitter=[jx, jy], success=bool(ok)))
        print(f"{variant} seed {s} | near {region}: {sum(wins)}/{len(wins)} = {np.mean(wins) * 100:.0f}%")
f = "target_results.json"
res = json.load(open(f)) if os.path.exists(f) else []
json.dump(res + out, open(f, "w"), indent=1)
