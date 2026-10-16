"""Controlled test: hover above the bowl's grip spot, descend straight down,
close, lift. Try several heights and report which ones pick the bowl up."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

BOWL, EEF = "akita_black_bowl_1_pos", "robot0_eef_pos"
OFFSET = np.array([-0.002, 0.042])   # expert grip spot relative to bowl (from your last run)

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(2)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
env.seed(0)
init = suite.get_task_init_states(2)[0]

def go(obs, target, grip, steps, tol=0.005):
    for _ in range(steps):
        a = np.zeros(7)
        a[:3] = np.clip((target - obs[EEF]) / 0.05, -1, 1)
        a[6] = grip
        obs, _, _, _ = env.step(a)
        if tol and np.linalg.norm(target - obs[EEF]) < tol:
            break
    return obs

for z in [0.935, 0.945, 0.955, 0.965, 0.975, 0.990]:
    env.reset()
    obs = env.set_init_state(init)
    for _ in range(5):
        obs, _, _, _ = env.step([0] * 6 + [-1])
    bowl0 = obs[BOWL].copy()
    spot = np.append(bowl0[:2] + OFFSET, z)
    obs = go(obs, spot + [0, 0, 0.10], -1, 80)     # hover above
    obs = go(obs, spot, -1, 60)                     # straight down
    shift = np.linalg.norm(obs[BOWL][:2] - bowl0[:2])
    reached = obs[EEF][2]
    obs = go(obs, spot, 1, 25, tol=0)               # close and wait
    obs = go(obs, spot + [0, 0, 0.12], 1, 40, tol=0)  # lift
    lift = obs[BOWL][2] - bowl0[2]
    print(f"target z {z:.3f} | reached z {reached:.3f} | bowl pushed {shift*100:.1f}cm "
          f"before closing | bowl lifted {lift*100:.1f}cm")
env.close()
