import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(2)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
env.seed(0)
init_states = suite.get_task_init_states(2)
for s in range(3):
    env.reset()
    env.set_init_state(init_states[s])
    obs, *_ = env.step([0.] * 7)
    print(f"--- init state {s}")
    for k, v in obs.items():
        if k.endswith("_pos") and ("bowl" in k or "plate" in k or "eef" in k):
            print(f"  {k}: {np.round(v, 3)}")
env.close()
