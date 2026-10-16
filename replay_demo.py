import os
os.environ["MUJOCO_GL"] = "egl"
import h5py, imageio, numpy as np
from huggingface_hub import list_repo_files, hf_hub_download
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

TASK_ID = 2  # "pick up the black bowl from table center and place it on the plate"
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(TASK_ID)
print("Task:", task.language)

# Download only this task's demo file (not the whole dataset)
repo = "yifengzhu-hf/LIBERO-datasets"
files = list_repo_files(repo, repo_type="dataset")
match = [f for f in files if "libero_spatial" in f and task.name in f]
print("Found:", match)
path = hf_hub_download(repo, match[0], repo_type="dataset")

with h5py.File(path, "r") as f:
    demo = f["data/demo_0"]
    actions = demo["actions"][()]
    states = demo["states"][()]
    print("Number of demos in file:", len(f["data"]))
print("Action shape:", actions.shape)
print("Action min:", actions.min(0).round(2))
print("Action max:", actions.max(0).round(2))

bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)
env.reset()
obs = env.set_init_state(states[0])

frames, success = [], False
for a in actions:
    obs, reward, done, info = env.step(a)
    frames.append(obs["agentview_image"][::-1])
    if done:
        success = True
        break
env.close()
imageio.mimsave("replay_demo.mp4", frames, fps=20)
print("Success:", success, "| steps:", len(frames))
