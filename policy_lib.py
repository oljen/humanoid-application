"""Shared helpers: load trained policies, set up LIBERO, move the bowl, run a policy."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, torch.nn as nn, torchvision
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

DEV, CHUNK, EXEC, MAX_STEPS = "cuda", 8, 4, 300
EEF, BOWL = "robot0_eef_pos", "akita_black_bowl_1"

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        def encoder():
            m = torchvision.models.resnet18(weights=None)
            m.fc = nn.Identity()
            return m
        self.enc_a, self.enc_w = encoder(), encoder()
        self.head = nn.Sequential(nn.Linear(512 * 2 + 5, 512), nn.ReLU(),
                                  nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, CHUNK * 7))
        self.register_buffer("m", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("s", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)
    def forward(self, a, w, prop):
        return self.head(torch.cat([self.enc_a((a - self.m) / self.s),
                                    self.enc_w((w - self.m) / self.s), prop], 1)).view(-1, CHUNK, 7)

def load_models(variant, seeds):
    out = []
    for s in seeds:
        ck = torch.load(f"models/{variant}_s{s}.pt", map_location=DEV, weights_only=False)
        m = Policy().to(DEV); m.load_state_dict(ck["model"]); m.eval()
        out.append((m, ck["p_mean"], ck["p_std"]))
    return out

def make_env():
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(2)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)

def bowl_q(env):
    return env.env.sim.data.get_joint_qpos(env.env.objects_dict[BOWL].joints[0]).copy()

def move_bowl(env, dx, dy):
    joint = env.env.objects_dict[BOWL].joints[0]
    q = bowl_q(env); q[0] += dx; q[1] += dy
    env.env.sim.data.set_joint_qpos(joint, q); env.env.sim.forward()

def reset_at(env, dx, dy, seed):
    """New random layout, bowl shifted by (dx, dy). Returns obs, or None if the placement is invalid."""
    np.random.seed(seed); env.seed(seed)
    env.reset()
    for _ in range(10):
        env.step([0] * 6 + [-1])
    q0 = bowl_q(env)
    move_bowl(env, dx, dy)
    for _ in range(5):
        obs, _, _, _ = env.step([0] * 6 + [-1])
    q1 = bowl_q(env)
    bad = (np.linalg.norm(q1[:2] - obs["plate_1_pos"][:2]) < 0.13 or
           np.linalg.norm(q1[:2] - (q0[:2] + [dx, dy])) > 0.02 or abs(q1[2] - q0[2]) > 0.01)
    return None if bad else obs

def predict(model, p_mean, p_std, obs):
    im = lambda k: torch.from_numpy(np.ascontiguousarray(obs[k])).permute(2, 0, 1)[None].float().to(DEV)
    prop = np.concatenate([obs[EEF], obs["robot0_gripper_qpos"]]).astype(np.float32)
    prop = torch.from_numpy((prop - p_mean) / p_std)[None].to(DEV)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model(im("agentview_image"), im("robot0_eye_in_hand_image"), prop)[0].float().cpu().numpy()

def disagreement(models, obs):
    chunks = np.stack([predict(m, pm, ps, obs) for m, pm, ps in models])
    return float(chunks[:, :EXEC, :3].std(0).mean())

def rollout(env, model, obs, rotate=False):
    m, pm, ps = model
    queue = []
    for t in range(MAX_STEPS):
        if not queue:
            queue = list(predict(m, pm, ps, obs)[:EXEC])
        act = np.clip(queue.pop(0), -1, 1)
        act[6] = 1.0 if act[6] > 0 else -1.0
        if not rotate:
            act[3:6] = 0.0
        obs, _, done, _ = env.step(act)
        if done:
            return True
    return False
