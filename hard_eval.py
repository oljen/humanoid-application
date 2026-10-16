import os, sys, json
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, torch.nn as nn, torchvision, imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

VARIANT = sys.argv[1]
QUICK = "--quick" in sys.argv
DEV, CHUNK, EXEC, MAX_STEPS = "cuda", 8, 4, 300
EEF, BOWL = "robot0_eef_pos", "akita_black_bowl_1"
SHIFTS = [-0.10, -0.05, 0.0, 0.05, 0.10]   # metres, bowl moved in x and y
ACTORS = [0, 1]                       # which seeds take turns controlling the robot
EPS = 2                                     # episodes per (actor, shift)
if QUICK:
    SHIFTS, ACTORS, EPS = [0.0], [0], 1

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
        fa = self.enc_a((a - self.m) / self.s)
        fw = self.enc_w((w - self.m) / self.s)
        return self.head(torch.cat([fa, fw, prop], 1)).view(-1, CHUNK, 7)

models = []
SEEDS = [1, 2, 3, 4, 5, 6, 7]   # uniform seed 0 was overwritten by a 50-step test run
for s in SEEDS:
    ck = torch.load(f"models/{VARIANT}_s{s}.pt", map_location=DEV, weights_only=False)
    m = Policy().to(DEV); m.load_state_dict(ck["model"]); m.eval()
    models.append((m, ck["p_mean"], ck["p_std"]))
print(f"loaded {len(models)} {VARIANT} policies")

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(2)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)

def move_bowl(dx, dy):
    sim = env.env.sim
    joint = env.env.objects_dict[BOWL].joints[0]
    q = sim.data.get_joint_qpos(joint).copy()
    q[0] += dx; q[1] += dy
    sim.data.set_joint_qpos(joint, q)
    sim.forward()

def bowl_q():
    return env.env.sim.data.get_joint_qpos(env.env.objects_dict[BOWL].joints[0]).copy()

def predict(model, p_mean, p_std, obs):
    im = lambda k: torch.from_numpy(np.ascontiguousarray(obs[k])).permute(2, 0, 1)[None].float().to(DEV)
    prop = np.concatenate([obs[EEF], obs["robot0_gripper_qpos"]]).astype(np.float32)
    prop = torch.from_numpy((prop - p_mean) / p_std)[None].to(DEV)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model(im("agentview_image"), im("robot0_eye_in_hand_image"), prop)[0].float().cpu().numpy()

def run(actor, dx, dy, seed):
    np.random.seed(seed); env.seed(seed)
    env.reset()
    for _ in range(10):                   # let objects drop onto the table first
        env.step([0] * 6 + [-1])
    q0 = bowl_q()
    move_bowl(dx, dy)
    for _ in range(5):
        obs, _, _, _ = env.step([0] * 6 + [-1])
    bowl_start = obs[BOWL + "_pos"].copy()
    q1 = bowl_q()
    if QUICK:
        print("before move:", np.round(q0[:3], 3), "| after settling:", np.round(q1[:3], 3))
    too_close = np.linalg.norm(bowl_start[:2] - obs["plate_1_pos"][:2]) < 0.13      # on / at the plate
    pushed = np.linalg.norm(q1[:2] - (q0[:2] + [dx, dy])) > 0.02          # hit something
    wrong_height = abs(q1[2] - q0[2]) > 0.01                              # on top of / in something
    if too_close or pushed or wrong_height:
        return None
    queue, ok, unc, frames = [], False, [], []
    for t in range(MAX_STEPS):
        if not queue:
            chunks = np.stack([predict(m, pm, ps, obs) for m, pm, ps in models])   # (8, CHUNK, 7)
            unc.append(float(chunks[:, :EXEC, :3].std(0).mean()))                  # how much the 8 disagree
            queue = list(chunks[actor][:EXEC])
        act = np.clip(queue.pop(0), -1, 1)
        act[6] = 1.0 if act[6] > 0 else -1.0
        if VARIANT != "expert":
            act[3:6] = 0.0
        frames.append(obs["agentview_image"][::-1])
        obs, _, done, _ = env.step(act)
        if done:
            ok = True
            break
    return ok, unc, frames, bowl_start

rows = []
cells = [(dx, dy) for dx in SHIFTS for dy in SHIFTS]
for a in sys.argv:
    if a.startswith("--at="):
        cells = [tuple(float(v) for v in a[5:].split(","))]
for ci, (dx, dy) in enumerate(cells):
    wins = []
    for actor in ACTORS:
        for ep in range(EPS):
            out = run(actor, dx, dy, 30_000 + ci * 10 + ep)
            if out is None:
                continue
            ok, unc, frames, bowl = out
            wins.append(ok)
            rows.append(dict(variant=VARIANT, actor=actor, seed=SEEDS[actor], dx=dx, dy=dy, ep=ep, success=ok,
                             unc_mean=float(np.mean(unc)), unc_max=float(np.max(unc)),
                             unc_first=float(np.mean(unc[:5]))))
            if QUICK:
                imageio.imwrite("hard_quick_start.png", np.kron(frames[0], np.ones((3, 3, 1), dtype=np.uint8)))
                imageio.mimsave(f"hard_quick_{VARIANT}_{dx:+.2f}_{dy:+.2f}.mp4", frames, fps=20)
                print(f"bowl moved to {np.round(bowl, 3)} | success {ok}")
    print(f"shift dx={dx:+.2f} dy={dy:+.2f}: " + (f"{np.mean(wins) * 100:3.0f}% success" if wins else "skipped (invalid placement)"))
if not QUICK:
    json.dump(rows, open(f"hard_{VARIANT}.json", "w"), indent=1)
    print("saved", f"hard_{VARIANT}.json")
env.close()
