import os, glob, json, time, argparse
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision, h5py, imageio
from huggingface_hub import hf_hub_download
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

p = argparse.ArgumentParser()
p.add_argument("variant", choices=["uniform", "weighted", "verified", "expert", "dup", "recovery", "noisy", "targeted", "random"])
p.add_argument("--steps", type=int, default=15000)
p.add_argument("--episodes", type=int, default=50)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
DEV, CHUNK, EXEC, MAX_STEPS, TASK_ID, BATCH = "cuda", 8, 4, 300, 2, 64
EEF = "robot0_eef_pos"
KEYS = ["agentview", "wrist", "eef_pos", "gripper", "actions"]

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(TASK_ID)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
env.seed(args.seed)
init_states = suite.get_task_init_states(TASK_ID)

# ---------- data ----------
def load_phone(variant):
    demos = []
    for f in sorted(glob.glob("retargeted/*.npz")):
        z = np.load(f)
        name = os.path.basename(f)[:-4]
        if name.split("_")[0] not in ("good", "hard", "fail"):   # ORIG_GROUPS: no blurred copies
            continue
        if variant == "verified" and not bool(z["success"]):
            continue
        d = {k: z[k] for k in KEYS}
        if variant == "weighted":
            conf = np.load(f"tracks/{name}_unc.npz")["conf"]
            fr = np.clip(np.round(z["frame"]).astype(int), 0, len(conf) - 1)
            d["weight"] = conf[fr]
        demos.append(d)
    return demos

def load_expert(n):
    """Replay LIBERO expert demos in this exact env so images match the phone data."""
    os.makedirs("expert", exist_ok=True)
    path = hf_hub_download("yifengzhu-hf/LIBERO-datasets",
                           f"libero_spatial/{task.name}_demo.hdf5", repo_type="dataset")
    demos = []
    with h5py.File(path, "r") as f:
        for i in range(len(f["data"])):
            if len(demos) >= n:
                break
            cache = f"expert/demo_{i}.npz"
            if not os.path.exists(cache):
                acts = f[f"data/demo_{i}/actions"][()]
                env.reset()
                obs = env.set_init_state(f[f"data/demo_{i}/states"][0])
                rec, ok = {k: [] for k in KEYS}, False
                for a in acts:
                    rec["agentview"].append(obs["agentview_image"])
                    rec["wrist"].append(obs["robot0_eye_in_hand_image"])
                    rec["eef_pos"].append(obs[EEF]); rec["gripper"].append(obs["robot0_gripper_qpos"])
                    rec["actions"].append(a)
                    obs, _, done, _ = env.step(a)
                    if done:
                        ok = True
                        break
                np.savez_compressed(cache, success=ok, **{k: np.array(v) for k, v in rec.items()})
            z = np.load(cache)
            if bool(z["success"]):
                demos.append({k: z[k] for k in KEYS})
    return demos

def load_dir(d):
    out = []
    for f in sorted(glob.glob(f"{d}/*.npz")):
        z = np.load(f)
        if bool(z["success"]):
            out.append({k: z[k] for k in KEYS})
    return out

if args.variant == "expert":
    demos = load_expert(len(load_phone("verified")))
elif args.variant == "dup":        # control: same extra data, but copies of good demos
    demos = load_phone("verified")
    demos = demos + [dict(d) for d in sorted(demos, key=lambda d: -len(d["actions"]))[:4]]
elif args.variant == "recovery":   # nudged replays, labelled with the clean correction
    demos = load_phone("verified") + load_dir("recovery")
elif args.variant == "noisy":      # same nudged replays, labelled with the nudged action
    demos = load_phone("verified") + load_dir("recovery_noisy")
elif args.variant == "targeted":   # 42 demos + new clips placed at the unsure spots
    demos = load_phone("uniform") + load_dir("new_targeted")
elif args.variant == "random":     # 42 demos + the same new clips placed at random spots
    demos = load_phone("uniform") + load_dir("new_random")
else:
    demos = load_phone(args.variant)

A, W, PROP, ACT, WT = [], [], [], [], []
for d in demos:
    T = len(d["actions"])
    acts = d["actions"].astype(np.float32)
    for t in range(T):
        ACT.append(acts[np.minimum(np.arange(t, t + CHUNK), T - 1)])
    A.append(d["agentview"]); W.append(d["wrist"])
    PROP.append(np.concatenate([d["eef_pos"], d["gripper"]], 1).astype(np.float32))
    WT.append(d.get("weight", np.ones(T)).astype(np.float32))
A = torch.from_numpy(np.concatenate(A)).permute(0, 3, 1, 2).contiguous().to(DEV)   # uint8 on GPU
W = torch.from_numpy(np.concatenate(W)).permute(0, 3, 1, 2).contiguous().to(DEV)
PROP = np.concatenate(PROP); p_mean, p_std = PROP.mean(0), PROP.std(0) + 1e-6
PROP = torch.from_numpy((PROP - p_mean) / p_std).to(DEV)
ACT = torch.from_numpy(np.stack(ACT)).to(DEV)
WT = np.concatenate(WT); WT = torch.from_numpy(WT / WT.mean()).to(DEV)   # mean weight 1
N = len(ACT)
print(f"[{args.variant}] {len(demos)} demos, {N} training samples")

# ---------- model ----------
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        def encoder():
            m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
            m.fc = nn.Identity()
            return m
        self.enc_a, self.enc_w = encoder(), encoder()
        self.head = nn.Sequential(nn.Linear(512 * 2 + 5, 512), nn.ReLU(),
                                  nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, CHUNK * 7))
        self.register_buffer("m", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("s", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)
    def forward(self, a, w, prop):           # a, w: float images 0-255
        fa = self.enc_a((a - self.m) / self.s)
        fw = self.enc_w((w - self.m) / self.s)
        return self.head(torch.cat([fa, fw, prop], 1)).view(-1, CHUNK, 7)

def random_shift(x, pad=8):                  # small random image shifts = data augmentation
    x = F.pad(x.float(), (pad,) * 4, mode="replicate")
    h = x.shape[-1] - 2 * pad
    out = torch.empty((x.shape[0], 3, h, h), device=x.device)
    for i in range(x.shape[0]):
        dx, dy = np.random.randint(0, 2 * pad + 1, 2)
        out[i] = x[i, :, dy:dy + h, dx:dx + h]
    return out

model = Policy().to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

# ---------- test helper ----------
def evaluate(n, seed_base, video_prefix=None):
    model.eval()
    wins = []
    for ep in range(n):
        np.random.seed(seed_base + ep); env.seed(seed_base + ep)   # same layouts for every brain
        obs = env.reset()
        for _ in range(5):
            obs, _, _, _ = env.step([0] * 6 + [-1])
        queue, ok, frames = [], False, []
        for t in range(MAX_STEPS):
            if not queue:
                im = lambda k: torch.from_numpy(np.ascontiguousarray(obs[k])).permute(2, 0, 1)[None].float().to(DEV)
                prop = np.concatenate([obs[EEF], obs["robot0_gripper_qpos"]]).astype(np.float32)
                prop = torch.from_numpy((prop - p_mean) / p_std)[None].to(DEV)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    chunk = model(im("agentview_image"), im("robot0_eye_in_hand_image"), prop)[0].float().cpu().numpy()
                queue = list(chunk[:EXEC])
            act = np.clip(queue.pop(0), -1, 1)
            act[6] = 1.0 if act[6] > 0 else -1.0
            if args.variant != "expert":
                act[3:6] = 0.0          # phone demos never rotate the gripper
            if video_prefix and ep < 3:
                frames.append(obs["agentview_image"][::-1])
            obs, _, done, _ = env.step(act)
            if done:
                ok = True
                break
        wins.append(ok)
        if video_prefix:
            if ep < 3:
                imageio.mimsave(f"eval_videos/{video_prefix}_ep{ep}.mp4", frames, fps=20)
            print(f"episode {ep + 1}/{n}: {'SUCCESS' if ok else 'fail'}  (running {np.mean(wins) * 100:.0f}%)")
    model.train()
    return float(np.mean(wins))

# ---------- train; keep the checkpoint that does best on the practice layouts ----------
CHECK_AT = {1000, 2500, 5000, 7500, 10000, args.steps}
VAL_SEEDS, TEST_SEEDS = 20_000, 10_000
os.makedirs("eval_videos", exist_ok=True); os.makedirs("models", exist_ok=True)
best_rate, best_step, best_state = -1.0, 0, None
t0 = time.time()
model.train()
for step in range(1, args.steps + 1):
    i = torch.randint(0, N, (BATCH,), device=DEV)
    prop = PROP[i] + 0.1 * torch.randn_like(PROP[i])          # noise: don't rely only on arm position
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(random_shift(A[i]), random_shift(W[i]), prop)
    loss = ((pred.float() - ACT[i]).abs().mean((1, 2)) * WT[i]).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 500 == 0 or step == 1:
        print(f"step {step}/{args.steps}  loss {loss.item():.4f}  {time.time() - t0:.0f}s")
    if step in CHECK_AT:
        r = evaluate(10, VAL_SEEDS)
        print(f"  practice layouts at step {step}: {r * 100:.0f}%")
        if r > best_rate:
            best_rate, best_step = r, step
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
train_time = time.time() - t0
model.load_state_dict(best_state)
torch.save(dict(model=best_state, p_mean=p_mean, p_std=p_std, step=best_step),
           f"models/{args.variant}_s{args.seed}.pt")
print(f"using checkpoint from step {best_step} (practice score {best_rate * 100:.0f}%)")

# ---------- final test on the 50 untouched TEST layouts ----------
rate = evaluate(args.episodes, TEST_SEEDS, video_prefix=f"{args.variant}_s{args.seed}")
print(f"\n[{args.variant}] SUCCESS RATE {rate * 100:.0f}% over {args.episodes} episodes "
      f"| {len(demos)} demos | best step {best_step} | {train_time / 60:.1f} min")
res_file = "train_results.json"
allres = json.load(open(res_file)) if os.path.exists(res_file) else []
allres.append(dict(variant=args.variant, seed=args.seed, steps=args.steps, best_step=best_step,
                   practice_rate=best_rate, demos=len(demos), samples=N,
                   success_rate=rate, episodes=args.episodes))
json.dump(allres, open(res_file, "w"), indent=1)
env.close()
