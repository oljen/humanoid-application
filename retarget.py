"""Turn tracked hand clips into robot motions in LIBERO and replay them.
Usage: python retarget.py                 (all clips)
       python retarget.py good_IMG_0790   (just these clips)"""
import os, sys, json, glob
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, h5py, imageio
from huggingface_hub import hf_hub_download
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

TASK_ID = 2
ASPECT = 1920 / 1080      # clips are 1920x1080: puts image x and y in the same units
CLOSE, OPEN = 0.12, 0.30  # pinch thresholds
SLOW = 2.0                # replay the human motion at half speed so the arm can keep up
HOLD = 12                 # steps to wait while the gripper closes / opens
RAMP = 0.08               # metres: start descending when this close to the bowl / plate
ARRIVE_TOL = 0.01         # metres: how close the gripper must be before closing / opening
ARRIVE_MAX = 50           # max steps to wait for that
BOWL, PLATE, EEF = "akita_black_bowl_1_pos", "plate_1_pos", "robot0_eef_pos"
NOISE = float(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--noise=")), 0))
OUTDIR = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--outdir=")), "retargeted")
RESULTS = RESULTS if OUTDIR == "retargeted" else f"{OUTDIR}_results.json"
only = [a for a in sys.argv[1:] if not a.startswith("--")]
rng_noise = np.random.default_rng(0)
PLACE = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--place=")), None)
PLACE_CELLS = json.load(open("targets.json"))[PLACE] if PLACE else None
place_i = 0
LABEL_NOISY = "--labels=noisy" in sys.argv

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(TASK_ID)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
env.seed(0)
init_states = suite.get_task_init_states(TASK_ID)

# 1. From 5 expert demos: where the gripper sits relative to the bowl when the bowl
#    actually starts lifting, and relative to the plate on release
path = hf_hub_download("yifengzhu-hf/LIBERO-datasets",
                       f"libero_spatial/{task.name}_demo.hdf5", repo_type="dataset")
g_rel, r_rel, g_quat = [], [], []
with h5py.File(path, "r") as f:
    for d in range(5):
        acts = f[f"data/demo_{d}/actions"][()]
        env.reset()
        obs = env.set_init_state(f[f"data/demo_{d}/states"][0])
        bz0, grasped = obs[BOWL][2], False
        for a in acts:
            if not grasped and obs[BOWL][2] > bz0 + 0.01:
                g_rel.append(obs[EEF] - obs[BOWL]); g_quat.append(obs["robot0_eef_quat"])
                grasped = True
            elif grasped and a[6] < 0:
                r_rel.append(obs[EEF] - obs[PLATE]); break
            obs, _, _, _ = env.step(a)
g_rel, r_rel = np.mean(g_rel, 0), np.mean(r_rel, 0)
env.reset()
obs = env.set_init_state(init_states[0])
G_OFF, R_OFF = g_rel[:2], r_rel[:2]
Z_GRASP = obs[BOWL][2] + g_rel[2]
Z_RELEASE = obs[PLATE][2] + r_rel[2]
Z_HOVER = Z_GRASP + 0.15
Z_CARRY = max(Z_GRASP, Z_RELEASE) + 0.10
print(f"expert: grip spot rel. to bowl {g_rel.round(3)} | release rel. to plate {r_rel.round(3)} | "
      f"z grasp {Z_GRASP:.3f} release {Z_RELEASE:.3f}")
print("gripper orientation at expert grasp:", np.mean(g_quat, 0).round(3),
      "| ours at start:", np.round(obs["robot0_eef_quat"], 3))

# Override heights with what test_grasp.py showed works
Z_GRASP = 0.945            # grasps succeed at eef z <= 0.955, fail at >= 0.965
Z_RELEASE = Z_GRASP + 0.015
Z_HOVER = Z_GRASP + 0.15
Z_CARRY = Z_GRASP + 0.10
HOLD = 20

def find_events(pinch):
    """Grab/release = the longest stretch where the pinch stays closed."""
    segs, start = [], None
    for i, p in enumerate(pinch):
        if np.isnan(p):
            continue
        if start is None and p < CLOSE:
            start = i
        elif start is not None and p > OPEN:
            segs.append((start, i)); start = None
    return max(segs, key=lambda s: s[1] - s[0]) if segs else (None, None)

def fill_all(x):
    ok = np.isfinite(x)
    return np.interp(np.arange(len(x)), np.where(ok)[0], x[ok])

def at(arr, fr):
    i = int(np.floor(fr)); j = min(i + 1, len(arr) - 1); t = fr - i
    return (1 - t) * arr[i] + t * arr[j]

os.makedirs(OUTDIR, exist_ok=True)
results = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
for k, f in enumerate(sorted(glob.glob("tracks/*_proc.npz"))):
    name = os.path.basename(f)[:-len("_proc.npz")]
    if only and name not in only:
        continue
    group = name.split("_")[0]
    d = np.load(f)
    point, pinch, fps = d["point"], d["pinch"], float(d["fps"])
    gi, ri = find_events(pinch)
    if gi is None:
        results[name] = dict(group=group, events=False, success=False)
        print(f"{name}: no grab/release found, skipped")
        continue

    # 2. Map image path to table: grab point -> bowl, release point -> plate
    env.reset()
    obs = env.set_init_state(init_states[k % len(init_states)])
    if PLACE:   # move the sim bowl to one of the chosen spots (cycles through them)
        pdx, pdy = PLACE_CELLS[place_i % len(PLACE_CELLS)]; place_i += 1
        jn = env.env.objects_dict["akita_black_bowl_1"].joints[0]
        q = env.env.sim.data.get_joint_qpos(jn).copy(); q[0] += pdx; q[1] += pdy
        env.env.sim.data.set_joint_qpos(jn, q); env.env.sim.forward()
    for _ in range(5):
        obs, _, _, _ = env.step([0] * 6 + [-1])
    bowl_z0 = obs[BOWL][2]
    B = complex(*(obs[BOWL][:2] + G_OFF))
    P = complex(*(obs[PLATE][:2] + R_OFF))
    seen = np.where(np.isfinite(point[:, 0]))[0]
    s0, s1 = seen[0], min(seen[-1], ri + int(0.7 * fps))
    zi = fill_all(point[:, 0]) * ASPECT - 1j * fill_all(point[:, 1])  # flip y: image y points down
    a = (P - B) / (zi[ri] - zi[gi])
    w = a * zi + (B - a * zi[gi])          # one rotation + scale + shift for the whole clip
    xy = np.stack([np.clip(w.real, -0.30, 0.25), np.clip(w.imag, -0.30, 0.35)], 1)

    # 3. Height is scripted: stay 6-8cm up while following the hand, then straight down
    #    and straight up at the ends, like the expert robot
    pos = xy[:, 0] + 1j * xy[:, 1]
    dB, dP = np.abs(pos - B), np.abs(pos - P)
    ramp = lambda dist: np.clip(dist / RAMP, 0, 1)
    idx = np.arange(len(xy))
    PRE_G, PRE_R = Z_GRASP + 0.08, Z_RELEASE + 0.06
    z = np.where(idx <= gi, PRE_G + (Z_HOVER - PRE_G) * ramp(dB),
        np.where(idx < ri,
                 np.where(dB < dP, PRE_G + (Z_CARRY - PRE_G) * ramp(dB),
                                   PRE_R + (Z_CARRY - PRE_R) * ramp(dP)),
                 PRE_R + (Z_HOVER - PRE_R) * ramp(dP)))
    target = np.column_stack([xy, z])
    grasp_spot = np.array([B.real, B.imag, Z_GRASP])
    release_spot = np.array([P.real, P.imag, Z_RELEASE])

    # 4. Plan: follow the hand at half speed; at grab / release go straight down,
    #    close / open, straight up
    plan, fr, done_g, done_r = [], float(s0), False, False   # (target, gripper, frame, arrive?)
    step = fps / (20 * SLOW)
    while fr <= s1:
        if not done_g and fr >= gi:
            plan.append((grasp_spot + [0, 0, 0.08], -1, gi, True))
            plan.append((grasp_spot, -1, gi, True))
            plan += [(grasp_spot, 1, gi, False)] * HOLD
            plan.append((grasp_spot + [0, 0, 0.08], 1, gi, True))
            done_g = True
        if not done_r and fr >= ri:
            plan.append((release_spot + [0, 0, 0.06], 1, ri, True))
            plan.append((release_spot, 1, ri, True))
            plan += [(release_spot, -1, ri, False)] * HOLD
            plan.append((release_spot + [0, 0, 0.06], -1, ri, True))
            done_r = True
        plan.append((at(target, fr), 1 if gi <= fr < ri else -1, fr, False))
        fr += step
    plan += [(target[s1] + np.array([0, 0, 0.05]), -1, s1, False)] * 15

    # 5. Execute and record
    rec = {n: [] for n in ["agentview", "wrist", "eef_pos", "eef_quat", "gripper", "actions", "frame"]}
    success, video, grasp_err, bowl_lift = False, [], None, 0.0
    for tgt, grip, fr, arrive in plan:
        for _ in range(ARRIVE_MAX if arrive else 1):
            act = np.zeros(7)
            act[:3] = np.clip((tgt - obs[EEF]) / 0.05, -1, 1)
            act[6] = grip
            rec["agentview"].append(obs["agentview_image"])
            rec["wrist"].append(obs["robot0_eye_in_hand_image"])
            rec["eef_pos"].append(obs[EEF]); rec["eef_quat"].append(obs["robot0_eef_quat"])
            rec["gripper"].append(obs["robot0_gripper_qpos"])
            rec["actions"].append(act); rec["frame"].append(fr)
            video.append(obs["agentview_image"][::-1])
            run = act.copy()   # executed = nudged; recorded label = clean correction (or nudged)
            if NOISE and not arrive and fr not in (gi, ri):
                run[:3] = np.clip(act[:3] + NOISE * rng_noise.standard_normal(3), -1, 1)
            if LABEL_NOISY:
                rec["actions"][-1] = run
            obs, _, done, _ = env.step(run)
            bowl_lift = max(bowl_lift, obs[BOWL][2] - bowl_z0)
            if done:
                success = True
                break
            if arrive and np.linalg.norm(tgt - obs[EEF]) < ARRIVE_TOL:
                break
        if arrive and grip == -1 and grasp_err is None and abs(tgt[2] - Z_GRASP) < 1e-6:
            grasp_err = np.linalg.norm(tgt - obs[EEF])
        if success:
            break
    np.savez_compressed(f"{OUTDIR}/{name}.npz", success=success, group=group,
                        grasp_frame=gi, release_frame=ri,
                        **{n: np.array(v) for n, v in rec.items()})
    imageio.mimsave(f"{OUTDIR}/{name}.mp4", video, fps=20)
    results[name] = dict(group=group, events=True, success=bool(success),
                         steps=len(rec["actions"]), bowl_lift=round(float(bowl_lift), 3))
    ge = f"{grasp_err * 100:.1f}cm" if grasp_err is not None else "n/a"
    print(f"{name}: {'SUCCESS' if success else 'fail'} | {len(rec['actions'])} steps | "
          f"gripper off by {ge} when closing | bowl lifted {bowl_lift * 100:.1f}cm")

json.dump(results, open(RESULTS, "w"), indent=1)
for g in ["new", "good", "gauss6", "gauss8", "gauss10", "gauss12", "gauss24", "motion25", "motion40", "motion60", "hard", "fail"]:
    r = [v for v in results.values() if v["group"] == g]
    if r:
        print(f"{g}: {sum(v['success'] for v in r)}/{len(r)} replays succeeded")
env.close()
