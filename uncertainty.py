import glob, os, json
import numpy as np, cv2, mediapipe as mp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASPECT = 1920 / 1080
ZOOM = 0.8
M = (1 - ZOOM) / 2
SCALE = 0.15
S = np.array([ASPECT, 1.0])
COLORS = {"good": "tab:blue", "gauss6": "tab:green", "gauss8": "tab:cyan", "gauss10": "tab:brown",
          "gauss12": "tab:olive", "gauss24": "tab:purple", "motion25": "gold", "motion40": "darkorange",
          "motion60": "black", "hard": "tab:orange", "fail": "tab:red"}
mp_hands = mp.solutions.hands

def landmarks(res, flip=False, zoom=False):
    if not res.multi_hand_landmarks:
        return np.full((21, 3), np.nan)
    a = np.array([[p.x, p.y, p.z] for p in res.multi_hand_landmarks[0].landmark])
    if flip:
        a[:, 0] = 1 - a[:, 0]
    if zoom:
        a[:, :2] = M + ZOOM * a[:, :2]
    return a

def run_views(clip):
    lf, lz = [], []
    cap = cv2.VideoCapture(clip)
    with mp_hands.Hands(max_num_hands=1) as hf, mp_hands.Hands(max_num_hands=1) as hz:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            flip = np.ascontiguousarray(rgb[:, ::-1])
            crop = cv2.resize(rgb[int(M * h):int((1 - M) * h), int(M * w):int((1 - M) * w)], (w, h))
            lf.append(landmarks(hf.process(flip), flip=True))
            lz.append(landmarks(hz.process(crop), zoom=True))
    cap.release()
    return np.array(lf), np.array(lz)

def moving_avg(x, k=9):
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        w = x[max(0, i - k // 2): i + k // 2 + 1]
        if np.isfinite(w).any():
            out[i] = np.nanmean(w)
    return out

results = json.load(open("replay_results.json"))
rows = []
for clip in sorted(glob.glob("my_clips/*/*")):
    group = os.path.basename(os.path.dirname(clip))
    name = group + "_" + os.path.splitext(os.path.basename(clip))[0]
    if not os.path.exists(f"tracks/{name}.npz"):
        print("no track yet, skipping", name)
        continue
    base = np.load(f"tracks/{name}.npz")
    L0, S0 = base["landmarks"], base["score"]
    cache = f"tracks/{name}_unc.npz"
    c = np.load(cache) if os.path.exists(cache) else None
    if c is not None and "lf" in c.files:
        lf, lz = c["lf"], c["lz"]
    else:
        lf, lz = run_views(clip)
    n = min(len(L0), len(lf))
    L0, S0, lf, lz = L0[:n], S0[:n], lf[:n], lz[:n]

    p0 = (L0[:, 4, :2] + L0[:, 8, :2]) / 2
    pf = (lf[:, 4, :2] + lf[:, 8, :2]) / 2
    pz = (lz[:, 4, :2] + lz[:, 8, :2]) / 2
    hand = np.linalg.norm((L0[:, 0, :2] - L0[:, 9, :2]) * S, axis=1)
    seen = np.isfinite(p0[:, 0])
    def dist(p):   # distance to the original view, in hand-sizes; missing view = 1 hand-size
        d = np.linalg.norm((p - p0) * S, axis=1) / hand
        return np.where(seen & np.isnan(p[:, 0]), 1.0, d)
    with np.errstate(all="ignore"):
        disagree = np.nanmean(np.stack([dist(pf), dist(pz)]), 0)
        smooth = np.stack([moving_avg(p0[:, 0]), moving_avg(p0[:, 1])], 1)
        jitter = np.linalg.norm((p0 - smooth) * S, axis=1) / hand
    conf = np.where(seen, np.exp(-(np.nan_to_num(disagree, nan=1) + np.nan_to_num(jitter)) / SCALE), 0.0)
    np.savez(cache, conf=conf, disagree=disagree, jitter=jitter, lf=lf, lz=lz)

    idx = np.where(seen)[0]
    span = slice(idx[0], idx[-1] + 1) if len(idx) else slice(0, 0)
    rows.append(dict(name=name, group=group,
                     conf=float(conf[span].mean()) if len(idx) else 0.0,
                     mp_score=float(S0[span].mean()) if len(idx) else 0.0,
                     success=bool(results.get(name, {}).get("success", False))))
    print(f"{name}: v1 {rows[-1]['conf']:.2f} | mediapipe {rows[-1]['mp_score']:.2f} | "
          f"replay {'OK' if rows[-1]['success'] else 'FAIL'}")

json.dump(rows, open("clip_scores.json", "w"), indent=1)

def auroc(score, label):
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]))

y = np.array([r["success"] for r in rows], dtype=int)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
rng = np.random.default_rng(0)
for ax, key, label in [(axes[0], "conf", "our tracking confidence (v1)"),
                       (axes[1], "mp_score", "MediaPipe detector score")]:
    x = np.array([r[key] for r in rows])
    for g in sorted({r["group"] for r in rows}):
        m = np.array([r["group"] == g for r in rows])
        ax.scatter(x[m], y[m] + rng.uniform(-0.05, 0.05, m.sum()), c=COLORS.get(g, "gray"), label=g, alpha=0.7)
    bins = np.array_split(np.argsort(x), 4)
    ax.plot([x[b].mean() for b in bins], [y[b].mean() for b in bins], "k-o", label="success rate (quartiles)")
    ax.set_xlabel(f"clip {label}"); ax.set_ylabel("replay succeeded")
    ax.set_title(f"{label}  |  AUROC = {auroc(x, y):.2f}")
    ax.legend(loc="center right", fontsize=7)
fig.tight_layout(); fig.savefig("calibration.png", dpi=120)
print(f"\nAUROC v1: {auroc(np.array([r['conf'] for r in rows]), y):.2f} | "
      f"MediaPipe: {auroc(np.array([r['mp_score'] for r in rows]), y):.2f}")
