import glob, json, os
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLOSE, OPEN = 0.12, 0.30   # pinch below CLOSE = grabbed, above OPEN = released

def fill_short_gaps(x, max_gap=5):
    x = x.copy(); isn = np.isnan(x); ok = np.where(~isn)[0]
    if len(ok) < 2:
        return x
    filled = np.interp(np.arange(len(x)), ok, x[ok])
    i = 0
    while i < len(x):
        if isn[i]:
            j = i
            while j < len(x) and isn[j]:
                j += 1
            if i > 0 and j < len(x) and j - i <= max_gap:
                x[i:j] = filled[i:j]
            i = j
        else:
            i += 1
    return x

def smooth(x, k=5):
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        w = x[max(0, i - k // 2): i + k // 2 + 1]
        if np.isfinite(w).any():
            out[i] = np.nanmean(w)
    out[np.isnan(x)] = np.nan
    return out

events = {}
for f in sorted(glob.glob("tracks/*.npz")):
    name = os.path.basename(f)[:-4]
    if name.endswith(("_proc", "_unc")):
        continue
    d = np.load(f)
    lms, fps = d["landmarks"], float(d["fps"])
    point = (lms[:, 4, :2] + lms[:, 8, :2]) / 2          # between thumb and index tips
    pinch = (np.linalg.norm(lms[:, 4, :2] - lms[:, 8, :2], axis=1) /
             np.linalg.norm(lms[:, 0, :2] - lms[:, 9, :2], axis=1))
    point = np.stack([smooth(fill_short_gaps(point[:, k])) for k in range(2)], 1)
    pinch = smooth(fill_short_gaps(pinch))

    grasp = release = None
    for i, p in enumerate(pinch):
        if np.isnan(p):
            continue
        if grasp is None and p < CLOSE:
            grasp = i
        elif grasp is not None and p > OPEN:
            release = i
            break

    seen = np.where(np.isfinite(lms[:, 0, 0]))[0]
    coverage = float(np.isfinite(lms[seen[0]:seen[-1] + 1, 0, 0]).mean()) if len(seen) else 0.0
    ok = grasp is not None and release is not None
    events[name] = dict(group=name.split("_")[0], fps=fps, grasp=grasp, release=release,
                        ok=ok, coverage=round(coverage, 3),
                        grasp_xy=point[grasp].tolist() if ok else None,
                        release_xy=point[release].tolist() if ok else None)
    np.savez(f"tracks/{name}_proc.npz", point=point, pinch=pinch, fps=fps)

json.dump(events, open("events.json", "w"), indent=1)

for g in ["good", "hard", "fail"]:
    e = [v for v in events.values() if v["group"] == g]
    n_ok = sum(v["ok"] for v in e)
    cov = np.mean([v["coverage"] for v in e]) if e else 0
    print(f"{g}: grab+release found in {n_ok}/{len(e)} clips | mean tracking coverage {cov:.2f}")
for k, v in events.items():
    if not v["ok"]:
        print("  no clean grab/release:", k)

# Overview figure: every good and hard path drawn over an empty-table frame
first = sorted(glob.glob("my_clips/good/*"))[0]
ok_, bg = cv2.VideoCapture(first).read()
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, g, col in [(axes[0], "good", "tab:blue"), (axes[1], "hard", "tab:orange")]:
    ax.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB), extent=[0, 1, 1, 0])
    for k, v in events.items():
        if v["group"] != g:
            continue
        p = np.load(f"tracks/{k}_proc.npz")["point"]
        ax.plot(p[:, 0], p[:, 1], color=col, alpha=0.5, lw=1)
        if v["ok"]:
            ax.plot(*v["grasp_xy"], "o", color="lime", ms=5)
            ax.plot(*v["release_xy"], "x", color="red", ms=6)
    ax.set_title(f"{g} clips  (green = grab, red = release)")
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.axis("off")
fig.tight_layout(); fig.savefig("overview_paths.png", dpi=120)
print("Saved events.json and overview_paths.png")
