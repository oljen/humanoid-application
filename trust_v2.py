"""Trust score v2 -- written BEFORE looking at any motion-blur results.
Question: do the original, mirrored and zoomed views agree on WHEN and WHERE the grab
and the release happen?  (v1 only checked where the hand is, frame by frame.)
  place_error  = mean distance (hand sizes) between each extra view's grab/release point
                 and the original view's grab/release point
  timing_error = largest gap (seconds) between a view's grab/release time and the original's
  v2 = exp(-place_error / 0.25) * exp(-timing_error / 0.5);  v2 = 0 if any view finds no grab/release
  combined = v1 * v2
The scales 0.25 / 0.5 are fixed here; AUROC does not depend on them anyway."""
import json, numpy as np

ASPECT = 1920 / 1080
S = np.array([ASPECT, 1.0])
CLOSE, OPEN = 0.12, 0.30

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

def signals(L):
    point = (L[:, 4, :2] + L[:, 8, :2]) / 2
    with np.errstate(all="ignore"):
        pinch = (np.linalg.norm(L[:, 4, :2] - L[:, 8, :2], axis=1) /
                 np.linalg.norm(L[:, 0, :2] - L[:, 9, :2], axis=1))
    point = np.stack([smooth(fill_short_gaps(point[:, k])) for k in range(2)], 1)
    return point, smooth(fill_short_gaps(pinch))

def events(pinch):
    segs, start = [], None
    for i, p in enumerate(pinch):
        if np.isnan(p):
            continue
        if start is None and p < CLOSE:
            start = i
        elif start is not None and p > OPEN:
            segs.append((start, i)); start = None
    return max(segs, key=lambda s: s[1] - s[0]) if segs else (None, None)

def auroc(score, label):
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]))

rows = json.load(open("clip_scores.json"))
for r in rows:
    t = np.load(f"tracks/{r['name']}.npz")
    c = np.load(f"tracks/{r['name']}_unc.npz")
    n = min(len(t["landmarks"]), len(c["lf"]))
    views = [t["landmarks"][:n], c["lf"][:n], c["lz"][:n]]
    fps = float(t["fps"])
    sig = [signals(v) for v in views]
    ev = [events(p) for _, p in sig]
    if any(e[0] is None for e in ev):
        r["v2"] = 0.0
    else:
        L0 = views[0]
        with np.errstate(all="ignore"):
            hand = np.nanmedian(np.linalg.norm((L0[:, 0, :2] - L0[:, 9, :2]) * S, axis=1))
        (p0, _), (g0, r0) = sig[0], ev[0]
        place, timing = [], []
        for (pt, _), (g, rr) in zip(sig[1:], ev[1:]):
            for a, b in [(g, g0), (rr, r0)]:
                d = np.linalg.norm((pt[a] - p0[b]) * S) / hand
                place.append(d if np.isfinite(d) else 10.0)
                timing.append(abs(a - b) / fps)
        r["v2"] = float(np.exp(-np.mean(place) / 0.25) * np.exp(-max(timing) / 0.5))
    r["combined"] = r["conf"] * r["v2"]
json.dump(rows, open("clip_scores_v2.json", "w"), indent=1)

def report(subset, label):
    y = np.array([x["success"] for x in subset], int)
    line = f"{label:32s} n={len(y):3d} ok={y.sum():3d} |"
    for k, nm in [("mp_score", "MediaPipe"), ("conf", "v1"), ("v2", "v2"), ("combined", "v1*v2")]:
        line += f" {nm} {auroc(np.array([x[k] for x in subset]), y):.2f} |"
    print(line)

design = [x for x in rows if not x["group"].startswith("motion")]
test = [x for x in rows if x["group"].startswith("motion")]
print("AUROC: how well each score predicts replay success (0.5 = coin flip, 1.0 = perfect)")
report(design, "DESIGN set (already looked at)")
for g in ["gauss8", "gauss10"]:
    report([x for x in design if x["group"] == g], f"  within {g}")
if test:
    report(test, "TEST set (motion blur, unseen)")
    for g in sorted({x["group"] for x in test}):
        report([x for x in test if x["group"] == g], f"  within {g}")
