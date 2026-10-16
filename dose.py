import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = json.load(open("clip_scores.json"))
ev = json.load(open("events.json"))
levels = [("good", 0), ("gauss6", 6), ("gauss8", 8), ("gauss10", 10), ("gauss12", 12), ("gauss24", 24)]
xs, conf, found, succ = [], [], [], []
for g, s in levels:
    r = [x for x in rows if x["group"] == g]
    xs.append(s)
    conf.append(np.mean([x["conf"] for x in r]))
    succ.append(np.mean([x["success"] for x in r]))
    found.append(np.mean([v["ok"] for v in ev.values() if v["group"] == g]))
    print(f"{g:8s} blur {s:2d} | trust {conf[-1]:.2f} | grab/release found {found[-1]*100:3.0f}% | replay success {succ[-1]*100:3.0f}%")
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(xs, conf, "o-", label="mean trust score")
ax.plot(xs, found, "s--", label="grab/release found")
ax.plot(xs, succ, "^-", label="robot replay succeeded")
ax.set_xlabel("Gaussian blur added (sigma, pixels at 1080p)"); ax.set_ylabel("fraction / score")
ax.set_title("Same 39 clips, increasing blur"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("blur_dose_response.png", dpi=130)

def auroc(s, l):
    p, n = s[l == 1], s[l == 0]
    return np.mean([(a > b) + 0.5 * (a == b) for a in p for b in n]) if len(p) and len(n) else float("nan")
for g in ["gauss8", "gauss10"]:
    r = [x for x in rows if x["group"] == g]
    s = np.array([x["conf"] for x in r]); m = np.array([x["mp_score"] for x in r]); y = np.array([x["success"] for x in r], int)
    print(f"Within {g} only ({y.sum()} ok / {len(y)}): AUROC ours {auroc(s, y):.2f} | MediaPipe {auroc(m, y):.2f}")
print("Saved blur_dose_response.png")
