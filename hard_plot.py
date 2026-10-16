import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SH = [-0.10, -0.05, 0.0, 0.05, 0.10]
data = {"uniform": json.load(open("hard_uniform.json")), "expert": json.load(open("hard_expert.json"))}
titles = {"uniform": "trained on my phone videos", "expert": "trained on robot demos"}

# --- 1. success map ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, (v, rows) in zip(axes, data.items()):
    M = np.full((5, 5), np.nan)
    for i, dy in enumerate(SH[::-1]):
        for j, dx in enumerate(SH):
            r = [x["success"] for x in rows if round(x["dx"], 2) == dx and round(x["dy"], 2) == dy]
            if r:
                M[i, j] = np.mean(r)
    ax.imshow(np.ma.masked_invalid(M), cmap="RdYlGn", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, "skip" if np.isnan(M[i, j]) else f"{M[i, j] * 100:.0f}%", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{d * 100:+.0f}" for d in SH])
    ax.set_yticks(range(5)); ax.set_yticklabels([f"{d * 100:+.0f}" for d in SH[::-1]])
    ax.set_xlabel("bowl moved in x (cm)"); ax.set_ylabel("bowl moved in y (cm)")
    ax.set_title(titles[v])
fig.tight_layout(); fig.savefig("hardtest_heatmap.png", dpi=130)

# --- 2. do the 7 policies disagree when it's going to fail? ---
def auroc(s, l):
    p, n = s[l == 1], s[l == 0]
    return np.mean([(a > b) + 0.5 * (a == b) for a in p for b in n]) if len(p) and len(n) else float("nan")

fig, ax = plt.subplots(figsize=(6.5, 4))
rng = np.random.default_rng(0)
for v, c in [("uniform", "tab:blue"), ("expert", "tab:orange")]:
    rows = data[v]
    y = np.array([x["success"] for x in rows], int)
    u = np.array([x["unc_first"] for x in rows])
    print(f"{v:8s} | {len(y)} episodes, {y.mean() * 100:.0f}% success | "
          f"disagreement predicts failure: AUROC {auroc(-u, y):.2f} (first few steps), "
          f"{auroc(-np.array([x['unc_mean'] for x in rows]), y):.2f} (whole episode)")
    bins = np.array_split(np.argsort(u), 4)
    print("   success rate from least to most disagreement (quarters):",
          " ".join(f"{y[b].mean() * 100:.0f}%" for b in bins))
    ax.scatter(u, y + rng.uniform(-0.05, 0.05, len(y)), alpha=0.5, c=c, label=titles[v])
    ax.plot([u[b].mean() for b in bins], [y[b].mean() for b in bins], "-o", c=c)
ax.set_xlabel("disagreement between the 7 policies (first few steps)")
ax.set_ylabel("succeeded"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("hardtest_disagreement.png", dpi=130)
print("saved hardtest_heatmap.png and hardtest_disagreement.png")
