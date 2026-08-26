"""fig_permissibility_h2h.py — publication figure: RKL's DIRECT head-to-head win rate vs each other
divergence (and the untrained base), same judge gpt-4.1-2025-04-14, 2 games/prompt with position swap,
prompt-level bootstrap 95% CI. Reads the h2h_*.json produced by pairwise_h2h.py.

    python fig_permissibility_h2h.py --dir results/bench/arena_v01/h2h
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, os, glob, argparse

LABEL = {"rkl_bdpo": "FKL", "js_bdpo": "JS", "hel_bdpo": "Hellinger",
         "adiv_bdpo": r"$\alpha$-div", "chi2_bdpo": r"$\chi^2$", "Base_repo": "base"}

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/bench/arena_v01/h2h", help="dir with h2h_*.json")
ap.add_argument("--out", default="results/stageB_divergence_h2h")
args = ap.parse_args()

rows = []
for f in glob.glob(os.path.join(args.dir, "*.json")):
    d = json.load(open(f))
    if d.get("a") != "kl_bdpo":     # only RKL-as-A head-to-heads
        continue
    b = d["b"]; rows.append((LABEL.get(b, b), d["win_rate"], d["ci_lo"], d["ci_hi"]))

rows.sort(key=lambda r: -r[1])      # highest RKL win rate first
labels = [r[0] for r in rows]; wr = [r[1] for r in rows]
lo = [r[1] - r[2] for r in rows]; hi = [r[3] - r[1] for r in rows]
x = range(len(rows))

plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans", "axes.linewidth": 0.9})
fig, ax = plt.subplots(figsize=(6.4, 4.0))
colors = ["#c0392b" if l != "base" else "#2c3e50" for l in labels]
ax.bar(x, wr, yerr=[lo, hi], width=0.62, color=colors, edgecolor="black", linewidth=0.7,
       capsize=4, error_kw=dict(elinewidth=1.1, ecolor="#222"))
ax.axhline(50, ls="--", lw=1.3, color="#555", label="tie (50%)")
for i, (w, h) in enumerate(zip(wr, hi)):
    ax.text(i, w + h + 0.6, f"{w:.1f}", ha="center", va="bottom", fontsize=10)

ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylabel("RKL win rate, head-to-head (%)")
# B1: no in-figure title. Caption → RKL's direct head-to-head win rate vs each other divergence (and the
# base), judge gpt-4.1-2025-04-14, 2 games/prompt (position-swapped), bootstrap 95% CI; all bars >50%.
ax.set_ylim(0, max(wr + [50]) + 8)
ax.legend(loc="lower right", frameon=True, fontsize=9.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", ls=":", lw=0.6, alpha=0.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote {args.out}.png / .pdf  ({len(rows)} opponents: {labels})")
