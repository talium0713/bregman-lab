"""fig_permissibility_wr.py — publication figure: base->f-DPO divergence Arena-Hard v0.1 win rate (vs
gpt-4-0314, judge gpt-4.1-2025-04-14) with 95% bootstrap CIs. Only RKL beats the untrained base; FKL/chi2
fall below it — the generation-side evidence for the unique off-policy PERMISSIBILITY of KL.

A4: numbers are LOADED from results/bench/arena_v01/divergence_wr.json (no longer hard-coded), so they
regenerate with the judge outputs and stay in sync if the seeds/judge change.

    python fig_permissibility_wr.py [divergence_wr.json]
"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WR_JSON = sys.argv[1] if len(sys.argv) > 1 else "results/bench/arena_v01/divergence_wr.json"
DISPLAY = {"kl": "RKL", "js": "JS", "hel": "Hellinger", "adiv": r"$\alpha$-div",
           "chi2": r"$\chi^2$", "rkl": "FKL"}   # divergence key -> figure label

d = json.load(open(WR_JSON))
DIV = [(DISPLAY.get(x["key"], x["key"]), x["wr"], x["lo"], x["hi"]) for x in d["divergences"]]
BASE = (d["base"]["wr"], d["base"]["lo"], d["base"]["hi"])   # untrained Qwen3-1.7B-Base

labels = [x[0] for x in DIV]
wr     = [x[1] for x in DIV]
lo     = [x[2] for x in DIV]
hi     = [x[3] for x in DIV]
x = range(len(DIV))

plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans", "axes.linewidth": 0.9})
fig, ax = plt.subplots(figsize=(6.4, 4.0))

colors = ["#c0392b" if l == "RKL" else "#7f8c9b" for l in labels]   # RKL highlighted
ax.bar(x, wr, yerr=[lo, hi], width=0.66, color=colors, edgecolor="black", linewidth=0.7,
       capsize=4, error_kw=dict(elinewidth=1.1, ecolor="#222"))

# B9: single reference line — the untrained base (the sign-reversal baseline). SFT line dropped (caption).
ax.axhline(BASE[0], ls="--", lw=1.3, color="#2c3e50", label=f"Qwen3-1.7B-Base ({BASE[0]})")
ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9.5, handlelength=2.2)

for i, (w, h) in enumerate(zip(wr, hi)):
    ax.text(i, w + h + 0.25, f"{w:.1f}", ha="center", va="bottom", fontsize=10,
            fontweight="bold" if labels[i] == "RKL" else "normal")

ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylabel("Arena-Hard v0.1 win rate vs gpt-4-0314 (%)")
# B1: no in-figure title. Caption items → base->f-DPO (Qwen3-1.7B); only RKL beats the base; FKL/chi2 fall
# below it — the generation-side evidence for the unique off-policy permissibility of KL.
ax.set_ylim(0, 17.5)
ax.margins(x=0.02)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", ls=":", lw=0.6, alpha=0.5)
fig.tight_layout()
os.makedirs("results", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"results/stageB_divergence_wr.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote results/stageB_divergence_wr.png / .pdf  (from {WR_JSON}; base={BASE[0]})")
