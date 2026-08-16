"""fig_permissibility_wr.py — publication figure: base->f-DPO divergence Arena-Hard v0.1 win rate (vs
gpt-4-0314, judge gpt-4.1-2025-04-14) with 95% bootstrap CIs. Only RKL beats the untrained base; FKL/chi2
fall below it — the generation-side evidence for the unique off-policy PERMISSIBILITY of KL.

    python fig_permissibility_wr.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# base->f-DPO, Arena-Hard v0.1 WR vs gpt-4-0314 (point, -lo, +hi) from the sweep judge (gpt-4.1-2025-04-14)
DIV = [  # label, WR, lo, hi
    ("RKL",       14.8, 1.2, 1.2),
    ("JS",        10.7, 1.2, 1.2),
    ("Hellinger", 10.1, 1.0, 1.3),
    (r"$\alpha$-div", 9.9, 1.0, 0.9),
    (r"$\chi^2$",  7.4, 0.7, 0.9),
    ("FKL",        6.8, 0.8, 0.9),
]
BASE = (11.3, 1.1, 1.0)   # untrained Qwen3-1.7B-Base
SFT  = (6.0, 0.8, 0.8)    # UltraChat-SFT

labels = [d[0] for d in DIV]
wr     = [d[1] for d in DIV]
lo     = [d[2] for d in DIV]
hi     = [d[3] for d in DIV]
x = range(len(DIV))

plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans", "axes.linewidth": 0.9})
fig, ax = plt.subplots(figsize=(6.4, 4.0))

colors = ["#c0392b" if l == "RKL" else "#7f8c9b" for l in labels]   # RKL highlighted
bars = ax.bar(x, wr, yerr=[lo, hi], width=0.66, color=colors, edgecolor="black", linewidth=0.7,
              capsize=4, error_kw=dict(elinewidth=1.1, ecolor="#222"))

# reference lines: untrained base and UltraChat-SFT (labelled via legend to avoid overlapping bars)
ax.axhline(BASE[0], ls="--", lw=1.3, color="#2c3e50", label=f"Qwen3-1.7B-Base ({BASE[0]})")
ax.axhline(SFT[0], ls=":", lw=1.3, color="#95a5a6", label=f"UltraChat-SFT ({SFT[0]})")
ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9.5, handlelength=2.2)

for i, (w, h) in enumerate(zip(wr, hi)):
    ax.text(i, w + h + 0.25, f"{w:.1f}", ha="center", va="bottom", fontsize=10,
            fontweight="bold" if labels[i] == "RKL" else "normal")

ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylabel("Arena-Hard v0.1 win rate vs gpt-4-0314 (%)")
ax.set_title("base $\\to$ $f$-DPO: only RKL beats the base\n(unique off-policy permissibility of KL)", fontsize=12)
ax.set_ylim(0, 17.5)
ax.margins(x=0.02)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", ls=":", lw=0.6, alpha=0.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"results/stageB_divergence_wr.{ext}", dpi=200, bbox_inches="tight")
print("wrote results/stageB_divergence_wr.png / .pdf")
