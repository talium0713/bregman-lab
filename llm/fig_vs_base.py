"""fig_vs_base.py — what the training did, and what the normalization did (fL4).

Three measurements, all on the same 500 Arena-Hard v0.1 prompts, same judge, same protocol
(each prompt judged twice with the answer positions swapped):

  top    — each policy against the model it started from, Qwen3-1.7B-Base. 50% means training
           changed nothing. Canonical (dark red) lifts every divergence clear of the line;
           Amari (grey) does not, and for RKL and χ² it lands significantly BELOW it.
  bottom — the two forms head-on, as the bootstrap boxes from fL3.

So the figure separates "did DPO help at all" from "which generator normalization was used" —
and shows the second effect is as large as the first.

Run from llm/:  python fig_vs_base.py  ->  results/stageB_vs_base.{png,pdf}
`python export_for_paper.py` then copies it to figure4overleaf/fL4_vs_base_normcmp.* with a sidecar.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from divergences import COLORS

plt.rcParams.update({"font.family": "serif", "font.size": 11,
                     "mathtext.fontset": "cm", "axes.linewidth": 0.9})

DIVS = ["RKL", "α-div", "FKL", "JS", "Hel", "χ²"]
BAR_KEYS = ["kl", "adiv", "rkl", "js", "hel", "chi2"]   # palette keys: divergences.COLORS (rkl = FKL)
KEY = {"RKL": "kl", "α-div": "adiv", "FKL": "fkl", "JS": "js", "Hel": "hel", "χ²": "chi2"}  # file names

BASE_DIR = "results/bench/arena_v01/h2h_vs_base"
H2H_DIR = "results/bench/arena_v01/h2h_normcmp"
C_AMARI, C_CANON = "#6b7280", "#b0224b"


def vs_base(div, form):
    """(win rate, lo, hi) for one policy against Base_repo — the file already reports it as 'a'."""
    d = json.load(open(f"{BASE_DIR}/h2h_{KEY[div]}_{form}_vs_base_mini.json"))
    return d["win_rate"], d["ci_lo"], d["ci_hi"]


def pct(xs, p):
    i = p * (len(xs) - 1)
    lo = int(i)
    return xs[lo] * (1 - (i - lo)) + xs[min(lo + 1, len(xs) - 1)] * (i - lo)


def h2h_box(div):
    k = KEY[div]
    summ = json.load(open(f"{H2H_DIR}/h2h_{k}_amari_vs_canon_mini2.json"))
    boot = json.load(open(f"{H2H_DIR}/h2h_{k}_amari_vs_canon_mini2_raw.json"))["boot"]
    canon = sorted(100.0 * (1.0 - b) for b in boot)
    return {"med": round(100 - summ["win_rate"], 1), "q1": pct(canon, .25), "q3": pct(canon, .75),
            "whislo": pct(canon, .025), "whishi": pct(canon, .975), "label": div}


fig = plt.figure(figsize=(10.6, 5.6))
gs = fig.add_gridspec(2, 1, height_ratios=(1.25, 1), hspace=0.06,
                      left=0.085, right=0.985, top=0.825, bottom=0.085)
axT = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1], sharex=axT)
x = list(range(len(DIVS)))

# top: both forms against the untrained starting model. The 50% line is "training changed nothing".
axT.axhline(50, color="#999999", ls="--", lw=1.1, zorder=1)
axT.text(-0.55, 50.5, "Base (no change)", color="#777777", fontsize=8.5, va="bottom", ha="left")
for i, d in enumerate(DIVS):
    for form, col, bold in ((("canon"), C_CANON, True), (("amari"), C_AMARI, False)):
        w, lo, hi = vs_base(d, form)
        axT.errorbar(i, w, yerr=[[w - lo], [hi - w]], fmt="o", ms=8, color=col, ecolor=col,
                     elinewidth=1.3, capsize=3, zorder=5)
        axT.text(i - 0.10, w, f"{w:.1f}", color=col, fontsize=9, va="center", ha="right",
                 fontweight="bold" if bold else "normal")

# bottom: the two forms head-on (fL3's boxes)
stats = [h2h_box(d) for d in DIVS]
cols = [COLORS[k] for k in BAR_KEYS]
bp = axB.bxp(stats, positions=x, widths=0.52, patch_artist=True, showfliers=False, zorder=2)
for k, (box, med, col) in enumerate(zip(bp["boxes"], bp["medians"], cols)):
    box.set(facecolor=col, alpha=0.24, edgecolor=col, linewidth=1.4)
    med.set(color=col, linewidth=2.4)
    for art in (bp["whiskers"][2 * k], bp["whiskers"][2 * k + 1], bp["caps"][2 * k], bp["caps"][2 * k + 1]):
        art.set(color=col, linewidth=1.4, alpha=0.9)
for st, col in zip(stats, cols):
    axB.text(DIVS.index(st["label"]), st["whishi"] + 0.4, f"{st['med']:.1f}", color=col, fontsize=9,
             va="bottom", ha="center", fontweight="bold")

axT.text(0.99, 0.97, "Judge: gpt-4.1-mini-2025-04-14", transform=axT.transAxes,
         ha="right", va="top", fontsize=8.5, color="#666666")

axT.set_ylim(37, 68); axT.set_yticks([40, 45, 50, 55, 60, 65])
axB.set_ylim(52, 70); axB.set_yticks([55, 60, 65, 70])
axT.set_xlim(-0.6, len(DIVS) - 0.4)
axT.set_ylabel("Qwen 1.7B\nvs Base\nWin Rate (%)", fontsize=11)
axB.set_ylabel("Qwen 1.7B\nCanonical vs Amari\nWin Rate (%)", fontsize=11)
axB.set_xticks(x); axB.set_xticklabels(DIVS, fontsize=12)
for ax in (axT, axB):
    ax.yaxis.set_label_coords(-0.058, 0.5)
    ax.set_axisbelow(True); ax.grid(axis="y", alpha=0.18)
plt.setp(axT.get_xticklabels(), visible=False)
axT.tick_params(axis="x", length=0)

legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C_CANON, ms=10,
                 label="Canonical (Ours)  $f'(1)=f''(1)$"),
          Line2D([0], [0], marker="o", color="w", markerfacecolor=C_AMARI, ms=10,
                 label="Amari  $f'(1)=0$")]
fig.suptitle("Arena-Hard Results", fontsize=13.5, y=0.985)
fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, fontsize=11,
           frameon=True, framealpha=0.95, edgecolor="#bbbbbb", columnspacing=2.2,
           handletextpad=0.6, borderpad=0.6, labelspacing=0.5)

out = "results/stageB_vs_base"
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
print(f"[saved] {out}.png/.pdf")
