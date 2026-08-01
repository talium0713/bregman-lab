"""
fig_stage_batch_lr_grid.py — the full batch×lr grid for kln, resolving batch vs lr. kln "jaggedness"
= mean over divergences of |kln Δ| (Δ = acc(+kln) − acc(no-kln)); signed mean Δ shows help(+)/harm(−).

  lr 5e-6 row: b8 (lrfix) · b64 (mid) · b512 (lrfix)      → kln mild & consistent (|Δ| ~.02–.04) at every batch
  lr 5e-5 row: b8 (hilr)  · b64 (hilr) · b512 (matched)    → kln JAGGED (|Δ| ~.05–.08) at every batch

Conclusion: kln's jagged/harmful behavior is a HIGH-LR effect present at EVERY batch, absent at every
batch when lr is low → LR drives it, NOT batch. (Mechanism: high lr × kln's f'(1)=1 renormalization,
which shifts the per-step objective by up to KLN_SHIFT, makes large distorting steps.)

Reads results/{lrfix,mid,hilr,matched}. Missing cells (e.g. an ECC-failed task) are nan-skipped.

    python3 fig_stage_batch_lr_grid.py [--out results/stageB_kln_batch-x-lr-grid]
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from divergences import SHORT

DIVS = ["kl", "adiv", "rkl", "js", "hel", "chi2"]


def _acc(path):
    if not os.path.exists(path):
        return np.nan
    h = json.load(open(path))["history"]
    e = [x for x in h if "eval_acc" in x]
    return e[-1]["eval_acc"] if e else np.nan


def acc(recipe, batch, div, gran, kln):
    """held-out acc for a (recipe, batch) cell."""
    if recipe == "lrfix":                                # b8, b512 @ lr5e-6
        lab = f"b{batch}_{gran}_kln" if kln else f"b{batch}_{gran}"
        return _acc(f"results/lrfix/stageB_{div}_{lab}_lrfix1p7b.json")
    if recipe == "mid":                                  # b64 @ lr5e-6 (UC-SFT)
        lab = f"sft_{gran}_kln" if kln else f"sft_{gran}"
        return _acc(f"results/mid/stageB_{div}_{lab}_mid1p7b.json")
    if recipe == "hilr":                                 # b8, b64 @ lr5e-5
        lab = f"b{batch}_{gran}_kln" if kln else f"b{batch}_{gran}"
        return _acc(f"results/hilr/stageB_{div}_{lab}_hilr1p7b.json")
    if recipe == "matched":                              # b512 @ lr5e-5
        suf = ("kln_match1p7b" if kln else "match1p7b") if gran == "token" \
              else ("newline_kln_match1p7b" if kln else "newline_match1p7b")
        return _acc(f"results/matched/stageB_{div}_{suf}.json")
    return np.nan


# (lr label, {batch: recipe})
ROWS = [("lr 5e-6", "#1d6fb8", {8: "lrfix", 64: "mid", 512: "lrfix"}),
        ("lr 5e-5", "#e63946", {8: "hilr", 64: "hilr", 512: "matched"})]
BATCHES = [8, 64, 512]


def cell_stats(recipe, batch, gran):
    """mean|Δ| and mean Δ over divergences for a (recipe,batch,gran) cell (nan-skip)."""
    d = np.array([acc(recipe, batch, dv, gran, True) - acc(recipe, batch, dv, gran, False) for dv in DIVS])
    return np.nanmean(np.abs(d)), np.nanmean(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/stageB_kln_batch-x-lr-grid")
    args = ap.parse_args()

    x = np.arange(len(BATCHES)); w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, gran in zip(axes, ["token", "newline"]):
        for j, (lrlab, col, rec_of) in enumerate(ROWS):
            mad = [cell_stats(rec_of[b], b, gran)[0] for b in BATCHES]
            sgn = [cell_stats(rec_of[b], b, gran)[1] for b in BATCHES]
            bars = ax.bar(x + (j - 0.5) * w, mad, w, color=col, edgecolor="#222", lw=0.5,
                          label=f"{lrlab}  (mean|Δ| {np.mean(mad):.3f})")
            for xi, m, s in zip(x, mad, sgn):
                ax.annotate(f"{s:+.03f}", (xi + (j - 0.5) * w, m + 0.002), ha="center", fontsize=7.5,
                            color="#1a7a1a" if s >= 0 else "#b00")
        ax.set_xticks(x); ax.set_xticklabels([f"b{b}" for b in BATCHES])
        ax.set_xlabel("batch (per-step variance ↓ →)")
        ax.set_ylabel("kln jaggedness = mean$_Ω$|Δ|   (Δ = acc(+kln) − acc(no-kln))")
        ax.set_ylim(0, 0.095)
        ax.set_title(f"{gran} · lr5e-5 jagged at EVERY batch, lr5e-6 mild at every batch "
                     f"(text = signed mean Δ)", fontsize=9.5)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.95)

    fig.suptitle("Stage B · Qwen3-1.7B · batch×lr grid: kln's jaggedness is set by LR, not batch "
                 "— lr5e-5 high & lr5e-6 low at all of b{8,64,512}", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150, bbox_inches="tight")
    print("wrote", args.out + ".png")

    # ── PER-DIVERGENCE figure: one subplot per Ω; kln Δ vs batch, one line per (lr × granularity) ──
    def kd(rec, b, div, gran):
        return acc(rec, b, div, gran, True) - acc(rec, b, div, gran, False)
    STYLE = [("token", "-", "o"), ("newline", "--", "s")]
    figp, axp = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for i, div in enumerate(DIVS):
        ax = axp[i // 3][i % 3]
        for lrlab, col, rec_of in ROWS:
            for gran, ls, mk in STYLE:
                ys = [kd(rec_of[b], b, div, gran) for b in BATCHES]
                ax.plot(np.arange(len(BATCHES)), ys, color=col, ls=ls, marker=mk, ms=6, lw=1.8,
                        label=f"{lrlab} · {gran}")
        ax.axhline(0, color="#333", lw=1)
        ax.set_title(SHORT[div], fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(len(BATCHES))); ax.set_xticklabels([f"b{b}" for b in BATCHES])
        ax.grid(axis="y", ls=":", alpha=0.4)
    for r in range(2):
        axp[r][0].set_ylabel("kln Δ = acc(+kln) − acc(no-kln)")
    axp[0][0].legend(fontsize=7.5, loc="best", framealpha=0.95, ncol=2)
    figp.suptitle("Stage B · Qwen3-1.7B · kln Δ per divergence across the batch×lr grid "
                  "(blue lr5e-6 / red lr5e-5 · solid token / dashed newline) — lr5e-5 drives the swings, batch-flat",
                  fontsize=11, y=1.0)
    figp.tight_layout()
    figp.savefig(args.out + "_per-div.png", dpi=150, bbox_inches="tight")
    print("wrote", args.out + "_per-div.png")

    for gran in ["token", "newline"]:
        print(f"\n{gran}: mean|Δ| (signed mean Δ) per cell")
        for lrlab, _, rec_of in ROWS:
            cells = [cell_stats(rec_of[b], b, gran) for b in BATCHES]
            print(f"  {lrlab}: " + "  ".join(f"b{b} {c[0]:.3f}({c[1]:+.3f})" for b, c in zip(BATCHES, cells)))


if __name__ == "__main__":
    main()
