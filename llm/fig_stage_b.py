"""
fig_stage_b.py — visualize the Stage B (token-level f-DPO) sweep: exact vs single-sample inner term.

Reads results/stageB_{div}_{inner}_{tag}.json (written by stage_b_train.py), one per (divergence,
arm), and shows the headline: RKL is invariant across arms (Φ≡1, single-sample == exact), while every
non-admissible divergence degrades under the noisy single-sample inner term and recovers under the
exact vocab sum — the off-policy admissibility thesis at training scale.

No GPU / models — run on a login node or your Mac after fetching results:
    python fig_stage_b.py [--dir results] [--tag 1p7b]
"""
from __future__ import annotations

import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from divergences import KEYS, SHORT, COLORS

SAMPLE_LS = (0, (4, 2))          # dashed = single-sample arm; solid = exact


def load(d, tag):
    """runs[div][inner] = history (list of per-log-step dicts)."""
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, f"stageB_*_{tag}.json"))):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        a, hist = j.get("args", {}), j.get("history", [])
        div, inner = a.get("div"), a.get("inner")
        if div and inner and hist:
            runs.setdefault(div, {})[inner] = hist
    return runs


def _final(hist, key, default=np.nan):
    return hist[-1].get(key, default) if hist else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--tag", default="1p7b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = load(args.dir, args.tag)
    if not runs:
        raise SystemExit(f"no stageB_*_{args.tag}.json found in {args.dir}/")
    out = args.out or os.path.join(args.dir, f"stageB_{args.tag}")
    order = [k for k in KEYS if k in runs]                 # canonical divergence order

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # ── Left: eval-accuracy learning curves. solid = exact, dashed = single-sample ──
    for k in order:
        for inner, ls in (("exact", "-"), ("sample", SAMPLE_LS)):
            h = runs[k].get(inner)
            if not h:
                continue
            axL.plot([r["step"] for r in h], [r.get("eval_acc", np.nan) for r in h],
                     ls=ls, color=COLORS[k], lw=2.6 if k == "kl" else 1.6,
                     zorder=9 if k == "kl" else 3)
    axL.axhline(0.5, color="0.7", ls=":", lw=1)
    axL.set_xlabel("training step"); axL.set_ylabel("held-out preference accuracy")
    axL.set_title("learning curves  ·  solid = exact inner term, dashed = single-sample")
    handles = [Line2D([0], [0], color=COLORS[k], lw=2.6 if k == "kl" else 1.8, label=SHORT[k]) for k in order]
    handles += [Line2D([0], [0], color="0.35", ls="-", label="exact"),
                Line2D([0], [0], color="0.35", ls=SAMPLE_LS, label="single-sample")]
    axL.legend(handles=handles, ncol=2, fontsize=7.5, framealpha=0.9)

    # ── Right: final accuracy, exact vs single-sample per divergence (the headline gap) ──
    x = np.arange(len(order)); w = 0.38
    ex = [_final(runs[k].get("exact", []), "eval_acc") for k in order]
    sa = [_final(runs[k].get("sample", []), "eval_acc") for k in order]
    axR.bar(x - w / 2, ex, w, color=[COLORS[k] for k in order], edgecolor="#222", linewidth=0.5)
    axR.bar(x + w / 2, sa, w, color=[COLORS[k] for k in order], alpha=0.4, hatch="///", edgecolor="#222", linewidth=0.5)
    for xi, e, s in zip(x, ex, sa):                        # annotate the exact−sample gap (noise cost)
        if np.isfinite(e) and np.isfinite(s) and abs(e - s) > 0.005:
            axR.annotate(f"{e - s:+.02f}", (xi, max(e, s) + 0.01), ha="center", fontsize=7, color="#444")
    axR.axhline(0.5, color="0.7", ls=":", lw=1)
    axR.set_xticks(x); axR.set_xticklabels([SHORT[k] for k in order])
    axR.set_ylabel("final held-out preference accuracy")
    axR.set_title("exact vs single-sample  ·  RKL invariant; non-admissible pay the off-policy noise")
    axR.legend(handles=[Patch(facecolor="0.5", edgecolor="#222", label="exact inner term"),
                        Patch(facecolor="0.5", alpha=0.4, hatch="///", edgecolor="#222", label="single-sample")],
               fontsize=8, loc="lower right")

    fig.suptitle("Stage B · token-level f-DPO (Qwen3-1.7B, UltraFeedback) · exact vs single-sample inner term",
                 fontsize=11, y=1.0)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
    print("wrote", out + ".png  ·  divergences:", ", ".join(SHORT[k] for k in order))


if __name__ == "__main__":
    main()
