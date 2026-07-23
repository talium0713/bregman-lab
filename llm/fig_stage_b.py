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

SAMPLE_LS = (0, (6, 2))          # dashed (bold, long) = single-sample arm; solid = exact


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


def _smooth(y, w=9):
    """Centered moving average (edge-padded, no lag) to de-noise the eval-acc curves."""
    y = np.asarray(y, dtype=float)
    if len(y) < w:
        return y
    return np.convolve(np.pad(y, w // 2, mode="edge"), np.ones(w) / w, mode="valid")[:len(y)]


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
    arms = set().union(*(set(runs[k]) for k in runs))
    paired = "exact" in arms and "sample" in arms          # both arms → exact-vs-sample; else grad-noise panel
    arm = "sample" if "sample" in arms else "exact"        # the arm for the single-arm (right) panel

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # ── Left: eval-accuracy learning curves. solid = exact, dashed = single-sample ──
    # both arms solid + smoothed; distinguished by opacity (single-sample = crisp reference, exact = faded)
    for k in order:
        for inner, alpha, z in (("exact", 0.5, 2), ("sample", 1.0, 4)):
            h = runs[k].get(inner)
            if not h:
                continue
            axL.plot([r["step"] for r in h], _smooth([r.get("eval_acc", np.nan) for r in h]),
                     color=COLORS[k], lw=3.0 if k == "kl" else 2.0, alpha=alpha,
                     solid_capstyle="round", zorder=(z + 6) if k == "kl" else z)
    axL.axhline(0.5, color="0.7", ls=":", lw=1)
    axL.set_xlabel("training step"); axL.set_ylabel("held-out preference accuracy (smoothed)")
    handles = [Line2D([0], [0], color=COLORS[k], lw=2.6 if k == "kl" else 1.8, label=SHORT[k]) for k in order]
    if paired:
        axL.set_title("learning curves  ·  single-sample (solid) vs exact (faded, α=0.5)")
        handles += [Line2D([0], [0], color="0.3", lw=2.4, alpha=1.0, label="single-sample"),
                    Line2D([0], [0], color="0.3", lw=2.4, alpha=0.5, label="exact")]
    else:
        axL.set_title(f"learning curves  ·  {arm} inner term  (only RKL rises; others stay ≈ 0.5)")
    axL.legend(handles=handles, ncol=2, fontsize=7.5, framealpha=0.9)

    x = np.arange(len(order))
    if paired:
        # ── Right (paired): final accuracy, exact vs single-sample per divergence ──
        w = 0.38
        ex = [_final(runs[k].get("exact", []), "eval_acc") for k in order]
        sa = [_final(runs[k].get("sample", []), "eval_acc") for k in order]
        axR.bar(x - w / 2, ex, w, color=[COLORS[k] for k in order], edgecolor="#222", linewidth=0.5)
        axR.bar(x + w / 2, sa, w, color=[COLORS[k] for k in order], alpha=0.4, hatch="///", edgecolor="#222", linewidth=0.5)
        for xi, e, s in zip(x, ex, sa):
            if np.isfinite(e) and np.isfinite(s) and abs(e - s) > 0.005:
                axR.annotate(f"{e - s:+.02f}", (xi, max(e, s) + 0.01), ha="center", fontsize=7, color="#444")
        axR.axhline(0.5, color="0.7", ls=":", lw=1)
        axR.set_ylabel("final held-out preference accuracy")
        axR.set_title("exact vs single-sample  ·  RKL invariant; non-admissible pay the off-policy noise")
        axR.legend(handles=[Patch(facecolor="0.5", edgecolor="#222", label="exact inner term"),
                            Patch(facecolor="0.5", alpha=0.4, hatch="///", edgecolor="#222", label="single-sample")],
                   fontsize=8, loc="lower right")
    else:
        # ── Right (single-arm): the off-policy inner-term NOISE itself — |∇| per divergence (§3b) ──
        gmed = [float(np.median([r["grad_norm"] for r in runs[k][arm]])) for k in order]
        gmax = [float(max(r["grad_norm"] for r in runs[k][arm])) for k in order]
        axR.bar(x, gmed, 0.6, color=[COLORS[k] for k in order], edgecolor="#222", linewidth=0.5, zorder=3)
        for xi, gm, gx in zip(x, gmed, gmax):               # whisker from median up to max (the spikes)
            axR.plot([xi, xi], [gm, gx], color="#111", lw=0.9, zorder=4)
        axR.scatter(x, gmax, marker="_", s=340, color="#111", zorder=5)
        axR.set_yscale("log")
        axR.set_ylabel(r"gradient noise  $\|\nabla\|$  per step  (bar = median, tick = max)")
        axR.set_title(r"off-policy inner-term noise (§3b): RKL bounded, non-admissible blow up")
    axR.set_xticks(x); axR.set_xticklabels([SHORT[k] for k in order])

    base = {"sft1p7b": "SFT init", "kln1p7b": "kln"}.get(args.tag, args.tag)
    arm_desc = "exact vs single-sample" if paired else f"{arm} inner term"
    fig.suptitle(f"Stage B · token-level f-DPO (Qwen3-1.7B, UltraFeedback) · {base} · {arm_desc}",
                 fontsize=11, y=1.0)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
    print("wrote", out + ".png  ·  divergences:", ", ".join(SHORT[k] for k in order))


if __name__ == "__main__":
    main()
