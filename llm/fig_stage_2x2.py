"""
fig_stage_2x2.py — kln effect at token vs sentence (the 2x2: granularity × {no-kln, kln}) at the MATCHED
recipe. Reads results/matched/stageB_{div}{,_kln,_newline,_newline_kln}_match1p7b.json.

Two panels (token | sentence); in each, no-kln vs +kln bars per divergence with the Δ. The honest
finding: kln's effect is NOT uniform — at token it helps χ² (heavy tail even batch-512 can't average)
and Hel but hurts α-div/FKL; at sentence it is neutral-to-harmful. Only RKL is stable everywhere
(kln shift 0). So kln is not a reliable fix for the coarse-granularity collapse.

    python fig_stage_2x2.py [--dir results/matched] [--out ...]
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from divergences import KEYS, SHORT, COLORS


def _le(h):
    e = [x for x in h if "eval_acc" in x]
    return e[-1].get("eval_acc", np.nan) if e else np.nan


def val(d, div, tag):
    f = os.path.join(d, f"stageB_{div}_{tag}.json")
    return _le(json.load(open(f))["history"]) if os.path.exists(f) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="matched", choices=["matched", "light"])
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    order = [k for k in KEYS if k != "euc"]
    if args.recipe == "light":                          # light tags: stageB_{div}_{token,token_kln,newline,newline_kln}_light1p7b
        args.dir = args.dir or "results/sft1p7b"
        cells = {"token": ("token_light1p7b", "token_kln_light1p7b"),
                 "sentence": ("newline_light1p7b", "newline_kln_light1p7b")}
    else:
        args.dir = args.dir or "results/matched"
        cells = {"token": ("match1p7b", "kln_match1p7b"),
                 "sentence": ("newline_match1p7b", "newline_kln_match1p7b")}
    args.out = args.out or f"results/stageB_{args.recipe}_token-vs-sentence_no-kln-vs-kln"
    x = np.arange(len(order)); w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)
    for ax, (gran, (t0, t1)) in zip(axes, cells.items()):
        no = [val(args.dir, k, t0) for k in order]
        kl = [val(args.dir, k, t1) for k in order]
        ax.bar(x - w / 2, no, w, color=[COLORS[k] for k in order], edgecolor="#222", linewidth=0.5)
        ax.bar(x + w / 2, kl, w, color=[COLORS[k] for k in order], alpha=0.45, hatch="///", edgecolor="#222", linewidth=0.5)
        for xi, a, b in zip(x, no, kl):
            if np.isfinite(a) and np.isfinite(b):
                ax.annotate(f"{b - a:+.02f}", (xi + w / 2, b + 0.006), ha="center", fontsize=7.5,
                            color="#1a7a1a" if b >= a else "#b00")
        ax.axhline(0.5, color="0.6", ls=":", lw=1)
        ax.set_title(f"{gran}-level  ·  no-kln vs +kln"); ax.set_xticks(x); ax.set_xticklabels([SHORT[k] for k in order])
    axes[0].set_ylim(0.40, 0.76); axes[0].set_ylabel("held-out preference accuracy")
    axes[0].legend(handles=[Patch(facecolor="0.5", edgecolor="#222", label="no-kln"),
                            Patch(facecolor="0.5", alpha=0.45, hatch="///", edgecolor="#222", label="+kln")],
                   fontsize=8, loc="upper right")
    fig.suptitle(f"Stage B · Qwen3-1.7B · {args.recipe} recipe · kln effect: token vs sentence",
                 fontsize=11, y=1.0)
    fig.tight_layout(); fig.savefig(args.out + ".png", dpi=150, bbox_inches="tight")
    print("wrote", args.out + ".png")


if __name__ == "__main__":
    main()
