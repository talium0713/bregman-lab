"""
fig_stage_ladder.py — granularity dose-response (single-sample, matched recipe): token(1) → fixed-8 →
sentence(~newline) → fixed-64, held-out accuracy per divergence. Reads
results/matched/stageB_{div}{,_s8,_newline,_s64}_match1p7b.json.

Headline: RKL is flat across the ladder (granularity-invariant); every non-admissible Ω sits near
0.70 at token and drops to ~0.5–0.6 once the step coarsens (the exact ordering among the coarse steps
is eval-noise, but token vs coarse is a clear cliff). χ² is broken throughout (Pipano confound).

    python fig_stage_ladder.py [--dir results/matched] [--out ...]
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from divergences import KEYS, SHORT, COLORS

RUNGS = [("token", "match1p7b"), ("fixed-8", "s8_match1p7b"),
         ("sentence", "newline_match1p7b"), ("fixed-64", "s64_match1p7b")]


def _le(h):
    e = [x for x in h if "eval_acc" in x]
    return e[-1].get("eval_acc", np.nan) if e else np.nan


def val(d, div, tag):
    f = os.path.join(d, f"stageB_{div}_{tag}.json")
    return _le(json.load(open(f))["history"]) if os.path.exists(f) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/matched")
    ap.add_argument("--out", default="results/stageB_matched_granularity-ladder_single")
    args = ap.parse_args()

    order = [k for k in KEYS if k != "euc"]
    xs = np.arange(len(RUNGS))
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for k in order:
        ys = [val(args.dir, k, tag) for _, tag in RUNGS]
        ax.plot(xs, ys, color=COLORS[k], lw=3.2 if k == "kl" else 1.9, marker="o",
                ms=6 if k == "kl" else 4, label=SHORT[k], zorder=6 if k == "kl" else 3,
                alpha=0.95 if k == "kl" else 0.85)
    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels([f"{name}\n(step≈{s})" for name, s in
                                           zip([r[0] for r in RUNGS], [1, 8, "20", 64])])
    ax.set_ylabel("held-out preference accuracy"); ax.set_xlabel("granularity  (coarser →)")
    ax.set_ylim(0.44, 0.75)
    ax.set_title("granularity dose-response (single-sample) · only RKL stays flat")
    ax.legend(ncol=2, fontsize=9, framealpha=0.9)
    fig.suptitle("Stage B · Qwen3-1.7B · matched recipe · token → fixed-8 → sentence → fixed-64", fontsize=11, y=0.99)
    fig.tight_layout(); fig.savefig(args.out + ".png", dpi=150, bbox_inches="tight")
    print("wrote", args.out + ".png")
    for k in order:
        print(f"  {SHORT[k]:5s}:", "  ".join(f"{n} {val(args.dir,k,t):.3f}" for n, t in RUNGS))


if __name__ == "__main__":
    main()
