"""
fig_stage_trajectory.py — held-out preference-accuracy trajectories, token vs newline, at the MATCHED
standard-DPO recipe. Reads results/stageB_{div}_match1p7b.json (token) and
results/stageB_{div}_newline_match1p7b.json (newline).

The point: at token granularity every DPO-inducing divergence climbs to ~0.70 (the batch-512 averaging
tames the single-sample noise); at the coarser newline (sentence) step ONLY the admissible RKL keeps
climbing, while α-div / JS / Hel flat-line near chance — the "collapse" is the disappearance of the
learning curve, not merely a lower endpoint. χ² is the Pipano not-DPO-inducing confound (flat at both).

    python fig_stage_trajectory.py [--dir results] [--out results/stageB_trajectory]
"""
from __future__ import annotations

import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from divergences import KEYS, SHORT, COLORS


def traj(path):
    if not os.path.exists(path):
        return None
    h = json.load(open(path))["history"]
    e = [(x["step"], x["eval_acc"]) for x in h if "eval_acc" in x]
    if not e:
        return None
    s, a = zip(*e)
    return np.array(s), np.array(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/matched")
    ap.add_argument("--out", default="results/stageB_matched_token-vs-sentence_single_curves")
    args = ap.parse_args()

    order = [k for k in KEYS if k != "euc"]
    tok = {k: traj(os.path.join(args.dir, f"stageB_{k}_match1p7b.json")) for k in order}
    nl = {k: traj(os.path.join(args.dir, f"stageB_{k}_newline_match1p7b.json")) for k in order}
    order = [k for k in order if tok.get(k) is not None and nl.get(k) is not None]
    if not order:
        raise SystemExit("no matched token+newline trajectories found in " + args.dir)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, data, title in ((axL, tok, "token-level  ·  every DPO-inducing Ω climbs to ≈ 0.70"),
                            (axR, nl, "newline (sentence)  ·  only RKL keeps climbing")):
        for k in order:
            s, a = data[k]
            lw = 3.2 if k == "kl" else 1.9
            z = 6 if k == "kl" else 3
            ax.plot(s, a, color=COLORS[k], lw=lw, marker="o", ms=4 if k == "kl" else 3,
                    label=SHORT[k], zorder=z, alpha=0.95 if k == "kl" else 0.85)
        ax.axhline(0.5, color="0.6", ls=":", lw=1, zorder=1)
        ax.set_xlabel("training step"); ax.set_title(title, fontsize=10.5)
        ax.set_ylim(0.42, 0.76)
    axL.set_ylabel("held-out preference accuracy")
    axL.legend(ncol=2, fontsize=8.5, framealpha=0.92, loc="lower right")
    fig.suptitle("Stage B · Qwen3-1.7B · standard-DPO recipe (init-noise 0.01) · learning curves: token vs newline",
                 fontsize=11, y=1.0)
    fig.tight_layout(); fig.savefig(args.out + ".png", dpi=150, bbox_inches="tight")
    print("wrote", args.out + ".png")
    for k in order:
        print(f"  {SHORT[k]:5s} token Δ {tok[k][1][-1]-tok[k][1][0]:+.3f}   newline Δ {nl[k][1][-1]-nl[k][1][0]:+.3f}")


if __name__ == "__main__":
    main()
