"""
fig_stage_granularity.py — token vs newline (Step-DPO-style) at the MATCHED standard-DPO recipe.

Reads results/stageB_{div}_match1p7b.json (token) and results/stageB_{div}_newline_match1p7b.json
(newline). Shows the headline: at token granularity every well-behaved divergence is comparable, but
at the coarser newline (sentence) granularity only the admissible RKL survives — its held-out accuracy
is invariant and its gradient noise stays bounded, while every non-admissible Ω collapses toward chance
and its ‖∇‖ blows up by orders of magnitude (the off-policy admissibility thesis, Appendix F).

    python fig_stage_granularity.py [--dir results] [--out results/stageB_granularity]
"""
from __future__ import annotations

import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from divergences import KEYS, SHORT, COLORS


def _last_eval(h):
    e = [x for x in h if "eval_acc" in x]
    return e[-1] if e else {}


def load(d, div, seg):
    tag = f"{div}_newline_match1p7b" if seg == "newline" else f"{div}_match1p7b"
    f = os.path.join(d, f"stageB_{tag}.json")
    if not os.path.exists(f):
        return None
    j = json.load(open(f)); h = j["history"]
    return {"eval": _last_eval(h).get("eval_acc", np.nan),
            "gmax": max(x["grad_norm"] for x in h)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--out", default="results/stageB_granularity")
    args = ap.parse_args()

    order = [k for k in KEYS if k != "euc"]                 # 6 f-divergences
    tok = {k: load(args.dir, k, "token") for k in order}
    nl = {k: load(args.dir, k, "newline") for k in order}
    order = [k for k in order if tok.get(k) and nl.get(k)]
    if not order:
        raise SystemExit("no matched token+newline json pairs found in " + args.dir)

    x = np.arange(len(order)); w = 0.38
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # ── Left: held-out preference accuracy, token vs newline ──
    te = [tok[k]["eval"] for k in order]; ne = [nl[k]["eval"] for k in order]
    axL.bar(x - w / 2, te, w, color=[COLORS[k] for k in order], edgecolor="#222", linewidth=0.5)
    axL.bar(x + w / 2, ne, w, color=[COLORS[k] for k in order], alpha=0.4, hatch="///", edgecolor="#222", linewidth=0.5)
    for xi, t, n in zip(x, te, ne):
        if np.isfinite(t) and np.isfinite(n):
            axL.annotate(f"{n - t:+.02f}", (xi + w / 2, n + 0.008), ha="center", fontsize=7.5, color="#b00")
    axL.axhline(0.5, color="0.6", ls=":", lw=1)
    axL.set_ylim(0.45, 0.76); axL.set_ylabel("held-out preference accuracy")
    axL.set_title("token vs newline  ·  only RKL keeps its accuracy at the coarser step")
    axL.legend(handles=[Patch(facecolor="0.5", edgecolor="#222", label="token-level"),
                        Patch(facecolor="0.5", alpha=0.4, hatch="///", edgecolor="#222", label="newline (sentence)")],
               fontsize=8, loc="upper right")

    # ── Right: worst-case gradient noise ‖∇‖max, token vs newline (log) ──
    tg = [tok[k]["gmax"] for k in order]; ng = [nl[k]["gmax"] for k in order]
    axR.bar(x - w / 2, tg, w, color=[COLORS[k] for k in order], edgecolor="#222", linewidth=0.5)
    axR.bar(x + w / 2, ng, w, color=[COLORS[k] for k in order], alpha=0.4, hatch="///", edgecolor="#222", linewidth=0.5)
    for xi, t, n in zip(x, tg, ng):
        axR.annotate(f"×{n / t:.0f}" if t > 0 and n / t >= 2 else "", (xi + w / 2, n * 1.3), ha="center", fontsize=7.5, color="#b00")
    axR.set_yscale("log")
    axR.set_ylabel(r"worst-case gradient noise  $\|\nabla\|_{\max}$")
    axR.set_title(r"newline amplifies the off-policy noise — except RKL ($\Phi\equiv1$, invariant)")

    for ax in (axL, axR):
        ax.set_xticks(x); ax.set_xticklabels([SHORT[k] for k in order])
    fig.suptitle("Stage B · Qwen3-1.7B · standard-DPO recipe (init-noise 0.01) · granularity: token → newline",
                 fontsize=11, y=1.0)
    fig.tight_layout(); fig.savefig(args.out + ".png", dpi=150, bbox_inches="tight")
    print("wrote", args.out + ".png")
    print("  token  :", "  ".join(f"{SHORT[k]} {tok[k]['eval']:.3f}" for k in order))
    print("  newline:", "  ".join(f"{SHORT[k]} {nl[k]['eval']:.3f}" for k in order))


if __name__ == "__main__":
    main()
