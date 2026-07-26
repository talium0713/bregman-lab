"""
fig_stage_estimator.py — newline baseline vs a newline+{variant} estimator (variant = kln or trl) at
the MATCHED recipe. Reads results/stageB_{div}_newline_match1p7b.json (baseline single-sample) and
results/stageB_{div}_newline_{variant}_match1p7b.json.

Two panels: (left) held-out preference accuracy, (right) worst-case gradient noise ‖∇‖max (log).
The kln story that actually came out: kln removes the 1/u_k gradient tail for χ²/Hel exactly as the
f(0⁺)=1 theory predicts (‖∇‖ drops ×100–3000), but this does NOT recover accuracy — noise removal
≠ learning. Only RKL (Φ≡1, neither noise nor bias) is invariant.

    python fig_stage_estimator.py --variant kln   [--dir results] [--out ...]
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from divergences import KEYS, SHORT, COLORS


def _last_eval(h):
    e = [x for x in h if "eval_acc" in x]
    return e[-1] if e else {}


def load(d, div, variant):
    tag = f"{div}_newline_match1p7b" if variant == "base" else f"{div}_newline_{variant}_match1p7b"
    f = os.path.join(d, f"stageB_{tag}.json")
    if not os.path.exists(f):
        return None
    h = json.load(open(f))["history"]
    return {"eval": _last_eval(h).get("eval_acc", np.nan),
            "gmax": max(x["grad_norm"] for x in h)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="kln", choices=["kln", "trl"])
    ap.add_argument("--dir", default="results/matched")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"results/stageB_newline_{args.variant}"

    order = [k for k in KEYS if k != "euc"]
    base = {k: load(args.dir, k, "base") for k in order}
    var = {k: load(args.dir, k, args.variant) for k in order}
    order = [k for k in order if base.get(k) and var.get(k)]
    if not order:
        raise SystemExit(f"no newline vs newline+{args.variant} pairs found in {args.dir}")

    vlab = {"kln": "newline+kln", "trl": "newline+TRL"}[args.variant]
    x = np.arange(len(order)); w = 0.38
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    be = [base[k]["eval"] for k in order]; ve = [var[k]["eval"] for k in order]
    axL.bar(x - w / 2, be, w, color=[COLORS[k] for k in order], edgecolor="#222", linewidth=0.5)
    axL.bar(x + w / 2, ve, w, color=[COLORS[k] for k in order], alpha=0.4, hatch="///", edgecolor="#222", linewidth=0.5)
    for xi, b, v in zip(x, be, ve):
        if np.isfinite(b) and np.isfinite(v):
            axL.annotate(f"{v - b:+.02f}", (xi + w / 2, v + 0.008), ha="center", fontsize=7.5, color="#b00")
    axL.axhline(0.5, color="0.6", ls=":", lw=1)
    axL.set_ylim(0.38, 0.74); axL.set_ylabel("held-out preference accuracy")
    axL.set_title(f"accuracy · {vlab} does not recover learning")
    axL.legend(handles=[Patch(facecolor="0.5", edgecolor="#222", label="newline"),
                        Patch(facecolor="0.5", alpha=0.4, hatch="///", edgecolor="#222", label=vlab)],
               fontsize=8, loc="upper right")

    bg = [base[k]["gmax"] for k in order]; vg = [var[k]["gmax"] for k in order]
    axR.bar(x - w / 2, bg, w, color=[COLORS[k] for k in order], edgecolor="#222", linewidth=0.5)
    axR.bar(x + w / 2, vg, w, color=[COLORS[k] for k in order], alpha=0.4, hatch="///", edgecolor="#222", linewidth=0.5)
    for xi, b, v in zip(x, bg, vg):
        if b > 0 and v > 0 and (b / v >= 2 or v / b >= 2):
            axR.annotate(f"÷{b / v:.0f}" if b > v else f"×{v / b:.0f}",
                         (xi + w / 2, v * (1.4 if v > b else 0.4)), ha="center", fontsize=7.5, color="#b00")
    axR.set_yscale("log")
    axR.set_ylabel(r"worst-case gradient noise  $\|\nabla\|_{\max}$")
    axR.set_title(f"gradient noise · {vlab} kills the χ²/Hel 1/u tail as predicted")

    for ax in (axL, axR):
        ax.set_xticks(x); ax.set_xticklabels([SHORT[k] for k in order])
    fig.suptitle(f"Stage B · Qwen3-1.7B · newline: single-sample vs {vlab} (matched recipe)", fontsize=11, y=1.0)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
    print("wrote", out + ".png")
    print("  newline :", "  ".join(f"{SHORT[k]} {base[k]['eval']:.3f}" for k in order))
    print(f"  {vlab:11s}:", "  ".join(f"{SHORT[k]} {var[k]['eval']:.3f}" for k in order))


if __name__ == "__main__":
    main()
