"""
fig_stage_estimators.py — three newline estimators side by side at the MATCHED recipe:
  newline (single-sample inner Φ) · newline+kln (normalized inner) · newline+TRL (no inner term).
Reads results/matched/stageB_{div}_newline{,_kln,_trl}_match1p7b.json.

The estimator axis DECOMPOSES the newline collapse:
  · TRL (drop the inner term) rescues divergences whose CHOSEN term f' is benign — JS (bounded f',
    .524→.667, ‖∇‖ 1.9e6→13) and α-div (√u tail, .512→.579): the inner term was their whole problem.
  · TRL leaves divergences whose CHOSEN term is heavy-tailed still broken — FKL (f'=-1/u, .631→.452)
    and χ² (Pipano). kln removes the inner tail but distorts the objective → no recovery.
  · RKL is robust across all three (Φ≡1: no inner noise, mild f'); its TRL +βK length bias is minor
    (K_w≈K_l cancels for same-prompt pairs).

    python fig_stage_estimators.py [--dir results/matched] [--out results/stageB_estimators]
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from divergences import KEYS, SHORT, COLORS

VARIANTS = [("newline", "", 1.0, None), ("+kln", "_kln", 0.55, "///"), ("+TRL", "_trl", 0.5, "xxx")]


def _last_eval(h):
    e = [x for x in h if "eval_acc" in x]
    return e[-1] if e else {}


def load(d, div, suf):
    f = os.path.join(d, f"stageB_{div}_newline{suf}_match1p7b.json")
    if not os.path.exists(f):
        return None
    h = json.load(open(f))["history"]
    return {"eval": _last_eval(h).get("eval_acc", np.nan), "gmax": max(x["grad_norm"] for x in h)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/matched")
    ap.add_argument("--out", default="results/stageB_matched_sentence_single-vs-kln-vs-trl")
    args = ap.parse_args()

    order = [k for k in KEYS if k != "euc"]
    data = {lab: {k: load(args.dir, k, suf) for k in order} for lab, suf, _, _ in VARIANTS}
    order = [k for k in order if all(data[lab].get(k) for lab, *_ in VARIANTS)]
    if not order:
        raise SystemExit("need newline + newline_kln + newline_trl jsons in " + args.dir)

    x = np.arange(len(order)); w = 0.27
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.2))
    offs = [-w, 0, w]
    for (lab, suf, alpha, hatch), off in zip(VARIANTS, offs):
        ev = [data[lab][k]["eval"] for k in order]
        gm = [data[lab][k]["gmax"] for k in order]
        axL.bar(x + off, ev, w, color=[COLORS[k] for k in order], alpha=alpha, hatch=hatch, edgecolor="#222", linewidth=0.5)
        axR.bar(x + off, gm, w, color=[COLORS[k] for k in order], alpha=alpha, hatch=hatch, edgecolor="#222", linewidth=0.5)

    axL.axhline(0.5, color="0.6", ls=":", lw=1); axL.set_ylim(0.38, 0.74)
    axL.set_ylabel("held-out preference accuracy")
    axL.set_title("accuracy · TRL rescues benign-f' divergences (JS, α-div); kln does not")
    axR.set_yscale("log"); axR.set_ylabel(r"worst-case gradient noise  $\|\nabla\|_{\max}$")
    axR.set_title(r"gradient noise · dropping the inner term (TRL) removes the inner tail")
    leg = [Patch(facecolor="0.5", alpha=a, hatch=h, edgecolor="#222", label=lab) for lab, _, a, h in VARIANTS]
    axL.legend(handles=leg, fontsize=8, loc="upper right", ncol=3)
    for ax in (axL, axR):
        ax.set_xticks(x); ax.set_xticklabels([SHORT[k] for k in order])
    fig.suptitle("Stage B · Qwen3-1.7B · newline estimators: single-sample vs kln vs TRL (matched recipe)", fontsize=11, y=1.0)
    fig.tight_layout(); fig.savefig(args.out + ".png", dpi=150, bbox_inches="tight")
    print("wrote", args.out + ".png")
    for lab, *_ in VARIANTS:
        print(f"  {lab:8s}:", "  ".join(f"{SHORT[k]} {data[lab][k]['eval']:.3f}" for k in order))


if __name__ == "__main__":
    main()
