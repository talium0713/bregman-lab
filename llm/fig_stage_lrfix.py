"""
fig_stage_lrfix.py — lr-CONTROLLED batch axis for kln. Earlier the batch comparison confounded lr
with batch (light b8 lr1e-6 · mid b64 lr5e-6 · matched b512 lr5e-5), which made kln look like it
flipped help→harm as batch grew. Holding lr=5e-6 AND UC-SFT init fixed and sweeping batch {8,64,512}
shows the truth: kln is CONSISTENT across batch (mild +, no sign chaos). The jagged/harmful kln at the
original matched run was the 10× higher lr (5e-5), not the batch.

Per granularity, kln Δ = acc(+kln) − acc(no-kln) per divergence, 4 bars:
  b8·lr5e-6, b64·lr5e-6, b512·lr5e-6   (clean batch axis, all UC-SFT — a consistent blue gradient)
  b512·lr5e-5 (matched)                (the lr contrast — red outlier: this is what goes jagged)

Reads: results/lrfix/stageB_{div}_b{8,512}_{gran}[_kln]_lrfix1p7b.json,
       results/mid/stageB_{div}_sft_{gran}[_kln]_mid1p7b.json  (b64, lr5e-6),
       results/matched/stageB_{div}_{…}[_kln]_match1p7b.json   (b512, lr5e-5).

    python3 fig_stage_lrfix.py [--out results/stageB_lrfix_kln_batch-at-fixed-lr]
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


def lrfix(batch, div, gran, kln):                        # lr 5e-6, UC-SFT (b8, b512)
    lab = f"b{batch}_{gran}_kln" if kln else f"b{batch}_{gran}"
    return _acc(f"results/lrfix/stageB_{div}_{lab}_lrfix1p7b.json")


def mid64(div, gran, kln):                               # lr 5e-6, UC-SFT (b64)
    lab = f"sft_{gran}_kln" if kln else f"sft_{gran}"
    return _acc(f"results/mid/stageB_{div}_{lab}_mid1p7b.json")


def matched512(div, gran, kln):                          # lr 5e-5, UC-SFT (b512, the confounded one)
    if gran == "token":
        suf = "kln_match1p7b" if kln else "match1p7b"
    else:
        suf = "newline_kln_match1p7b" if kln else "newline_match1p7b"
    return _acc(f"results/matched/stageB_{div}_{suf}.json")


def delta(fn, gran):
    return np.array([fn(d, gran, True) - fn(d, gran, False) for d in DIVS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/stageB_lrfix_kln_batch-at-fixed-lr")
    args = ap.parse_args()

    x = np.arange(len(DIVS))
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    for r, gran in enumerate(["token", "newline"]):
        ax = axes[r]
        series = [
            ("b8 · lr5e-6",           "#a8dadc", delta(lambda d, g, k: lrfix(8, d, g, k), gran)),
            ("b64 · lr5e-6",          "#457b9d", delta(lambda d, g, k: mid64(d, g, k), gran)),
            ("b512 · lr5e-6",         "#1d3557", delta(lambda d, g, k: lrfix(512, d, g, k), gran)),
            ("b512 · lr5e-5 (matched)", "#e63946", delta(lambda d, g, k: matched512(d, g, k), gran)),
        ]
        w = 0.2
        for i, (lab, col, dv) in enumerate(series):
            mad = np.nanmean(np.abs(dv))
            ax.bar(x + (i - 1.5) * w, dv, w, color=col, edgecolor="#222", lw=0.5,
                   label=f"{lab}  (|Δ| {mad:.3f})",
                   hatch="xx" if i == 3 else None)
        ax.axhline(0, color="#333", lw=1)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[d] for d in DIVS])
        ax.set_ylabel("kln Δ = acc(+kln) − acc(no-kln)")
        ax.set_ylim(-0.16, 0.18)
        ax.set_title(f"{gran} · at lr5e-6 kln is CONSISTENT across batch (blue, mild +); "
                     f"lr5e-5 matched-b512 (red) is what goes jagged → lr, not batch", fontsize=9.5)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.95, ncol=2)

    fig.suptitle("Stage B · Qwen3-1.7B · lr-controlled batch axis for kln: the b64↔b512 inconsistency "
                 "was LR (5e-5), not batch — at fixed lr5e-6, batch {8,64,512} is consistent", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150, bbox_inches="tight")
    print("wrote", args.out + ".png")

    for gran in ["token", "newline"]:
        print(f"\n{gran}: kln Δ per div")
        for lab, fn in [("b8@5e-6", lambda d, g, k: lrfix(8, d, g, k)),
                        ("b64@5e-6", lambda d, g, k: mid64(d, g, k)),
                        ("b512@5e-6", lambda d, g, k: lrfix(512, d, g, k)),
                        ("b512@5e-5", lambda d, g, k: matched512(d, g, k))]:
            dv = delta(fn, gran)
            print(f"  {lab:10s}: " + "  ".join(f"{SHORT[d]} {v:+.3f}" for d, v in zip(DIVS, dv))
                  + f"  | mean {np.nanmean(dv):+.3f} |Δ| {np.nanmean(np.abs(dv)):.3f}")


if __name__ == "__main__":
    main()
