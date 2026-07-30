"""
fig_stage_mid.py — mid-recipe kln disentangle: is kln's efficacy driven by BATCH or INIT?
Plots kln Δ = acc(+kln) − acc(no-kln) per divergence, in a 2×2 grid (granularity × axis):

  · BATCH axis: light b8 (chosen-SFT) → mid b64 (UC-SFT) → matched b512 (UC-SFT). The clean pair is
    b64 vs b512 (same UC-SFT init, lr linear-scaled ÷8 so only batch=per-step variance differs).
  · INIT  axis: at batch 64, UC-SFT vs base init.

Reads: results/mid/stageB_{div}_{sft|base}_{token|newline}[_kln]_mid1p7b.json,
       results/matched/stageB_{div}_{…}[_kln]_match1p7b.json,
       results/sft1p7b/stageB_{div}_{token|newline}[_kln]_light1p7b.json.

Headline: kln flips help(+)→harm(−) and gets jagged as batch grows (b64 +.03 → b512 −.05 at newline)
⇒ BATCH drives the sign (kln = variance reduction: only useful while per-step noise survives). base
init is nearly kln-inert (|Δ|≈.01) vs UC-SFT (|Δ|≈.04) ⇒ INIT/ref sets magnitude, not sign. RKL ≈0
everywhere (KLN_SHIFT 0).

    python3 fig_stage_mid.py [--out results/stageB_mid_kln_batch-vs-init]
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from divergences import SHORT

DIVS = ["kl", "adiv", "rkl", "js", "hel", "chi2"]        # RKL, α-div, FKL, JS, Hel, χ²


def _acc(path):
    if not os.path.exists(path):
        return np.nan
    h = json.load(open(path))["history"]
    e = [x for x in h if "eval_acc" in x]
    return e[-1]["eval_acc"] if e else np.nan


def light(div, gran, kln):                                # batch 8, chosen-SFT init
    suf = f"{gran}_kln_light1p7b" if kln else f"{gran}_light1p7b"
    return _acc(f"results/sft1p7b/stageB_{div}_{suf}.json")


def mid(init, div, gran, kln):                            # batch 64, {sft|base} init
    suf = f"{init}_{gran}_kln_mid1p7b" if kln else f"{init}_{gran}_mid1p7b"
    return _acc(f"results/mid/stageB_{div}_{suf}.json")


def matched(div, gran, kln):                              # batch 512, UC-SFT init
    if gran == "token":
        suf = "kln_match1p7b" if kln else "match1p7b"
    else:
        suf = "newline_kln_match1p7b" if kln else "newline_match1p7b"
    return _acc(f"results/matched/stageB_{div}_{suf}.json")


def delta(fn):                                            # kln Δ vector over DIVS
    return np.array([fn(d, True) - fn(d, False) for d in DIVS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/stageB_mid_kln_batch-vs-init")
    args = ap.parse_args()

    # (label, color) per series
    C = {"b8": "#8ecae6", "b64s": "#2a9d8f", "b512": "#e76f51", "b64b": "#adb5bd"}
    x = np.arange(len(DIVS))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [1.5, 1]})
    for r, gran in enumerate(["token", "newline"]):
        # ── BATCH axis: b8 → b64-sft → b512 ────────────────────────────────────────────────
        axB = axes[r][0]
        series = [("b8 · chosen-SFT (light)", C["b8"], delta(lambda d, k: light(d, gran, k))),
                  ("b64 · UC-SFT (mid)",      C["b64s"], delta(lambda d, k: mid("sft", d, gran, k))),
                  ("b512 · UC-SFT (matched)", C["b512"], delta(lambda d, k: matched(d, gran, k)))]
        w = 0.26
        for i, (lab, col, dv) in enumerate(series):
            axB.bar(x + (i - 1) * w, dv, w, color=col, edgecolor="#222", lw=0.5, label=lab)
        axB.axhline(0, color="#333", lw=1)
        axB.set_xticks(x); axB.set_xticklabels([SHORT[d] for d in DIVS])
        axB.set_ylabel("kln Δ  = acc(+kln) − acc(no-kln)")
        m8, m64, m512 = (np.nanmean(np.abs(s[2])) for s in series)
        axB.set_title(f"{gran} · BATCH axis · mean|Δ| {m8:.3f}→{m64:.3f}→{m512:.3f} "
                      f"(help→harm; kln = variance reduction, big batch already averaged)", fontsize=9.5)
        axB.legend(fontsize=8, loc="upper left", framealpha=0.95)
        axB.set_ylim(-0.16, 0.18)

        # ── INIT axis: b64 UC-SFT vs base ──────────────────────────────────────────────────
        axI = axes[r][1]
        seriesI = [("b64 · UC-SFT", C["b64s"], delta(lambda d, k: mid("sft", d, gran, k))),
                   ("b64 · base",   C["b64b"], delta(lambda d, k: mid("base", d, gran, k)))]
        wi = 0.38
        for i, (lab, col, dv) in enumerate(seriesI):
            axI.bar(x + (i - 0.5) * wi, dv, wi, color=col, edgecolor="#222", lw=0.5, label=lab)
        axI.axhline(0, color="#333", lw=1)
        axI.set_xticks(x); axI.set_xticklabels([SHORT[d] for d in DIVS])
        ms, mb = (np.nanmean(np.abs(s[2])) for s in seriesI)
        axI.set_title(f"{gran} · INIT axis (batch 64) · mean|Δ| UC-SFT {ms:.3f} vs base {mb:.3f} "
                      f"(base ≈ kln-inert)", fontsize=9.5)
        axI.legend(fontsize=8, loc="upper left", framealpha=0.95)
        axI.set_ylim(-0.16, 0.18)

    fig.suptitle("Stage B · Qwen3-1.7B · mid-recipe kln disentangle: BATCH drives sign (help→harm), "
                 "INIT/ref sets magnitude (base ≈ inert); RKL ≈ 0 (KLN_SHIFT 0)", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150, bbox_inches="tight")
    print("wrote", args.out + ".png")

    # numeric echo
    for gran in ["token", "newline"]:
        print(f"\n{gran}: kln Δ per div")
        for lab, fn in [("b8-light", lambda d, k: light(d, gran, k)),
                        ("b64-sft", lambda d, k: mid("sft", d, gran, k)),
                        ("b512-mat", lambda d, k: matched(d, gran, k)),
                        ("b64-base", lambda d, k: mid("base", d, gran, k))]:
            dv = delta(fn)
            print(f"  {lab:9s}: " + "  ".join(f"{SHORT[d]} {v:+.3f}" for d, v in zip(DIVS, dv))
                  + f"   | mean {np.nanmean(dv):+.3f} |Δ| {np.nanmean(np.abs(dv)):.3f}")


if __name__ == "__main__":
    main()
