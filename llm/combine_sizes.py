"""
combine_sizes.py — model-size trend from the Stage A ablation. Reads the per-size JSONs and plots
how the off-policy inner-term noise and the off-policyness scale with model size. No GPU / models —
run on a login node or a laptop:
    python combine_sizes.py --out results/stageA_sizes
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from divergences import KEYS, SHORT, COLORS

DEFAULT = [("results/stageA_qwen3_0p6b", 0.6),
           ("results/stageA_qwen3_1p7b", 1.7),
           ("results/stageA_qwen3_4b", 4.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/stageA_sizes")
    ap.add_argument("--metric", choices=["phi_abs_q99", "phi_std"], default="phi_abs_q99",
                    help="which noise measure to trend (q99 is robust; std is outlier-dominated)")
    args = ap.parse_args()

    runs = [(p, s) for p, s in DEFAULT if os.path.exists(p + ".json")]
    if not runs:
        raise SystemExit("no results/stageA_qwen3_*.json found — run the ablation first")
    sizes = [s for _, s in runs]
    summ = [json.load(open(p + ".json")) for p, _ in runs]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.8))
    floor = 1e-4
    xpos = np.arange(len(sizes))                 # categorical (only 3 sizes) — avoids log-tick clutter
    for k in KEYS:
        y = [max(d["divergences"][k].get(args.metric, floor), floor) for d in summ]
        axL.plot(xpos, y, marker="o", color=COLORS[k], lw=3 if k == "kl" else 1.8,
                 zorder=9 if k == "kl" else 3, label=SHORT[k])
    axL.set_yscale("log")
    axL.set_xticks(xpos); axL.set_xticklabels([f"{s}B" for s in sizes])
    axL.set_xlabel("Qwen3 model size (params)")
    axL.set_ylabel(f"single-token inner term  ({args.metric})")
    axL.set_title("off-policy inner-term noise vs model size  (RKL ≡ 0)")
    axL.legend(ncol=2, fontsize=8, framealpha=0.9); axL.grid(True, which="both", alpha=0.15)

    minu = [d["min_u"] for d in summ]
    q001 = [10 ** (d["log_u_quantiles"]["0.001"] / np.log(10)) for d in summ]
    axR.plot(xpos, minu, marker="s", color="#0672F5", label="min u (most off-policy token)")
    axR.plot(xpos, q001, marker="^", color="#008CC4", label="0.1% quantile of u")
    axR.set_yscale("log")
    axR.set_xticks(xpos); axR.set_xticklabels([f"{s}B" for s in sizes])
    axR.set_xlabel("Qwen3 model size (params)")
    axR.set_ylabel(r"$u = \pi_\theta/\pi_{\mathrm{ref}}$  (smaller = more off-policy)")
    axR.set_title("how extreme the off-policy tail gets vs model size")
    axR.legend(fontsize=8, framealpha=0.9); axR.grid(True, which="both", alpha=0.15)

    fig.suptitle("Stage A model-size ablation · π_θ = Qwen3-{size}, π_ref = Qwen3-{size}-Base · UltraFeedback",
                 fontsize=10.5, y=1.02)
    fig.tight_layout(); fig.savefig(args.out + ".png", dpi=150, bbox_inches="tight")
    print("wrote", args.out + ".png  from sizes", sizes)


if __name__ == "__main__":
    main()
