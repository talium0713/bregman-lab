"""
fig_stage_seeds.py — seed reproducibility of the settled recipe (batch 64, lr 5e-6, UC-SFT,
single-sample). Held-out accuracy mean ± std across seeds {0 = results/mid, 1,2 = results/seeds},
per divergence, token vs newline. Shows RKL-newline is both highest and tightest (most reproducible).

    python3 fig_stage_seeds.py [--kln] [--out results/stageB_seed-variance]
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from divergences import SHORT

DIVS = ["kl", "adiv", "rkl", "js", "hel", "chi2"]


def _le(f):
    if not os.path.exists(f):
        return None
    d = json.load(open(f)); h = d["history"]; tot = d["args"].get("steps", 1)
    if not h or h[-1]["step"] < 0.9 * tot:      # DROP incomplete/cut-short runs (e.g. ECC-killed at step 1)
        return None
    e = [x for x in h if "eval_acc" in x]
    return e[-1]["eval_acc"] if e else None


def seed_accs(div, gran, kln):
    lab = f"{gran}_kln" if kln else gran
    vals = [_le(f"results/mid/stageB_{div}_sft_{lab}_mid1p7b.json"),          # seed 0
            _le(f"results/seeds/stageB_{div}_{lab}_seed1_1p7b.json"),          # seed 1
            _le(f"results/seeds/stageB_{div}_{lab}_seed2_1p7b.json")]          # seed 2
    return [v for v in vals if v is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kln", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or ("results/stageB_seed-variance" + ("_kln" if args.kln else ""))

    x = np.arange(len(DIVS)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.6))
    print(f"single-sample {'+kln' if args.kln else '(no-kln)'} — held-out acc mean±std (raw seeds)")
    for j, (gran, col) in enumerate([("token", "#457b9d"), ("newline", "#e63946")]):
        means, stds = [], []
        for d in DIVS:
            a = seed_accs(d, gran, args.kln)
            m = np.mean(a) if a else np.nan
            s = np.std(a) if len(a) > 1 else 0.0
            means.append(m); stds.append(s)
            print(f"  {SHORT[d]:5s} {gran:8s}: {m:.3f}±{s:.3f} (n{len(a)})  raw={['%.3f'%v for v in a]}")
        ax.bar(x + (j - 0.5) * w, means, w, yerr=stds, capsize=4, color=col, edgecolor="#222", lw=0.5,
               error_kw=dict(lw=1.2), label=gran)
    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[d] for d in DIVS])
    ax.set_ylabel("held-out preference accuracy (mean ± std over seeds)")
    ax.set_ylim(0.45, 0.80)
    ax.set_title(f"Stage B · Qwen3-1.7B · seed reproducibility (recipe b64/lr5e-6/UC-SFT, single-sample"
                 f"{'+kln' if args.kln else ''}) · seeds {{0,1,2}}", fontsize=10.5)
    ax.legend(title="granularity", fontsize=9)
    fig.tight_layout(); fig.savefig(args.out + ".png", dpi=150, bbox_inches="tight")
    print("wrote", args.out + ".png")


if __name__ == "__main__":
    main()
