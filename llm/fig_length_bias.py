"""
fig_length_bias.py — LOCAL analysis of the length-bias probe (Stage B, Goal 1). Reads the per-pair
CSVs produced on the cluster by eval_length_bias.py
(results/lenbias/stageB_{div}_newline_match1p7b_perpair.csv) and shows that TRL's RKL/FKL held-out
accuracy change is MANUFACTURED by the length gap ΔK = K_w − K_l, via the exact identity

    margin_TRL = margin_single + β·f'(1)·ΔK        (f'(1) = 1 − KLN_SHIFT: kl=+1 RKL, rkl=−1 FKL)

No models, no torch inference — pure post-processing of the dumped single-sample margins + step counts.

Per divergence, two panels:
  L  held-out accuracy STRATIFIED by ΔK bin: single vs (single + β·f'(1)·ΔK) = TRL-reconstruction.
     The length term lifts accuracy only in the bins where ΔK favors the labelled winner, and is
     null at ΔK=0 ⇒ the "gain" is length, not learning.
  R  overall bars: single · TRL-reconstruction · OBSERVED matched TRL (from json) · ΔK=0 subset
     (length-neutral pairs ⇒ single≈TRL — the fair-comparison operating point, Goal 2 preview).

    python3 fig_length_bias.py [--recipe light|matched] [--divs kl rkl] [--beta 0.1]
"""
from __future__ import annotations

import argparse, csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from divergences import SHORT, COLORS, KLN_SHIFT

# ΔK bins (same partition as the data-side histogram): coarse enough to be stable at n=1000.
BIN_EDGES = [-np.inf, -4.5, -0.5, 0.5, 4.5, np.inf]
BIN_LABEL = ["ΔK≤-5", "-4..-1", "ΔK=0", "1..4", "ΔK≥5"]


def _last_eval_acc(path):
    if not os.path.exists(path):
        return None
    h = json.load(open(path))["history"]
    e = [x for x in h if "eval_acc" in x]
    return e[-1]["eval_acc"] if e else None


def load_csv(path):
    Sw, Sl, Kw, Kl = [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            Sw.append(float(r["Sw"])); Sl.append(float(r["Sl"]))
            Kw.append(int(r["Kw"])); Kl.append(int(r["Kl"]))
    m = np.array(Sw) - np.array(Sl)                 # single-sample margin
    dK = np.array(Kw, float) - np.array(Kl, float)  # length gap
    return m, dK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="light", choices=["light", "matched"])
    ap.add_argument("--dir", default="results/lenbias", help="per-pair CSVs from eval_length_bias.py")
    ap.add_argument("--divs", nargs="+", default=["kl", "rkl"])
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = "match1p7b" if args.recipe == "matched" else "light1p7b"
    obs_dir = "results/matched" if args.recipe == "matched" else "results/sft1p7b"
    args.out = args.out or f"results/stageB_{args.recipe}_length-bias_TRL-vs-single"

    present = [d for d in args.divs
               if os.path.exists(os.path.join(args.dir, f"stageB_{d}_newline_{tag}_perpair.csv"))]
    if not present:
        raise SystemExit(f"no per-pair CSVs for recipe={args.recipe} in {args.dir} — run "
                         f"jobs/eval_length_bias.slrm (RECIPE={args.recipe}) on the cluster and "
                         f"scp results/lenbias/*.csv here first.")

    b = args.beta
    fig, axes = plt.subplots(len(present), 1, figsize=(9.5, 4.8 * len(present)), squeeze=False)
    print(f"{'Ω':6s} {'f′(1)':>6s} {'single':>8s} {'TRLrec':>8s} {'TRLobs':>8s} {'ΔK=0':>8s}  (n)")
    for row, d in enumerate(present):
        m, dK = load_csv(os.path.join(args.dir, f"stageB_{d}_newline_{tag}_perpair.csv"))
        fp1 = 1.0 - KLN_SHIFT[d]                     # +1 RKL, −1 FKL, 0 non-adm
        col = COLORS[d]
        mt = m + b * fp1 * dK                        # TRL-reconstructed margin (exact for kl/rkl)
        acc = lambda x: float(np.mean(x > 0)) if len(x) else np.nan
        a_single, a_trl = acc(m), acc(mt)
        obs_trl = _last_eval_acc(os.path.join(obs_dir, f"stageB_{d}_newline_trl_{tag}.json"))
        obs_sgl = _last_eval_acc(os.path.join(obs_dir, f"stageB_{d}_newline_{tag}.json"))
        z = dK == 0
        a_z = acc(m[z])
        print(f"{SHORT[d]:6s} {fp1:>+6.0f} {a_single:>8.3f} {a_trl:>8.3f} "
              f"{('%.3f'%obs_trl) if obs_trl is not None else '   —  ':>8s} {a_z:>8.3f}  (n={len(m)}, n0={int(z.sum())})")

        # ── L: stratified accuracy by ΔK bin ─────────────────────────────────────────────
        axL = axes[row][0]
        idx = np.digitize(dK, BIN_EDGES[1:-1])       # 0..len(BIN_LABEL)-1
        xs = np.arange(len(BIN_LABEL)); w = 0.38
        s_bin = [acc(m[idx == k]) for k in xs]
        t_bin = [acc(mt[idx == k]) for k in xs]
        n_bin = [int(np.sum(idx == k)) for k in xs]
        axL.bar(xs - w / 2, s_bin, w, color="0.6", edgecolor="#222", lw=0.5, label="single")
        axL.bar(xs + w / 2, t_bin, w, color=col, alpha=0.85, hatch="xxx", edgecolor="#222", lw=0.5,
                label=f"+β·f′(1)·ΔK  = TRL")
        for k in xs:
            if np.isfinite(s_bin[k]) and np.isfinite(t_bin[k]):
                axL.annotate(f"{t_bin[k]-s_bin[k]:+.02f}", (k, max(s_bin[k], t_bin[k]) + 0.015),
                             ha="center", fontsize=8, fontweight="bold",
                             color="#1a7a1a" if t_bin[k] >= s_bin[k] else "#b00")
        axL.axhline(0.5, color="0.6", ls=":", lw=1)
        axL.set_xticks(xs)
        axL.set_xticklabels([f"{l}\nn={n}" for l, n in zip(BIN_LABEL, n_bin)], fontsize=8)
        axL.set_ylim(0.0, 1.0); axL.set_ylabel("held-out preference accuracy")
        axL.set_xlabel("length gap  ΔK = K_w − K_l")
        axL.set_title(f"{SHORT[d]}  (f′(1)={fp1:+.0f}) · overall single {a_single:.3f} → +β·f′(1)·ΔK {a_trl:.3f} "
                      f"(net {a_trl-a_single:+.3f}); length term acts only where ΔK favors the winner, 0 at ΔK=0")
        axL.legend(fontsize=9, loc="upper left", framealpha=0.95, edgecolor="#888")

    fig.suptitle(f"Stage B · Qwen3-1.7B · {args.recipe} · TRL's RKL/FKL accuracy change is the length "
                 "term β·f′(1)·(K_w−K_l), not the estimator", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150, bbox_inches="tight")
    print("wrote", args.out + ".png")


if __name__ == "__main__":
    main()
