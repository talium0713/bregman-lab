"""
fig_horizon.py — horizon (H) ablation: does the §4.2 √(H−1) inner-term noise show up in recovery?

A horizon-H trajectory sums H−1 inner-term contributions.  §4.2 predicts the per-trajectory inner
noise grows as √(H−1) for non-admissible Ω (0 for the admissible KL).  So in off-policy recovery
the gap Δπ should INCREASE with H for non-admissible Ω and stay FLAT for KL.  This reads the
run_H{2,3,4,6,8} runs (peak 0.7, 100 MDPs) and plots Δπ vs H per Ω, per regime, at each n_mc.

Produces  figs/horizon_nmc{1,16}.png  — 3 panels (off / off_on / on), 7 Ω, ±95% CI.

Run:  python fig_horizon.py
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regularizers import REGKEYS, COLORS, SHORT


def _kl_kw(rk, lw=1.7):
    return dict(lw=3.4 if rk == "kl" else lw, zorder=8 if rk == "kl" else 3)


def load():
    """agg[nm][regime][rk][H] = (mean, ci95), pooled over MDPs. off uses its nm=1 cells for any nm."""
    agg = {}
    Hs = set()
    for d in sorted(glob.glob("data/tabular/run_H*/results.json")):
        H = int(re.search(r"run_H(\d+)", d).group(1)); Hs.add(H)
        res = json.load(open(d))["results"]
        buckets = {}
        for r in res:
            buckets.setdefault((r["regime"], r["rk"], r["nm"]), []).extend(r["finals"])
        for (reg, rk, nm), fin in buckets.items():
            f = np.asarray(fin, float)
            ci = float(f.std(ddof=1) / np.sqrt(len(f)) * 1.96) if len(f) > 1 else 0.0
            agg.setdefault(nm, {}).setdefault(reg, {}).setdefault(rk, {})[H] = (float(f.mean()), ci)
    return agg, sorted(Hs)


def fig_for_nmc(agg, Hs, nm):
    regimes = ["off", "off_on", "on"]
    titles = {"off": "off-policy (n_mc irrelevant)",
              "off_on": f"off-on-policy / Dyna (n_mc={nm})",
              "on": f"on-policy (n_mc={nm})"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    ymax = 0
    for ax, reg in zip(axes, regimes):
        # off ignores n_mc -> its cells are stored under nm=1
        src_nm = 1 if reg == "off" else nm
        d = agg.get(src_nm, {}).get(reg, {})
        for rk in REGKEYS:
            if rk not in d:
                continue
            hs = [H for H in Hs if H in d[rk]]
            mu = np.array([d[rk][H][0] for H in hs]); ci = np.array([d[rk][H][1] for H in hs])
            ax.plot(hs, mu, marker="o", ms=4, color=COLORS[rk], label=SHORT[rk], **_kl_kw(rk))
            ax.fill_between(hs, mu - ci, mu + ci, color=COLORS[rk], alpha=0.12, zorder=2)
            ymax = max(ymax, (mu + ci).max())
        ax.set_xlabel("horizon  H  (layers)"); ax.set_title(titles[reg], fontsize=10)
        ax.set_xticks(Hs); ax.grid(alpha=0.2)
    for ax in axes:
        ax.set_ylim(0, ymax * 1.06)
    axes[0].set_ylabel("off-policy gap  Δπ  (mean TV vs π*)")
    axes[-1].legend(fontsize=8, ncol=2, title="Ω (RKL = admissible)")
    fig.suptitle(f"Horizon ablation (n_mc={nm}) — Δπ vs H · non-admissible Ω rise with H "
                 "(√(H−1) inner-noise accumulation); RKL stays flat · 100 MDPs, ±95% CI", fontsize=11, y=1.0)
    fig.tight_layout()
    p = f"figs/horizon_nmc{nm}.png"; fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close()
    return p


def main():
    os.makedirs("figs", exist_ok=True)
    agg, Hs = load()
    if not Hs:
        raise SystemExit("no data/tabular/run_H*/ — run the H ablation first")
    print(f"[data] H={Hs} · regimes/n_mc from run_H*")
    outs = []
    for nm in sorted({nm for nm in agg}):
        if nm in (1, 16):
            outs.append(fig_for_nmc(agg, Hs, nm))
    # quick table: RKL vs a non-admissible, off regime, vs H
    print("  off-policy Δπ vs H (RKL vs FKL):")
    off = agg.get(1, {}).get("off", {})
    print("   H  " + "  ".join(f"{H:>5d}" for H in Hs))
    for rk in ("kl", "rkl"):
        if rk in off:
            print(f"   {SHORT[rk]:4s}" + "  ".join(f"{off[rk][H][0]:5.3f}" for H in Hs if H in off[rk]))
    print("[saved]\n  " + "\n  ".join(outs))


if __name__ == "__main__":
    main()
