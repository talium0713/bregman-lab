"""
fig_adiv.py — figures for the fixed-temperature α-divergence parameter sweep (run_adiv.py).

  fig:adiv-curve   Δπ vs the α-div parameter `a` at fixed α (=RKL@peak0.8), with the realized peak
                   on a twin axis.  Off-policy recovery dips sharply only at a=1 (the admissible KL).
  fig:adiv-morph   5 representative `a` as SEPARATE panels (π* target vs π_θ recovered over the
                   (layer,state,action) index) — small multiples, to avoid the near-a=1 overlap that
                   a single overlaid morph plot suffers.

Run:  python fig_adiv.py [data/tabular/run_adiv/results.json]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from regularizers import COLORS
from mdp import SN, NA

DEPTH = 4
# gradient across the family: a→0 FKL · a=1 RKL (admissible, pink) · a→2 χ²
CMAP = LinearSegmentedColormap.from_list("adiv", [(0.0, COLORS["rkl"]), (0.5, COLORS["kl"]),
                                                  (1.0, "#f39c12")])


def col(a):
    return CMAP(min(max(a / 2.0, 0.0), 1.0))


def load(path):
    R = json.load(open(path))
    return R["manifest"], R["results"]


def aggregate(results, a_grid):
    out = {}
    for a in a_grid:
        finals = np.concatenate([np.asarray(c["finals"]) for c in results if c["a"] == a])
        peaks = np.array([c["peak"] for c in results if c["a"] == a])
        ci = float(finals.std(ddof=1) / np.sqrt(len(finals)) * 1.96) if len(finals) > 1 else 0.0
        out[a] = dict(mean=float(finals.mean()), ci=ci, peak=float(peaks.mean()))
    return out


def fig_curve(man, agg, a_grid):
    A = np.array(a_grid)
    mu = np.array([agg[a]["mean"] for a in a_grid]); ci = np.array([agg[a]["ci"] for a in a_grid])
    pk = np.array([agg[a]["peak"] for a in a_grid])
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.plot(A, mu, "-", color="#444", lw=1.4, zorder=2)
    ax.fill_between(A, mu - ci, mu + ci, color="#888", alpha=0.18, zorder=1)
    for a in a_grid:
        ax.errorbar(a, agg[a]["mean"], yerr=agg[a]["ci"], fmt="o", ms=6, color=col(a),
                    capsize=3, zorder=4)
    # highlight the admissible point a=1
    ax.scatter([1.0], [agg[1.0]["mean"]], s=180, facecolors="none", edgecolors=COLORS["kl"],
               linewidths=2.2, zorder=5)
    ax.axvline(1.0, color=COLORS["kl"], ls="--", lw=1.0, alpha=0.6)
    ax.annotate("a=1: RKL\n(admissible)", xy=(1.0, agg[1.0]["mean"]),
                xytext=(0.28, agg[1.0]["mean"] + 0.13), fontsize=8.5, color=COLORS["kl"],
                ha="center", arrowprops=dict(arrowstyle="->", color=COLORS["kl"], lw=1.0))
    ax.set_xlabel("α-divergence parameter  a   (a→0 FKL · a=1 RKL · a=2 χ²)")
    ax.set_ylabel("off-policy gap  Δπ  (mean TV vs π*)")
    ax.set_ylim(0, max(mu + ci) * 1.15); ax.grid(alpha=0.2)

    # zoom inset resolving the well around a=1 (the dense grid lives here)
    m = (A >= 0.84) & (A <= 1.16)
    if m.sum() > 2:
        axin = ax.inset_axes([0.62, 0.12, 0.34, 0.40])
        axin.plot(A[m], mu[m], "-", color="#444", lw=1.1)
        axin.fill_between(A[m], (mu - ci)[m], (mu + ci)[m], color="#888", alpha=0.18)
        for a in a_grid:
            if 0.84 <= a <= 1.16:
                axin.errorbar(a, agg[a]["mean"], yerr=agg[a]["ci"], fmt="o", ms=4, color=col(a), capsize=2)
        axin.scatter([1.0], [agg[1.0]["mean"]], s=70, facecolors="none", edgecolors=COLORS["kl"], linewidths=1.6)
        axin.set_xlim(0.84, 1.16); axin.set_title("zoom: a≈1", fontsize=8)
        axin.tick_params(labelsize=6.5); axin.grid(alpha=0.2)
        ax.indicate_inset_zoom(axin, edgecolor="#bbb")

    norm = "KL-consistent f'(1)=1" if man.get("kl_norm") else "standard f'(1)=0 (paper baseline)"
    ax.set_title(f"α-div sweep at fixed temperature α=RKL@peak{man['peak_ref']}  ·  {norm}\n"
                 f"{man['n_mdp']} MDPs · off-policy · ±95% CI", fontsize=10)
    fig.tight_layout()
    sfx = "_kln" if man.get("kl_norm") else ""
    p = f"figs/adiv_curve_p{int(round(man['peak_ref']*100))}{sfx}.png"; fig.savefig(p, dpi=140); plt.close()
    return p


def fig_morph(man, results, agg):
    rep = man["rep_a"]
    by = {c["a"]: c for c in results if c["mi"] == 0}      # MDP0 cells carry pistar + pol0
    block = SN * NA
    centers = [l * block + (block - 1) / 2 for l in range(DEPTH)]
    fig, axes = plt.subplots(1, len(rep), figsize=(3.4 * len(rep), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, rep):
        c = by[a]
        star = np.array(c["pistar"]).reshape(-1)
        est = np.array(c["pol0"]).reshape(-1)
        idx = np.arange(len(star))
        ax.plot(idx, star, "--", color="#444", lw=1.0, marker="o", ms=2.2, label="π* target", zorder=2)
        ax.plot(idx, est, "-", color=col(a), lw=2.0, marker="s", ms=2.4, label="π_θ recovered", zorder=3)
        for l in range(1, DEPTH):
            ax.axvline(l * block - 0.5, color="#ddd", lw=0.7)
        tag = "  (RKL, admissible)" if abs(a - 1) < 1e-9 else ""
        ax.set_title(f"a = {a:g}{tag}\nΔπ={agg[a]['mean']:.3f} · peak={agg[a]['peak']:.2f}", fontsize=9)
        ax.set_ylim(0, 1); ax.set_xticks(centers); ax.set_xticklabels([f"ℓ{l}" for l in range(DEPTH)])
        ax.tick_params(labelsize=7.5)
        if a == rep[0]:
            ax.legend(fontsize=7.5, loc="upper right"); ax.set_ylabel("π(a | s)")
    fig.suptitle(f"α-div morph at fixed α=RKL@peak{man['peak_ref']} (MDP 0): only a=1 (KL) recovers "
                 "its target off-policy; the others leave π_θ far from π*", fontsize=10.5, y=1.02)
    fig.tight_layout()
    sfx = "_kln" if man.get("kl_norm") else ""
    p = f"figs/adiv_morph_p{int(round(man['peak_ref']*100))}{sfx}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close()
    return p


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/tabular/run_adiv/results.json"
    man, results = load(path)
    os.makedirs("figs", exist_ok=True)
    a_grid = man["a_grid"]
    agg = aggregate(results, a_grid)
    print(f"[data] {len(results)} cells · {man['n_mdp']} MDPs · a={a_grid}")
    print("  a     Δπ±CI        peak")
    for a in a_grid:
        print(f"  {a:<4g}  {agg[a]['mean']:.3f}±{agg[a]['ci']:.3f}   {agg[a]['peak']:.3f}"
              + ("  <- admissible" if abs(a - 1) < 1e-9 else ""))
    p1 = fig_curve(man, agg, a_grid)
    p2 = fig_morph(man, results, agg)
    print(f"[saved]\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
