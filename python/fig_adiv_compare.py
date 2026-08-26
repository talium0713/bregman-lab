"""
fig_adiv_compare.py — before/after of the α-div generator-normalization fix.

Overlays the off-policy recovery Δπ vs the α-div parameter `a` under the two normalizations:
  standard  (f'(1)=0):  a→1 limit is t ln t − t + 1 ⇒ Φ→1−1/u ⇒ a SPURIOUS jump at a=1 (knife-edge),
  KL-consistent (f'(1)=1, `kl_norm`): a→1 limit is canonical KL t ln t ⇒ Φ→1 ⇒ continuous well.
Both share the exact a=1 point (canonical KL).  Same Ω/π* under either normalization — only the
inner-term estimator (hence off-policy recovery) differs.

Run:  python fig_adiv_compare.py [kln_results.json std_results.json]
      (defaults to run_adiv_p80 [kl_norm] and run_adiv_p80_std [standard])
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regularizers import COLORS

KLN = "data/tabular/run_adiv_p80_kln/results.json"    # KL-consistent (the diagnostic)
STD = "data/tabular/run_adiv_p80/results.json"        # standard normalization (PAPER BASELINE)


def agg(path):
    R = json.load(open(path)); man = R["manifest"]; res = R["results"]
    a_grid = man["a_grid"]
    out = {}
    for a in a_grid:
        f = np.concatenate([np.asarray(c["finals"]) for c in res if abs(c["a"] - a) < 1e-9])
        out[a] = (float(f.mean()), float(f.std(ddof=1) / np.sqrt(len(f)) * 1.96))
    return man, a_grid, out


def _plot(ax, a_grid, d, color, label, ls="-"):
    A = np.array(a_grid)
    mu = np.array([d[a][0] for a in a_grid]); ci = np.array([d[a][1] for a in a_grid])
    ax.plot(A, mu, ls, color=color, lw=1.8, marker="o", ms=4, label=label, zorder=3)
    ax.fill_between(A, mu - ci, mu + ci, color=color, alpha=0.13, zorder=1)
    return mu, ci


def render(kln_p, std_p, paper=False):
    """One peak's before/after α-div normalization figure (C2). paper=True → text-width profile (B12)."""
    man, a_grid, kln = agg(kln_p)
    _, _, std = agg(std_p)
    os.makedirs("figs", exist_ok=True)
    peak = man["peak_ref"]

    if paper:
        plt.rcParams.update({"font.size": 9})
    C_STD, C_KLN = "#d9534f", "#2c7fb8"
    fig, ax = plt.subplots(figsize=(5.5, 3.7) if paper else (7.8, 4.6))
    _plot(ax, a_grid, std, C_STD, r"standard normalization ($f'(1)=0$)", ls="--")
    _plot(ax, a_grid, kln, C_KLN, r"canonical normalization ($f'(1)=f''(1)$)")
    ax.scatter([1.0], [kln[1.0][0]], s=150, facecolors="none", edgecolors=COLORS["kl"],
               linewidths=2.0, zorder=6, label=r"$\alpha=1$: reverse KL (shared by both)")
    ax.axvline(1.0, color=COLORS["kl"], ls=":", lw=1.0, alpha=0.6)
    ax.set_xlabel(r"divergence-family parameter  $\alpha$"
                  "\n"
                  r"($\alpha\!\to\!0$: FKL   $\alpha\!=\!1$: RKL   $\alpha\!=\!2$: $\chi^2$)")
    ax.set_ylabel(r"recovery gap $\Delta_\pi$ (mean TV to $\pi^\star$)")
    ax.set_ylim(0, 0.55)
    ax.grid(alpha=0.2)

    # B1: no in-figure title. Caption items → α-div generator-normalization fix; off-policy; temperature
    # β = RKL@peak{peak}; {man['n_mdp']} MDPs; ±95% CI; same Ω/π*, only the inner-term estimator differs;
    # the standard curve's α→1 jump (Ψ→1−1/u) is the artifact, the canonical well is the fix.
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    stem = f"figs/adiv_compare_p{int(round(peak*100))}{'_paper' if paper else ''}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=140, bbox_inches="tight")
    plt.close()
    if paper:
        plt.rcParams.update(plt.rcParamsDefault)
    print(f"[saved] {stem}.png/.pdf  ·  α=1 Δπ={kln[1.0][0]:.3f} · "
          f"α=0.5 std={std[0.5][0]:.3f}/canon={kln[0.5][0]:.3f}")
    return stem


def main():
    if len(sys.argv) > 2:                       # explicit  kln_results.json  std_results.json
        render(sys.argv[1], sys.argv[2]); return
    for pk in (60, 70, 80):                     # C2⑤: p60 · p70 · p80
        kln_p, std_p = f"data/tabular/run_adiv_p{pk}_kln/results.json", f"data/tabular/run_adiv_p{pk}/results.json"
        if os.path.exists(kln_p) and os.path.exists(std_p):
            render(kln_p, std_p)
        else:
            print(f"[skip] peak 0.{pk}: missing {kln_p} or {std_p}")
    if os.path.exists(KLN) and os.path.exists(STD):   # p80 also as a text-width paper profile (B12)
        render(KLN, STD, paper=True)


if __name__ == "__main__":
    main()
