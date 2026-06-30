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

KLN = "data/tabular/run_adiv_p80/results.json"        # KL-consistent (the fix)
STD = "data/tabular/run_adiv_p80_std/results.json"    # standard normalization (the artifact)


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


def main():
    kln_p = sys.argv[1] if len(sys.argv) > 1 else KLN
    std_p = sys.argv[2] if len(sys.argv) > 2 else STD
    for p in (kln_p, std_p):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — run: python run_adiv.py --peak-ref 0.8"
                     + ("" if "std" not in p else " --no-kl-norm"))
    man, a_grid, kln = agg(kln_p)
    _, _, std = agg(std_p)
    os.makedirs("figs", exist_ok=True)

    C_STD, C_KLN = "#d9534f", "#2c7fb8"
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    _plot(ax, a_grid, std, C_STD, "standard  f'(1)=0  (Φ→1−1/u: artifact)", ls="--")
    _plot(ax, a_grid, kln, C_KLN, "KL-consistent  f'(1)=1  (Φ→1: fixed)")
    ax.scatter([1.0], [kln[1.0][0]], s=150, facecolors="none", edgecolors=COLORS["kl"],
               linewidths=2.0, zorder=6, label="a=1: canonical KL (shared)")
    ax.axvline(1.0, color=COLORS["kl"], ls=":", lw=1.0, alpha=0.6)
    ax.set_xlabel("α-divergence parameter  a   (a→0 FKL · a=1 RKL · a=2 χ²)")
    ax.set_ylabel("off-policy gap  Δπ  (mean TV vs π*)")
    ax.set_ylim(0, max(max(v[0] + v[1] for v in std.values()),
                       max(v[0] + v[1] for v in kln.values())) * 1.12)
    ax.grid(alpha=0.2)

    # zoom inset around a=1
    m = [a for a in a_grid if 0.84 <= a <= 1.16]
    if len(m) > 2:
        axin = ax.inset_axes([0.6, 0.40, 0.36, 0.42])
        for d, c, ls in [(std, C_STD, "--"), (kln, C_KLN, "-")]:
            mu = np.array([d[a][0] for a in m]); ci = np.array([d[a][1] for a in m])
            axin.plot(m, mu, ls, color=c, lw=1.4, marker="o", ms=3.5)
            axin.fill_between(m, mu - ci, mu + ci, color=c, alpha=0.13)
        axin.scatter([1.0], [kln[1.0][0]], s=60, facecolors="none", edgecolors=COLORS["kl"], linewidths=1.5)
        axin.set_xlim(0.84, 1.16); axin.set_title("zoom: a≈1", fontsize=8)
        axin.tick_params(labelsize=6.5); axin.grid(alpha=0.2)
        ax.indicate_inset_zoom(axin, edgecolor="#bbb")

    ax.set_title(f"α-div sweep — generator-normalization fix (off-policy, α=RKL@peak{man['peak_ref']}, "
                 f"{man['n_mdp']} MDPs, ±95% CI)\nsame Ω/π*; only the inner-term estimator differs",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    p = "figs/adiv_compare_p80.png"; fig.savefig(p, dpi=140); plt.close()
    print(f"[saved] {p}")
    print(f"  a=1 (shared): Δπ={kln[1.0][0]:.3f}")
    print(f"  a=0.5: std={std[0.5][0]:.3f}  kln={kln[0.5][0]:.3f}")
    print(f"  a=2.0: std={std[2.0][0]:.3f}  kln={kln[2.0][0]:.3f}")


if __name__ == "__main__":
    main()
