"""
fig_adiv_invariant.py — the normalization-INVARIANT admissibility curve for the α-div family.

The training Δπ sweep (fig_adiv) measures the plug-in estimator's effect and is
generator-normalization-DEPENDENT (standard → knife-edge, kl_norm → smooth well).  The invariant
alternative plots the intrinsic quantity std_π[C_Ω(π)] vs the α-div parameter a: it is affine-
invariant (adding c(t−1) shifts C_Ω by a constant, leaving its spread unchanged), deterministic
(no training), and 0 exactly at a=1 (reverse-KL) — the unique divergence whose inner term is
policy-independent.  We overlay both normalizations to make the invariance visible (they coincide).

Run:  python fig_adiv_invariant.py
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regularizers import REG, make_adiv, COLORS
from seeds import ROOT_SEED, rng_for

NA = 3
REF = np.full(NA, 1.0 / NA)
N_POL = 4000                      # random policies to estimate the spread of C_Ω


def C_omega(spec, p):
    u = p / REF
    return float(np.sum(p * spec.fp(u)) - np.sum(REF * spec.f(u)))


def std_C(reg, pols):
    return float(np.std([C_omega(reg.spec, np.maximum(p, 1e-9)) for p in pols]))


def reg_at(a, kl_norm):
    return REG["kl"] if abs(a - 1.0) < 1e-12 else make_adiv(a, kl_norm=kl_norm)


def main():
    os.makedirs("figs", exist_ok=True)
    rng = rng_for(ROOT_SEED, "reward", 999)          # fixed reproducible policy sample
    pols = [rng.dirichlet(np.ones(NA)) for _ in range(N_POL)]
    avals = np.concatenate([np.linspace(0.1, 0.95, 12), [1.0], np.linspace(1.05, 2.0, 12)])
    std_std = np.array([std_C(reg_at(a, False), pols) for a in avals])
    std_kln = np.array([std_C(reg_at(a, True), pols) for a in avals])

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(avals, std_std, "-", color="#444", lw=2.0, marker="o", ms=4,
            label=r"standard normalization  $f'(1)=0$")
    ax.plot(avals, std_kln, "--", color=COLORS["kl"], lw=1.6, marker="x", ms=5,
            label=r"canonical  $f'(1)=f''(1)$  (coincides $\to$ invariant)")
    ax.scatter([1.0], [0.0], s=170, facecolors="none", edgecolors=COLORS["kl"], linewidths=2.2, zorder=6)
    ax.axvline(1.0, color=COLORS["kl"], ls=":", lw=1.0, alpha=0.6)
    ax.annotate("$\\alpha=1$: reverse-KL\n(permissible, =0)", xy=(1.0, 0.0),
                xytext=(0.30, 0.55), textcoords="axes fraction", fontsize=8.5, color=COLORS["kl"],
                ha="center", arrowprops=dict(arrowstyle="->", color=COLORS["kl"], lw=1.0))
    ax.set_xlabel(r"divergence-family parameter  $\alpha$   "
                  r"($\alpha\!\to\!0$: FKL · $\alpha\!=\!1$: RKL · $\alpha\!=\!2$: $\chi^2$)")
    ax.set_ylabel(r"$\mathrm{std}_\pi[C_\Omega(\pi)]$   (0 $\Leftrightarrow$ inner term policy-independent = permissible)")
    ax.set_ylim(0, max(std_std.max(), std_kln.max()) * 1.06); ax.grid(alpha=0.2)
    ax.legend(fontsize=8.5, loc="upper center")
    ax.set_title(r"Normalization-invariant permissibility metric — $\mathrm{std}_\pi[C_\Omega]$ vs $\alpha$", fontsize=10)
    # B1 caption items: both normalizations coincide (invariant); zero only at α=1 (reverse-KL); deterministic, no training
    fig.tight_layout()
    p = "figs/adiv_invariant.png"; fig.savefig(p, dpi=140); plt.close()
    print(f"[saved] {p}")
    for a, s in zip(avals, std_std):
        if abs(a - 1) < 1e-9 or a in (avals[0], avals[-1]):
            print(f"  a={a:.2f}  std_π[C_Ω]={s:.4f}")


if __name__ == "__main__":
    main()
