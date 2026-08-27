"""
fig_permissibility_bias.py — the clean off-policy permissibility demonstration (no training, no noise).

For each divergence take its CANONICAL generator (the best-normalized, f'(1)=f''(1) representative). The
DPO off-policy inner term  C_Ω(π) = E_{a~π}[Ψ(u_a)]  (u = π/π_ref) is estimated from a single logged
action a′ ~ π_ref, so the single-sample estimator's bias as π drifts off-policy is
    bias(π) = E_{a′~π_ref}[Ψ(u_{a′})] − E_{a~π}[Ψ(u_a)].
Permissible ⇔ Ψ constant ⇔ bias ≡ 0 for EVERY π — true for RKL alone (canonical Ψ≡1). This sweeps π
from π_ref (drift t=0, on-policy) toward a random deterministic target (t=1), averaging |bias| over many
random (π_ref, target) draws. RKL stays exactly 0; every other canonical divergence fans out.

Run:  python fig_permissibility_bias.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regularizers import REGKEYS, SHORT, COLORS, make_canonical

NA = 3                                   # actions per state (matches the tabular MDP, mdp.py)
K = 400                                  # random (π_ref, target) draws to average |bias| over
DRIFTS = np.linspace(0.0, 0.97, 25)


def compute(seed: int = 0):
    rng = np.random.default_rng(seed)
    fdivs = [rk for rk in REGKEYS if rk != "euc"]           # euc: not an f-divergence, no Ψ
    Psi = {rk: make_canonical(rk).Phi for rk in fdivs}
    absb = {rk: np.zeros(len(DRIFTS)) for rk in fdivs}
    for _ in range(K):
        ref = rng.dirichlet(np.ones(NA))
        tgt = np.zeros(NA); tgt[rng.integers(NA)] = 1.0     # a concentrated (deterministic) target
        for j, t in enumerate(DRIFTS):
            pi = (1 - t) * ref + t * tgt
            pi = pi / pi.sum()
            u = np.maximum(pi, 1e-12) / ref
            for rk in fdivs:
                ps = np.atleast_1d(Psi[rk](u)) * np.ones(NA)
                bias = float(np.sum(ref * ps) - np.sum(pi * ps))
                absb[rk][j] += abs(bias)
    for rk in fdivs:
        absb[rk] /= K
    return fdivs, absb


def render():
    fdivs, absb = compute()
    os.makedirs("figs", exist_ok=True)
    floor = 1e-3                                             # log-axis floor; RKL's exact 0 is drawn here
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for rk in fdivs:                                         # non-RKL canonical divergences fan out
        if rk == "kl":
            continue
        ax.plot(DRIFTS, np.maximum(absb[rk], floor), color=COLORS[rk], lw=1.7,
                marker="o", ms=3, label=SHORT[rk], zorder=3)
    ax.plot(DRIFTS, np.full_like(DRIFTS, floor), color=COLORS["kl"], lw=3.2, zorder=6,
            label=r"RKL  ($|\mathrm{bias}|\equiv 0$, permissible)")
    ax.annotate(r"$\equiv 0$ (permissible)", xy=(0.62, floor), xytext=(0.30, floor * 3.0),
                color=COLORS["kl"], fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=COLORS["kl"], lw=1.0))
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.7, None)
    ax.set_xlabel(r"off-policy drift  $t$    "
                  r"($\pi=(1{-}t)\,\pi_{\mathrm{ref}}+t\,\pi_{\mathrm{target}}$;   $t{=}0$: on-policy)")
    ax.set_ylabel(r"$|\mathrm{bias}|$ of the single-sample off-policy inner term")
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=8, loc="upper left", title="canonical generator", title_fontsize=8)
    # Caption: only RKL's canonical Ψ≡1 makes the single logged a′ unbiased for every π; every other
    # divergence — even at its canonical normalization — is biased, and the bias grows without bound as
    # π moves off-policy. NA=3, mean |bias| over K=400 random (π_ref, deterministic target) draws.
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"figs/permissibility_bias.{ext}", dpi=140, bbox_inches="tight")
    plt.close()
    tail = {SHORT[rk]: round(float(absb[rk][-1]), 2) for rk in fdivs}
    print(f"[saved] figs/permissibility_bias.png/.pdf  ·  mean|bias| @ t={DRIFTS[-1]:.2f}: {tail}")


if __name__ == "__main__":
    render()
