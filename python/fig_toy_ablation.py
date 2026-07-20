"""
fig_toy_ablation.py — TOY check of the "stronger at scale" claim before Stage A.

Question: does the off-policy inner-term noise grow with the action-set size |A| (= vocab at
LLM scale), while the admissible RKL stays exactly 0?

This is a TOY (π_ref uniform, π = softmax of Gaussian logits). It demonstrates the *mechanism*,
not the real magnitude — the quantitative |A|-scaling depends on the true π_ref shape, which
Stage A must measure. What is exact (not toy-dependent) is the left panel: Φ_RKL ≡ 1 while every
other Φ diverges as u→0, and u→0 is exactly the off-policy condition.

Left  : Φ(u) = f'(u) − f(u)/u vs u (log-x). RKL flat at 1; others blow up as u→0.
Right : off-policy single-sample inner-term noise  std_{a~π_ref}[Φ(u_a)]  vs |A| (log-log),
        median ± IQR over seeds. RKL pinned at a floor (exact 0); others rise with |A|.

Run:  python fig_toy_ablation.py   →   figs/toy_Asize_ablation.png
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regularizers import REG, COLORS, SHORT

FIGDIR = os.path.join(os.path.dirname(__file__), "figs")
KEYS = ["kl", "adiv", "rkl", "js", "hel", "chi2"]     # f-divergences (skip euc)
A_GRID = [3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 152000]
N_SEEDS = 10
SCALE = 3.0                # logit std; peak ≈0.24 at |A| large
CAP = 2_000_000            # n_states * |A| budget per realization
FLOOR = 1e-3               # log-axis floor so RKL's exact 0 is visible at the bottom


def offpolicy_phi_std(reg_key: str, na: int, seed: int) -> float:
    """std of a single off-policy inner-term sample Φ(u_a), state~logits, a~π_ref (=uniform).
    Marginal over (state, a~uniform) ⇒ std over all entries of Φ."""
    rng = np.random.default_rng(seed)
    R = REG[reg_key]
    n_states = int(max(40, min(2000, CAP // na)))
    lg = rng.standard_normal((n_states, na)) * SCALE
    p = np.exp(lg - lg.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    u = p * na                                    # π_ref uniform ⇒ u = |A|·π
    phi = R.Phi(u)                                # (n_states, na); RKL ⇒ all 1
    return float(phi.std())


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.9))

    # ---- Left: Φ(u) vs u (the root cause; exact, not toy) ----
    u = np.logspace(-8, 2, 2000)
    for k in KEYS:
        phi = np.atleast_1d(REG[k].Phi(u)) * np.ones_like(u)
        axL.plot(u, np.abs(phi), color=COLORS[k], lw=3.2 if k == "kl" else 1.8,
                 zorder=8 if k == "kl" else 3, label=SHORT[k])
    axL.axhline(1.0, color="0.6", ls=":", lw=1, zorder=1)
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel(r"$u = \pi_\theta/\pi_{\mathrm{ref}}$  (small $u$ = off-policy)")
    axL.set_ylabel(r"$|\Phi(u)| = |f'(u)-f(u)/u|$")
    axL.set_title(r"root cause: $\Phi_{\mathrm{RKL}}\equiv 1$, others diverge as $u\to0$")
    axL.legend(ncol=2, fontsize=9, framealpha=0.9)
    axL.grid(True, which="both", alpha=0.15)

    # ---- Right: off-policy inner-term noise vs |A| ----
    for k in KEYS:
        med, lo, hi = [], [], []
        for na in A_GRID:
            vals = np.array([offpolicy_phi_std(k, na, seed=s) for s in range(N_SEEDS)])
            med.append(np.median(vals)); lo.append(np.quantile(vals, .25)); hi.append(np.quantile(vals, .75))
        med = np.maximum(np.array(med), FLOOR)
        lo = np.maximum(np.array(lo), FLOOR); hi = np.maximum(np.array(hi), FLOOR)
        axR.plot(A_GRID, med, color=COLORS[k], lw=3.2 if k == "kl" else 1.8,
                 marker="o", ms=4, zorder=8 if k == "kl" else 3, label=SHORT[k])
        if k != "kl":
            axR.fill_between(A_GRID, lo, hi, color=COLORS[k], alpha=0.12, zorder=2)
    axR.axhline(FLOOR, color="0.6", ls=":", lw=1)
    axR.axvline(3, color="0.75", ls="--", lw=1)
    axR.text(3, FLOOR * 1.4, " tabular |A|=3", fontsize=8, color="0.4", rotation=90, va="bottom")
    axR.axvline(152000, color="0.75", ls="--", lw=1)
    axR.text(152000, FLOOR * 1.4, " Qwen2.5 |A|=152k", fontsize=8, color="0.4", rotation=90, va="bottom", ha="right")
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlabel(r"action-set size $|A|$  (= vocab at LLM scale)")
    axR.set_ylabel(r"off-policy noise  $\mathrm{std}_{a\sim\pi_{\mathrm{ref}}}[\Phi(u_a)]$")
    axR.set_title("toy: single-sample inner-term noise vs |A|  (median ± IQR)")
    axR.legend(ncol=2, fontsize=9, framealpha=0.9)
    axR.grid(True, which="both", alpha=0.15)

    fig.suptitle("Toy check (π_ref uniform, Gaussian logits): RKL noise-free at every |A|; "
                 "non-admissible noise grows — magnitude is toy-dependent, Stage A measures reality",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "toy_Asize_ablation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)

    # small text table for the write-up
    print("\nstd_{a~π_ref}[Φ]  (median over %d seeds):" % N_SEEDS)
    show = [3, 100, 1000, 152000]
    print("|A|:  " + "  ".join("%9d" % a for a in show))
    for k in KEYS:
        row = [np.median([offpolicy_phi_std(k, a, s) for s in range(N_SEEDS)]) for a in show]
        print("%-6s " % SHORT[k] + "  ".join("%9.3g" % v for v in row))


if __name__ == "__main__":
    main()
