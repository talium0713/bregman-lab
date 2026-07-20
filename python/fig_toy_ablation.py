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
    fig = plt.figure(figsize=(12.8, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.26)
    axL = fig.add_subplot(gs[0])
    gsR = gs[1].subgridspec(2, 1, height_ratios=[5, 1], hspace=0.10)   # broken y-axis
    axRt = fig.add_subplot(gsR[0])                 # top: non-admissible band (zoomed)
    axRb = fig.add_subplot(gsR[1], sharex=axRt)    # bottom: RKL ≈ 0

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

    # ---- Right (broken y-axis): non-admissible band on top, RKL ≈ 0 on bottom ----
    data = {}
    for k in KEYS:
        med, lo, hi = [], [], []
        for na in A_GRID:
            vals = np.array([offpolicy_phi_std(k, na, seed=s) for s in range(N_SEEDS)])
            med.append(np.median(vals)); lo.append(np.quantile(vals, .25)); hi.append(np.quantile(vals, .75))
        data[k] = (np.array(med), np.array(lo), np.array(hi))
    for k in KEYS:                                  # top: everyone except RKL, zoomed to their band
        if k == "kl":
            continue
        med, lo, hi = data[k]
        axRt.plot(A_GRID, med, color=COLORS[k], lw=1.9, marker="o", ms=4, label=SHORT[k])
        axRt.fill_between(A_GRID, lo, hi, color=COLORS[k], alpha=0.12)
    axRt.set_xscale("log"); axRt.set_yscale("log"); axRt.set_ylim(2e4, 2e7)
    axRt.grid(True, which="both", alpha=0.15)
    axRt.legend(ncol=2, fontsize=8, framealpha=0.9, loc="upper left")
    axRt.set_title(r"toy: off-policy noise $\mathrm{std}_{a\sim\pi_{\mathrm{ref}}}[\Phi(u_a)]$ vs |A|  (median ± IQR)")
    axRt.text(3, 2.5e4, " |A|=3 (tabular)", fontsize=7.5, color="0.45", rotation=90, va="bottom")
    axRt.text(152000, 2.5e4, "|A|=152k (Qwen) ", fontsize=7.5, color="0.45", rotation=90, va="bottom", ha="right")

    axRb.plot(A_GRID, data["kl"][0], color=COLORS["kl"], lw=3.0, marker="o", ms=4)   # bottom: RKL ≈ 0
    axRb.axhline(0.0, color="0.6", ls=":", lw=1)
    axRb.set_xscale("log"); axRb.set_ylim(-6e-16, 6e-16); axRb.set_yticks([0])
    axRb.text(A_GRID[0] * 1.4, 2.6e-16, r"RKL $\equiv 0$  (machine $\varepsilon\!\approx\!2{\times}10^{-16}$)",
              fontsize=8, color=COLORS["kl"], va="bottom")
    for xa in (3, 152000):
        axRt.axvline(xa, color="0.8", ls="--", lw=1); axRb.axvline(xa, color="0.8", ls="--", lw=1)
    axRb.grid(True, which="both", axis="x", alpha=0.15)
    axRb.set_xlabel(r"action-set size $|A|$  (= vocab at LLM scale)")

    # broken-axis diagonal cut marks between the two right panels
    axRt.spines["bottom"].set_visible(False); axRt.tick_params(bottom=False, labelbottom=False)
    axRb.spines["top"].set_visible(False)
    dm = 0.5
    kw = dict(marker=[(-1, -dm), (1, dm)], markersize=12, linestyle="none",
              color="0.3", mec="0.3", mew=1.3, clip_on=False)
    axRt.plot([0, 1], [0, 0], transform=axRt.transAxes, **kw)
    axRb.plot([0, 1], [1, 1], transform=axRb.transAxes, **kw)

    fig.suptitle("Toy (π_ref uniform, Gaussian logits): RKL noise-free at every |A|; non-admissible "
                 "noise rises monotonically — magnitude toy-dependent, Stage A measures reality",
                 fontsize=10.5, y=0.99)
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
