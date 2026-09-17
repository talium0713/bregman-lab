"""
fig_mechanism.py — inner-term (Psi) variance figures, Amari vs canonical.

The estimator is Eq. 9's  Psi_hat^(n)(s') = (1/n) sum_i Psi_{a_i}(u_{a_i}),  a_i ~ pi(.|s').
Each figure is 2x2: rows are the generator normalization, columns are value and spread.

  rows      Amari  f'(1)=0     vs     canonical  f'(1)=f''(1)
  t01 cols  Psi_hat vs n (+-1 sigma, dashed = exact)  |  std of Psi_hat vs n   (~ 1/sqrt(n))
  t02 cols  sum_t Psi_hat vs H (+-1 sigma)            |  std of the sum vs H   (~ sqrt(H-1))

The point of the row split: Psi == 1 identically is a property of the CANONICAL RKL generator, not
of "being RKL". Re-normalized to f'(1)=0, RKL's Psi is 1 - 1/u and its estimator has ordinary
nonzero variance like everyone else. Euclidean is a Bregman divergence with no f(u) generator, so
it has no normalization freedom and is identical in both rows.

|A| is taken from mdp.NA so these panels sit in the same configuration as every training run.

Run:  python fig_mechanism.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regularizers import REGKEYS, COLORS, SHORT, make_standard, make_canonical, SHORT_TEX
from inner_term import single_state_variance, trajectory_variance
from seeds import ROOT_SEED
from mdp import NA as MDP_NA

FLOOR = 1e-3             # log-axis floor so an exact 0 is drawn at the bottom; sits just
                         # under the smallest real curve, so the axis is not mostly empty
N_SEEDS = 8              # seeds averaged for the std curves and their CI band
SS_NA, SS_SCALE = MDP_NA, 1.2
TJ_NA, TJ_SCALE = MDP_NA, 1.0
BUDGETS = [4, 64, 1024]
HORIZONS = [2, 4, 6, 8]

NORMS = [("amari", r"Amari  $f'(1)=0$"), ("canon", r"Canonical  $f'(1)=f''(1)$")]
ROW_FS = 13          # the row (normalization) label; deliberately larger than the axis labels


def _reg(rk, norm):
    """Euclidean has no f(u) generator, so no normalization freedom - natural form in both rows."""
    if rk == "euc":
        return None
    return make_standard(rk) if norm == "amari" else make_canonical(rk)


def _kl_kw(rk, lw=1.7):
    return dict(lw=3.0 if rk == "kl" else lw, zorder=8 if rk == "kl" else 3)


def _mean_ci(vals):
    a = np.asarray(vals, float)
    m = a.mean(axis=0)
    ci = a.std(axis=0, ddof=1) / np.sqrt(len(a)) * 1.96 if len(a) > 1 else np.zeros_like(m)
    return m, ci


def _save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return f"{stem}.png"


def _row_labels(fig, rows, left=0.125):
    """Draw the normalization label for each row in FIGURE coords, after the layout is final.
    Kept off the y label so the two can be sized separately and so neither collides with the
    tick labels, whose width depends on the data."""
    fig.tight_layout()
    fig.subplots_adjust(left=left)
    for ax, text in rows:
        b = ax.get_position()
        fig.text(0.012, b.y0 + b.height / 2, text, rotation=90, va="center", ha="left",
                 fontsize=ROW_FS)


def fig_single_state():
    ns = [2 ** k for k in range(2, 11)]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    table, rows = {}, []
    for r, (norm, nlab) in enumerate(NORMS):
        axV, axS = axes[r, 0], axes[r, 1]
        rep = {rk: single_state_variance(rk, ns, n_draws=4000, n_actions=SS_NA, scale=SS_SCALE,
                                         seed=ROOT_SEED, reg=_reg(rk, norm)) for rk in REGKEYS}
        stds = {rk: [] for rk in REGKEYS}
        for s in range(N_SEEDS):
            for rk in REGKEYS:
                d = single_state_variance(rk, ns, n_draws=1200, n_actions=SS_NA, scale=SS_SCALE,
                                          seed=ROOT_SEED + 1 + s, reg=_reg(rk, norm))
                stds[rk].append([d[n]["std"] for n in ns])
        for rk in REGKEYS:
            mu = np.array([rep[rk][n]["mean"] for n in ns]); sd = np.array([rep[rk][n]["std"] for n in ns])
            axV.plot(ns, mu, color=COLORS[rk], label=SHORT_TEX[rk], **_kl_kw(rk))
            axV.fill_between(ns, mu - sd, mu + sd, color=COLORS[rk], alpha=0.10)
            axV.plot(ns, [rep[rk][n]["exact"] for n in ns], ls="--", lw=0.8, color=COLORS[rk], alpha=0.7)
            m = np.maximum(_mean_ci(stds[rk])[0], FLOOR)
            axS.plot(ns, m, marker="s", ms=3.5, color=COLORS[rk], **_kl_kw(rk))
        ref = _mean_ci(stds["chi2"])[0]
        axS.plot(ns, np.maximum(ref[0] * np.sqrt(ns[0]) / np.sqrt(ns), FLOOR), ls=":", color="#555",
                 lw=1.1, label=r"$\propto 1/\sqrt{n}$")
        for ax in (axV, axS):
            ax.set_xscale("log", base=2); ax.set_xticks(ns)
            ax.set_xticklabels([rf"$2^{{{k}}}$" for k in range(2, 11)], fontsize=9)
            ax.grid(alpha=0.2, which="both")
        axS.set_yscale("log")
        rows.append((axV, nlab))
        axV.set_ylabel(r"$\widehat{\Psi}^{(n)}$", fontsize=11, labelpad=6)
        axS.set_ylabel(r"std of $\widehat{\Psi}^{(n)}$", fontsize=10, labelpad=4)
        if r == 0:
            axV.set_title(r"estimator value $\pm1\sigma$  (dashed = exact)", fontsize=10)
            axS.set_title(r"estimator spread", fontsize=10)
            axV.legend(fontsize=7, ncol=3)
        else:
            for ax in (axV, axS):
                ax.set_xlabel(r"$n$  (Monte-Carlo samples)")
        axS.legend(fontsize=7, loc="upper right")
        table[norm] = {rk: {int(n): float(np.mean([v[i] for v in stds[rk]]))
                            for i, n in enumerate(ns)} for rk in REGKEYS}
    _row_labels(fig, rows)
    return _save(fig, "figs/t01_single_state_variance"), table


def fig_trajectory():
    Hs = list(range(1, 9))
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    table, rows = {}, []
    for r, (norm, nlab) in enumerate(NORMS):
        axV, axS = axes[r, 0], axes[r, 1]
        rep = {rk: trajectory_variance(rk, Hs, n_mc=8, n_real=12000, n_actions=TJ_NA,
                                       scale=TJ_SCALE, seed=ROOT_SEED, reg=_reg(rk, norm))
               for rk in REGKEYS}
        stds = {rk: [] for rk in REGKEYS}
        for s in range(N_SEEDS):
            for rk in REGKEYS:
                d = trajectory_variance(rk, Hs, n_mc=8, n_real=4000, n_actions=TJ_NA,
                                        scale=TJ_SCALE, seed=ROOT_SEED + 1 + s, reg=_reg(rk, norm))
                stds[rk].append([d[H]["std"] for H in Hs])
        for rk in REGKEYS:
            mu = np.array([rep[rk][H]["mean"] for H in Hs]); sd = np.array([rep[rk][H]["std"] for H in Hs])
            axV.plot(Hs, mu, color=COLORS[rk], label=SHORT_TEX[rk], **_kl_kw(rk, 1.6))
            axV.fill_between(Hs, mu - sd, mu + sd, color=COLORS[rk], alpha=0.10)
            axV.plot(Hs, [rep[rk][H]["exact"] for H in Hs], ls="--", lw=0.8, color=COLORS[rk], alpha=0.7)
            m = np.maximum(_mean_ci(stds[rk])[0], FLOOR)
            axS.plot(Hs[1:], m[1:], marker="s", ms=3.5, color=COLORS[rk], **_kl_kw(rk, 1.6))
        ref = _mean_ci(stds["chi2"])[0]
        anchor = np.maximum(ref[1], FLOOR)
        axS.plot(Hs[1:], [max(anchor * np.sqrt(H - 1), FLOOR) for H in Hs[1:]], ls=":", color="#555",
                 lw=1.1, label=r"$\propto\sqrt{H-1}$")
        axS.set_yscale("log")
        for ax in (axV, axS):
            ax.grid(alpha=0.2, which="both")
        rows.append((axV, nlab))
        axV.set_ylabel(r"$\sum_t \widehat{\Psi}^{(n)}$", fontsize=11, labelpad=6)
        axS.set_ylabel(r"std of $\sum_t \widehat{\Psi}^{(n)}$", fontsize=10, labelpad=4)
        if r == 0:
            axV.set_title(r"trajectory sum $\pm1\sigma$  (dashed = exact)", fontsize=10)
            axS.set_title(r"spread of the sum", fontsize=10)
            axV.legend(fontsize=7, ncol=3)
        else:
            for ax in (axV, axS):
                ax.set_xlabel(r"horizon $H$")
        axS.legend(fontsize=7, loc="upper left")
        table[norm] = {rk: {int(H): float(np.mean([v[i] for v in stds[rk]]))
                            for i, H in enumerate(Hs)} for rk in REGKEYS}
    _row_labels(fig, rows)
    return _save(fig, "figs/t02_trajectory_compounding"), table


def _emit(title, header, table, cols):
    print(f"\n{title}")
    for norm, _ in NORMS:
        print(f"  [{norm}]" + "".join(f"{f'{header}={c}':>13s}" for c in cols))
        for rk in REGKEYS:
            print(f"   {SHORT[rk]:7s}" + "".join(f"{table[norm][rk][c]:>13.3e}" for c in cols))


def main():
    os.makedirs("figs", exist_ok=True)
    p1, t_ss = fig_single_state()
    p2, t_tj = fig_trajectory()
    print(f"[config] |A| = {SS_NA} (from mdp.NA), seeds averaged = {N_SEEDS}")
    _emit("TABLE 1 — std of Psi_hat^(n) vs budget n", "n", t_ss, BUDGETS)
    _emit("TABLE 2 — std of the trajectory sum vs horizon H", "H", t_tj, HORIZONS)
    print(f"\n[key] RKL std at n=1024:  amari {t_ss['amari']['kl'][1024]:.3e}   "
          f"canonical {t_ss['canon']['kl'][1024]:.3e}  "
          f"(Psi == 1 is a property of the canonical generator, not of RKL)")
    json.dump({"config": {"n_actions": int(SS_NA), "n_seeds_averaged": int(N_SEEDS),
                          "ss_scale": SS_SCALE, "tj_scale": TJ_SCALE, "tj_n_mc": 8},
               "single_state_std_vs_n": t_ss, "trajectory_std_vs_H": t_tj,
               "budgets_quoted": BUDGETS, "horizons_quoted": HORIZONS},
              open("figs/mechanism_tables.json", "w"), indent=1)
    print(f"\n[saved]\n  {p1} (+ .pdf)\n  {p2} (+ .pdf)\n  figs/mechanism_tables.json")


if __name__ == "__main__":
    main()
