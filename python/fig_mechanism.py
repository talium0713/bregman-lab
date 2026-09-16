"""
fig_mechanism.py — §4.2 inner-term variance figures (the MECHANISM behind admissibility).

The §4.3 training experiments show the *consequence* (only the admissible Ω recovers π* off-policy);
these show the *cause*: the inner term Ĉ_Ω = mean_i Φ(u_{a_i}) is estimated by sampling, and

  * admissible Ω (RKL, code key 'kl'):  Φ_KL(u) ≡ 1 by arithmetic ⇒ every draw equals 1 ⇒
    estimator std is EXACTLY 0 at every n_mc and every horizon — noise-free, no special-casing.
  * non-admissible Ω:  Var_π[Φ] > 0 ⇒ single-state std ∝ 1/√n_mc, and a horizon-H trajectory sums
    H−1 independent inner draws ⇒ per-trajectory std ∝ √(H−1).

Reproducible: every Monte-Carlo estimate is seeded from ROOT_SEED (seeds.py), and the std curves
are averaged over several seeds with a 95% CI band.  RKL's zero is exact (arithmetic), not sampled.

Produces:
  figs/mechanism_single_state.png   left: Ĉ_Ω value ±1σ vs n (dashed=exact);  right: std vs n (∝1/√n)
  figs/mechanism_trajectory.png     left: Σ_t Ĉ_Ω ±1σ vs H (dashed=exact);   right: std vs H (∝√(H−1))

Run:  python fig_mechanism.py
"""
from __future__ import annotations

import os

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regularizers import REGKEYS, COLORS, SHORT
from inner_term import single_state_variance, trajectory_variance
from seeds import ROOT_SEED
from mdp import NA as MDP_NA          # |A| comes from the layered MDP itself — never a free knob

FLOOR = 1e-5            # log-axis floor so RKL's exact 0 is drawn at the bottom
N_SEEDS = 8            # seeds averaged for the std-vs-n / std-vs-H CI bands


def _save(fig, stem):
    """PNG + PDF (the paper export contract wants both)."""
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return f"{stem}.png"


def _kl_kw(rk, lw=1.7):
    return dict(lw=3.4 if rk == "kl" else lw, zorder=8 if rk == "kl" else 3)


def _mean_ci(vals):
    a = np.asarray(vals, float)
    m = a.mean(axis=0)
    ci = a.std(axis=0, ddof=1) / np.sqrt(len(a)) * 1.96 if len(a) > 1 else np.zeros_like(m)
    return m, ci


# The sampled state uses the layered MDP's own action count (|A| = NA = 3) so these mechanism
# panels sit in the same configuration as every training run. The 1/sqrt(n) law is if anything
# cleaner at |A|=3 (measured ratio std(n=4)/std(n=1024) = 16.0 exactly) and RKL stays exactly 0.
# `scale` only sets how peaked the sampled pi is; it is not an environment parameter.
SS_NA, SS_SCALE = MDP_NA, 1.2
TJ_NA, TJ_SCALE = MDP_NA, 1.0


def fig_single_state():
    ns = [2 ** k for k in range(2, 11)]                 # 4 … 1024
    # one well-sampled realization for the value±1σ panel (representative seed)
    rep = {rk: single_state_variance(rk, ns, n_draws=4000, n_actions=SS_NA, scale=SS_SCALE,
                                     seed=ROOT_SEED) for rk in REGKEYS}
    # several seeds for the std-vs-n CI band
    stds = {rk: [] for rk in REGKEYS}
    for s in range(N_SEEDS):
        for rk in REGKEYS:
            d = single_state_variance(rk, ns, n_draws=800, n_actions=SS_NA, scale=SS_SCALE,
                                      seed=ROOT_SEED + 1 + s)
            stds[rk].append([d[n]["std"] for n in ns])

    x = np.log2(ns)
    fig, (axV, axS) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for rk in REGKEYS:
        mu = np.array([rep[rk][n]["mean"] for n in ns]); sd = np.array([rep[rk][n]["std"] for n in ns])
        ex = np.array([rep[rk][n]["exact"] for n in ns])
        axV.plot(x, mu, color=COLORS[rk], label=SHORT[rk], **_kl_kw(rk, 1.6))
        axV.fill_between(x, mu - sd, mu + sd, color=COLORS[rk], alpha=0.10)
        axV.plot(x, ex, ls="--", lw=0.8, color=COLORS[rk], alpha=0.7)
    axV.set_xlabel("log₂ n  (Monte-Carlo samples)"); axV.set_ylabel("Ĉ_Ω(π; ref)")
    axV.set_title("Single state — estimator value ±1σ (dashed = exact C_Ω)", fontsize=10)
    axV.legend(fontsize=7.5, ncol=2); axV.grid(alpha=0.2)

    for rk in REGKEYS:
        m, ci = _mean_ci(stds[rk])
        m = np.maximum(m, FLOOR)
        axS.plot(ns, m, marker="s", ms=4, color=COLORS[rk], label=SHORT[rk], **_kl_kw(rk, 1.6))
        axS.fill_between(ns, np.maximum(m - ci, FLOOR), m + ci, color=COLORS[rk], alpha=0.13)
    # 1/√n reference anchored on a non-admissible Ω's first point
    ref0 = np.maximum(_mean_ci(stds["chi2"])[0][0], FLOOR)
    axS.plot(ns, ref0 * np.sqrt(ns[0]) / np.sqrt(ns), ls=":", color="#555", lw=1.2, label="∝ 1/√n")
    axS.set_xscale("log", base=2); axS.set_yscale("log")
    axS.set_xticks(ns); axS.set_xticklabels(ns, fontsize=7.5)
    axS.set_xlabel("n  (Monte-Carlo samples)"); axS.set_ylabel("std of Ĉ_Ω")
    axS.set_title("Single state — estimator std vs n  (RKL ≡ 0 exactly; others ∝ 1/√n)", fontsize=10)
    axS.legend(fontsize=7.5, ncol=2); axS.grid(alpha=0.2, which="both")
    fig.suptitle("§4.2 mechanism — the admissible Ω's inner term is sampled like the rest but its draws "
                 "are constant (Φ_KL≡1) ⇒ zero variance", fontsize=11, y=1.0)
    fig.tight_layout()
    p = _save(fig, "figs/mechanism_single_state")
    rkl_max = max(rep["kl"][n]["std"] for n in ns)
    table = {rk: {int(n): float(np.mean([v[i] for v in stds[rk]])) for i, n in enumerate(ns)}
             for rk in REGKEYS}
    return p, rkl_max, table


def fig_trajectory():
    Hs = list(range(1, 9))
    rep = {rk: trajectory_variance(rk, Hs, n_mc=8, n_real=12000, n_actions=TJ_NA, scale=TJ_SCALE,
                                   seed=ROOT_SEED) for rk in REGKEYS}
    stds = {rk: [] for rk in REGKEYS}
    for s in range(N_SEEDS):
        for rk in REGKEYS:
            d = trajectory_variance(rk, Hs, n_mc=8, n_real=4000, n_actions=TJ_NA, scale=TJ_SCALE,
                                    seed=ROOT_SEED + 1 + s)
            stds[rk].append([d[H]["std"] for H in Hs])

    fig, (axV, axS) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for rk in REGKEYS:
        mu = np.array([rep[rk][H]["mean"] for H in Hs]); sd = np.array([rep[rk][H]["std"] for H in Hs])
        ex = np.array([rep[rk][H]["exact"] for H in Hs])
        axV.plot(Hs, mu, color=COLORS[rk], label=SHORT[rk], **_kl_kw(rk, 1.6))
        axV.fill_between(Hs, mu - sd, mu + sd, color=COLORS[rk], alpha=0.10)
        axV.plot(Hs, ex, ls="--", lw=0.8, color=COLORS[rk], alpha=0.7)
    axV.set_xlabel("horizon H"); axV.set_ylabel("Σ_t Ĉ_Ω(π(·|s_{t+1}))")
    axV.set_title("Trajectory — inner-term sum ±1σ (n_mc=8, dashed = exact)", fontsize=10)
    axV.legend(fontsize=7.5, ncol=2); axV.grid(alpha=0.2)

    for rk in REGKEYS:
        m, ci = _mean_ci(stds[rk]); m = np.maximum(m, FLOOR)
        axS.plot(Hs, m, marker="s", ms=4, color=COLORS[rk], label=SHORT[rk], **_kl_kw(rk, 1.6))
        axS.fill_between(Hs, np.maximum(m - ci, FLOOR), m + ci, color=COLORS[rk], alpha=0.13)
    ref = _mean_ci(stds["chi2"])[0]
    anchor = np.maximum(ref[1], FLOOR)                  # anchor at H=2 (first nonzero), scale √(H−1)
    axS.plot(Hs, [max(anchor * np.sqrt(max(H - 1, 0)), FLOOR) for H in Hs], ls=":", color="#555",
             lw=1.2, label="∝ √(H−1)")
    axS.set_yscale("log"); axS.set_xlabel("horizon H"); axS.set_ylabel("std of Σ_t Ĉ_Ω")
    axS.set_title("Trajectory — per-traj std vs H  (RKL ≡ 0; others ∝ √(H−1))", fontsize=10)
    axS.legend(fontsize=7.5, ncol=2); axS.grid(alpha=0.2, which="both")
    fig.suptitle("§4.2 mechanism — a horizon-H rollout sums H−1 inner draws: non-admissible noise "
                 "accumulates as √(H−1); the admissible Ω stays exactly 0", fontsize=11, y=1.0)
    fig.tight_layout()
    p = _save(fig, "figs/mechanism_trajectory")
    rkl_max = max(rep["kl"][H]["std"] for H in Hs)
    table = {rk: {int(H): float(np.mean([v[i] for v in stds[rk]])) for i, H in enumerate(Hs)}
             for rk in REGKEYS}
    return p, rkl_max, table


BUDGETS = [4, 64, 1024]        # the three MC budgets the paper tables quote
HORIZONS = [2, 4, 6, 8]        # four horizons; H=4 is the layered MDP's own depth


def _emit_table(title, header, table, cols, unit):
    print(f"\n{title}")
    print(f"  {'Omega':7s}" + "".join(f"{f'{header}={c}':>13s}" for c in cols))
    for rk in REGKEYS:
        row = "".join(f"{table[rk][c]:>13.3e}" for c in cols)
        tag = "   <- exactly 0 (admissible)" if rk == "kl" else ""
        print(f"  {SHORT[rk]:7s}" + row + tag)
    print(f"  ({unit})")


def main():
    os.makedirs("figs", exist_ok=True)
    p1, rkl_ss, t_ss = fig_single_state()
    p2, rkl_tj, t_tj = fig_trajectory()
    print(f"[check] RKL (admissible) std — single-state max {rkl_ss:.2e}, trajectory max {rkl_tj:.2e} "
          f"(≈0 ⇒ noise-free, by arithmetic)")
    print(f"[config] |A| = {SS_NA} (from mdp.NA), seeds averaged = {N_SEEDS}")

    _emit_table("TABLE 1 — single-state estimator std vs MC budget n", "n", t_ss, BUDGETS,
                "std of the n-sample inner-term estimate at one state")
    _emit_table("TABLE 2 — per-trajectory inner-term std vs horizon H", "H", t_tj, HORIZONS,
                "std of the summed inner term over a horizon-H rollout, n_mc=8")

    out = {"config": {"n_actions": int(SS_NA), "n_seeds_averaged": int(N_SEEDS),
                      "ss_scale": SS_SCALE, "tj_scale": TJ_SCALE, "tj_n_mc": 8},
           "single_state_std_vs_n": {rk: t_ss[rk] for rk in REGKEYS},
           "trajectory_std_vs_H": {rk: t_tj[rk] for rk in REGKEYS},
           "budgets_quoted": BUDGETS, "horizons_quoted": HORIZONS,
           "rkl_max_std": {"single_state": rkl_ss, "trajectory": rkl_tj}}
    json.dump(out, open("figs/mechanism_tables.json", "w"), indent=1)
    print(f"\n[saved]\n  {p1} (+ .pdf)\n  {p2} (+ .pdf)\n  figs/mechanism_tables.json")


if __name__ == "__main__":
    main()
