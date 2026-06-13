"""
Shared experiment helpers: α calibration, C_Ω statistics, the n_mc training sweep.
The FULL Part-3 experiment that produces every figure + results_data.js lives in run_part3.py;
this module holds the reusable pieces it imports.  `python experiments.py` delegates to run_part3.
"""
from __future__ import annotations

import numpy as np

from regularizers import REGKEYS
from mdp import solve_dp, uniform_pis, SN
from inner_term import C_exact, C_hat_phi
from dpo import TrainConfig, train_one


# ── α calibration: pick α_Ω so mean_s max_a π*(a|s) matches a target peak ──
def peakiness(reg_key, rewards, alpha, gamma, eps, ref=None):
    depth = rewards.shape[0]
    sol = solve_dp(reg_key, rewards, uniform_pis(depth), alpha, gamma, eps, ref)
    return float(np.mean([sol.pistar[l, s].max() for l in range(depth) for s in range(SN)]))


def calibrate(rewards, target_peak, gamma, eps, ref=None):
    """Bisection on log α (peakiness is monotone ↓ in α). Returns {reg_key: α}."""
    alphas = {}
    for rk in REGKEYS:
        lo, hi = 0.02, 40.0
        for _ in range(55):
            mid = np.sqrt(lo * hi)
            if peakiness(rk, rewards, mid, gamma, eps, ref) > target_peak:
                lo = mid
            else:
                hi = mid
        alphas[rk] = float(np.sqrt(lo * hi))
    return alphas


# ── C_Ω statistics at the calibrated π* (state spread + 1-sample Φ-form MC std).
#    No KL special case: std is MEASURED for every Ω with the same Φ-form; KL comes out ~0. ──
def c_stats(rewards, alphas, gamma, eps, estimator="phi", ref=None, rng=None):
    rng = rng or np.random.default_rng(0)
    depth = rewards.shape[0]
    out = {}
    for rk in REGKEYS:
        sol = solve_dp(rk, rewards, uniform_pis(depth), alphas[rk], gamma, eps, ref)
        Cs, sds = [], []
        for l in range(depth):
            for s in range(SN):
                p = np.maximum(sol.pistar[l, s], 1e-9)
                Cs.append(C_exact(rk, p, None if ref is None else ref))
                draws = [C_hat_phi(rk, p, 1, rng) for _ in range(200)]   # 1-sample Φ-form std
                sds.append(float(np.std(draws)))
        Cs = np.array(Cs)
        out[rk] = dict(meanC=float(Cs.mean()), state_std=float(Cs.std()),
                       mc_sd=float(np.mean(sds)))
    return out


# ── the three-stage sweep over n_mc ──
def run_sweep(rewards, alphas, data, cfg: TrainConfig, eps, n_mc_list, n_seeds, rng, ref=None):
    """Returns out[n_mc][reg_key] = dict(finals=[...], curves=[...], pols=[...])."""
    out = {}
    for nm in n_mc_list:
        out[nm] = {}
        for rk in REGKEYS:
            finals, curves, pols = [], [], []
            for _ in range(n_seeds):
                c = TrainConfig(**{**cfg.__dict__, "n_mc": nm})
                curve, final, pol = train_one(rk, rewards, alphas[rk], data, c, eps, rng, ref)
                finals.append(final)
                curves.append(curve)
                pols.append(pol)
            out[nm][rk] = dict(finals=finals, curves=curves, pols=pols)
    return out


def mean_std(xs):
    xs = np.asarray(xs, float)
    return float(xs.mean()), float(xs.std(ddof=1) if len(xs) > 1 else 0.0)



# ── `python experiments.py` delegates to the full Part-3 driver (single source of truth) ──
def main():
    import run_part3
    run_part3.main()


if __name__ == "__main__":
    main()
