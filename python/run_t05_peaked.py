"""
run_t05_peaked.py — t05: the recovery curves under a PEAKED reference policy.

Single-variable change from the t03 run: pi_ref is no longer uniform. Everything else — the
layered MDP, eps=0.2, Uniform(-0.8,0.8) rewards, gamma=0.9, the Bradley-Terry oracle on
gamma-discounted returns, 100 MDPs x 1 seed, weights calibrated by bisection per (Omega, peak) —
is byte-identical to the uniform-reference run.

CONFIG CHOICE THAT THE PAPER MUST QUOTE (it is not in the verified block, because the block
describes the uniform-reference environment):
    PEAKED_REF = (0.6, 0.2, 0.2)      max mass 0.6 vs 1/3 under uniform
The behaviour policy that logs the preference data stays uniform over actions, so this isolates
the reference in the regularizer from the data-collection distribution.

Writes data/tabular/run_t05_peaked.json so the figure can be redrawn without re-training.
Run from python/:  python run_t05_peaked.py [n_mdp]
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from regularizers import REGKEYS
from mdp import new_rewards
from seeds import ROOT_SEED, rng_for, REGIME_IDX
from experiments import calibrate, run_sweep
from dpo import TrainConfig, make_dataset

GAMMA, EPS, DEPTH = 0.9, 0.2, 4
PEAK = 0.7
NPAIRS, STEPS, BATCH, LR = 6000, 400, 16, 0.08
NMC = [1, 2, 4, 8, 16, 32]
PEAKED_REF = np.array([0.6, 0.2, 0.2])
N_MDP = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def one_mdp(mi):
    rew = new_rewards(DEPTH, rng_for(ROOT_SEED, "reward", mi))
    alphas = calibrate(rew, PEAK, GAMMA, EPS, ref=PEAKED_REF)
    data = make_dataset(rew, EPS, GAMMA, rng_for(ROOT_SEED, "data_off", mi, 0), n=NPAIRS)
    cfg = TrainConfig(gamma=GAMMA, steps=STEPS, batch=BATCH, lr=LR, n_mc=1, policy_mode="on")
    on = run_sweep(rew, alphas, data, cfg, EPS, NMC, 1,
                   rng_for(ROOT_SEED, "train", mi, 0, REGIME_IDX["on"], 0, 0, 0), ref=PEAKED_REF)
    cfg_off = TrainConfig(**{**cfg.__dict__, "policy_mode": "off"})
    off = run_sweep(rew, alphas, data, cfg_off, EPS, [1], 1,
                    rng_for(ROOT_SEED, "train", mi, 0, REGIME_IDX["off"], 0, 0, 0), ref=PEAKED_REF)
    return (mi,
            {nm: {rk: float(on[nm][rk]["finals"][0]) for rk in REGKEYS} for nm in NMC},
            {rk: float(off[1][rk]["finals"][0]) for rk in REGKEYS},
            {rk: float(alphas[rk]) for rk in REGKEYS})


def main():
    t0 = time.time()
    nproc = max(1, min(8, os.cpu_count() - 2))
    print(f"[t05] peaked ref {PEAKED_REF.tolist()} · {N_MDP} MDPs · n_mc {NMC} · {nproc} workers")
    with Pool(nproc) as p:
        res = p.map(one_mdp, range(N_MDP))
    res.sort(key=lambda r: r[0])
    out = {"config": {"env": {"gamma": GAMMA, "eps": EPS, "depth": DEPTH, "SN": 3, "NA": 3,
                              "reward": "Uniform(-0.8,0.8)"},
                      "peaked_ref": PEAKED_REF.tolist(), "peak_target": PEAK, "n_mdp": N_MDP,
                      "n_seeds": 1, "nmc": NMC, "npairs": NPAIRS, "steps": STEPS,
                      "batch": BATCH, "lr": LR,
                      "behaviour_policy": "uniform over actions (unchanged)"},
           "on":  {str(nm): {rk: [r[1][nm][rk] for r in res] for rk in REGKEYS} for nm in NMC},
           "off": {rk: [r[2][rk] for r in res] for rk in REGKEYS},
           "alphas": {rk: [r[3][rk] for r in res] for rk in REGKEYS}}
    os.makedirs("data/tabular", exist_ok=True)
    json.dump(out, open("data/tabular/run_t05_peaked.json", "w"))
    print(f"[t05] done in {time.time()-t0:.0f}s -> data/tabular/run_t05_peaked.json")
    for rk in REGKEYS:
        a = np.array(out["on"][str(NMC[-1])][rk]); b = np.array(out["off"][rk])
        print(f"  {rk:5s} on(n_mc={NMC[-1]}) {a.mean():.3f}±{1.96*a.std(ddof=1)/np.sqrt(len(a)):.3f}"
              f"   off {b.mean():.3f}±{1.96*b.std(ddof=1)/np.sqrt(len(b)):.3f}")


if __name__ == "__main__":
    main()
