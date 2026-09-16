"""
run_t06_alpha_nmc.py — t06: the alpha-family recovery gap against the Monte-Carlo budget.

Extends run_adiv.py (which sweeps `a` off-policy at n_mc=1 only) with the budget axis the paper's
alpha-family figure needs. Everything else is byte-identical to run_adiv.py: the same A_GRID, the
same pinned temperature (RKL's calibrated weight at `peak_ref`, so only `a` moves), the same
reg_for_a() handling of the a=1 knife-edge, the same standard (Amari, f'(1)=0) normalization that
is the paper baseline, 100 MDPs x 1 seed.

CONFIG CHOICE THE PAPER MUST QUOTE (the budget axis does not exist in any cached alpha run):
    NMC = [1, 4, 16]   on-policy;  the off-policy arm has no budget axis (single logged a')
peak_ref = 0.7 so the off-policy n_mc=1 curve reproduces data/tabular/run_adiv_p70 exactly —
that reproduction is asserted at the end as a cross-check.

Writes data/tabular/run_t06_alpha_nmc.json. Run from python/:  python run_t06_alpha_nmc.py [n_mdp]
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from mdp import new_rewards
from seeds import ROOT_SEED, rng_for
from experiments import calibrate
from dpo import TrainConfig, make_dataset, train_one
from run_adiv import A_GRID, reg_for_a, GAMMA, EPS, DEPTH, NPAIRS, STEPS, BATCH, LR

NMC = [1, 4, 16]
PEAK_REF = 0.7
KL_NORM = False          # standard (Amari) normalization — the paper baseline, as in run_adiv.py
N_MDP = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def one_mdp(mi):
    rewards = new_rewards(DEPTH, rng_for(ROOT_SEED, "reward", mi))
    alpha = float(calibrate(rewards, PEAK_REF, GAMMA, EPS)["kl"])       # pinned temperature
    data = make_dataset(rewards, EPS, GAMMA, rng_for(ROOT_SEED, "data_off", mi, 0), NPAIRS)
    on = {nm: [] for nm in NMC}
    off = []
    for ai, a in enumerate(A_GRID):
        reg = reg_for_a(a, KL_NORM)
        for nm in NMC:
            cfg = TrainConfig(gamma=GAMMA, steps=STEPS, batch=BATCH, lr=LR, n_mc=nm, policy_mode="on")
            trng = rng_for(ROOT_SEED, "train", mi, 0, 9, ai, nm, 1)      # 9 = the alpha-sweep block
            _, final, _ = train_one("adiv", rewards, alpha, data, cfg, EPS, trng, reg=reg)
            on[nm].append(float(final))
        cfg = TrainConfig(gamma=GAMMA, steps=STEPS, batch=BATCH, lr=LR, n_mc=1, policy_mode="off")
        trng = rng_for(ROOT_SEED, "train", mi, 0, 9, ai, 1, 0)           # matches run_adiv.py exactly
        _, final, _ = train_one("adiv", rewards, alpha, data, cfg, EPS, trng, reg=reg)
        off.append(float(final))
    return mi, on, off, alpha


def main():
    t0 = time.time()
    nproc = max(1, min(8, (os.cpu_count() or 4) - 2))
    print(f"[t06] |A_GRID|={len(A_GRID)} · n_mc {NMC} (+off) · {N_MDP} MDPs · "
          f"peak_ref {PEAK_REF} · kl_norm={KL_NORM} · {nproc} workers")
    with Pool(nproc) as p:
        res = p.map(one_mdp, range(N_MDP))
    res.sort(key=lambda r: r[0])
    out = {"config": {"a_grid": list(A_GRID), "nmc": NMC, "peak_ref": PEAK_REF,
                      "kl_norm": KL_NORM, "n_mdp": N_MDP, "n_seeds": 1,
                      "npairs": NPAIRS, "steps": STEPS, "batch": BATCH, "lr": LR,
                      "env": {"gamma": GAMMA, "eps": EPS, "depth": DEPTH, "SN": 3, "NA": 3,
                              "reward": "Uniform(-0.8,0.8)"},
                      "temperature": "pinned to RKL's calibrated weight at peak_ref (only `a` moves)"},
           "on": {str(nm): [r[1][nm] for r in res] for nm in NMC},   # [mdp][ai]
           "off": [r[2] for r in res],
           "alpha": [r[3] for r in res]}
    os.makedirs("data/tabular", exist_ok=True)
    json.dump(out, open("data/tabular/run_t06_alpha_nmc.json", "w"))
    print(f"[t06] done in {time.time()-t0:.0f}s -> data/tabular/run_t06_alpha_nmc.json")

    # cross-check: the off-policy n_mc=1 curve must reproduce the cached run_adiv_p70
    ref_path = "data/tabular/run_adiv_p70/results.json"
    if os.path.exists(ref_path) and N_MDP == 100:
        R = json.load(open(ref_path))["results"]
        cached = {}
        for c in R:
            cached.setdefault(c["ai"], []).extend(c["finals"])
        mine = np.asarray(out["off"])                                   # (mdp, ai)
        worst = 0.0
        for ai in range(len(A_GRID)):
            d = abs(float(np.mean(cached[ai])) - float(mine[:, ai].mean()))
            worst = max(worst, d)
        print(f"[cross-check] max |mean Δπ  mine − run_adiv_p70| over the {len(A_GRID)} a-values: "
              f"{worst:.2e}  ({'MATCH' if worst < 1e-9 else 'DIFFERS — investigate'})")


if __name__ == "__main__":
    main()
