"""Sanity for the A6 'exact' regime: RKL exact must ≈ RKL off (its inner term is constant, so exact
and off coincide), and every other Ω should collapse toward the RKL level under exact (the A6 claim)."""
import numpy as np
from mdp import new_rewards
from dpo import TrainConfig, make_dataset, train_one
from experiments import calibrate
from seeds import ROOT_SEED, rng_for
from regularizers import REGKEYS, SHORT

GAMMA, EPS, DEPTH, PEAK, MI = 0.9, 0.2, 4, 0.7, 0
rewards = new_rewards(DEPTH, rng_for(ROOT_SEED, "reward", MI))
alphas = calibrate(rewards, PEAK, GAMMA, EPS)
data = make_dataset(rewards, EPS, GAMMA, rng_for(ROOT_SEED, "data_off", MI, 0), 6000)


def run(rk, regime, nseeds=6):
    fs = []
    for sd in range(nseeds):
        cfg = TrainConfig(gamma=GAMMA, steps=400, lr=0.08, batch=16, n_mc=1, policy_mode=regime)
        trng = rng_for(ROOT_SEED, "train", MI, 0, 9, 0, 1, sd)
        _, final, _ = train_one(rk, rewards, alphas[rk], data, cfg, EPS, trng)
        fs.append(float(final))
    return float(np.mean(fs)), float(np.std(fs) / np.sqrt(nseeds) * 1.96)


print(f"peak {PEAK}, MDP {MI}, 6 seeds\nΩ        off Δπ          exact Δπ")
for rk in REGKEYS:
    mo, co = run(rk, "off")
    me, ce = run(rk, "exact")
    tag = "  <- RKL sanity: exact≈off" if rk == "kl" else ""
    print(f"{SHORT[rk]:7} {mo:.3f}±{co:.3f}    {me:.3f}±{ce:.3f}{tag}")
