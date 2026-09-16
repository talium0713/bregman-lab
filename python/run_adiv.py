"""
run_adiv.py — α-divergence parameter sweep at a FIXED regularizer temperature.

Unlike the headline runs (each Ω calibrated to a target peak), here the temperature α is PINNED to
RKL's value at peak=0.8 — calibrate(rewards, 0.8)["kl"] per MDP — and only the α-divergence
parameter `a` is swept.  This is the trained/recovery analogue of the deterministic α-morph: as `a`
moves through the family (a→0 FKL · a=1 RKL=KL, the admissible point · a=2 χ²), we measure the
off-policy recovery gap Δπ at that single temperature.  Expectation: Δπ dips at a=1 (admissible),
since only KL has a noise-free inner term; away from 1 the inner-term variance degrades recovery.

Off-policy regime only (n_mc irrelevant), 100 independent MDPs, ±95% CI — reproducible via
seeds.rng_for (train role keyed so it never collides with the main runs).

Run:  python run_adiv.py [--n-mdp 100] [--jobs 9]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np

from regularizers import REG, make_adiv, make_standard, make_canonical
from mdp import solve_dp, uniform_pis, new_rewards, SN
from dpo import TrainConfig, make_dataset, train_one
from experiments import calibrate
from seeds import ROOT_SEED, rng_for

GAMMA, EPS, DEPTH = 0.9, 0.2, 4
NPAIRS, STEPS, BATCH, LR = 6000, 400, 16, 0.08
# α pinned to RKL's calibration at this peak (overridable via --peak-ref).
# `a` grid: broad family coverage + DENSE around a=1 to resolve the admissibility well
# (Var_π[Φ_a]→0 continuously as a→1, so the dip should be a steep continuous well, not a jump).
A_GRID = [0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 0.99, 0.995,
          1.0, 1.005, 1.01, 1.02, 1.04, 1.07, 1.1, 1.15, 1.3, 1.6, 2.0]
REP_A = [0.3, 0.7, 1.0, 1.3, 2.0]              # representative `a` for the morph panels (subset)


def reg_for_a(a: float, kl_norm: bool = True):
    """The α-divergence member at `a`; at a=1 make_adiv is undefined (forbids a∈{0,1}), so use the
    exact KL representative in the SAME normalization as the neighbours — canonical (u ln u, Φ≡1) for
    kl_norm=True, standard (u ln u−(u−1), Φ=1−1/u) for kl_norm=False. Using the canonical KL for BOTH
    (the old behaviour) made the standard curve spuriously DIP to the canonical value only at a=1; each
    curve must keep its own normalization through a=1 so it stays continuous (standard high, canonical
    low — they do NOT meet at a=1)."""
    if abs(a - 1.0) < 1e-9:
        return make_canonical("kl") if kl_norm else make_standard("kl")
    return make_adiv(a, kl_norm=kl_norm)


def realized_peak(pistar):
    return float(np.mean([pistar[l, s].max() for l in range(DEPTH) for s in range(SN)]))


def run_mdp(block):
    mi, root, n_seeds, peak_ref = block["mi"], block["root"], block["n_seeds"], block["peak_ref"]
    kl_norm = block["kl_norm"]
    rewards = new_rewards(DEPTH, rng_for(root, "reward", mi))
    alpha = float(calibrate(rewards, peak_ref, GAMMA, EPS)["kl"])          # fixed temperature
    off = make_dataset(rewards, EPS, GAMMA, rng_for(root, "data_off", mi, 0), NPAIRS)
    cells = []
    for ai, a in enumerate(A_GRID):
        reg = reg_for_a(a, kl_norm)
        sol = solve_dp("adiv", rewards, uniform_pis(DEPTH), alpha, GAMMA, EPS, reg=reg)
        finals, pol0 = [], None
        for sd in range(n_seeds):
            # train role keyed (mi, 0, 9, ai, 1, sd) — the 9 keeps it disjoint from the main runs
            trng = rng_for(root, "train", mi, 0, 9, ai, 1, sd)
            cfg = TrainConfig(gamma=GAMMA, steps=STEPS, batch=BATCH, lr=LR, n_mc=1, policy_mode="off")
            _, final, pol = train_one("adiv", rewards, alpha, off, cfg, EPS, trng, reg=reg)
            finals.append(float(final))
            if sd == 0 and mi == 0:
                pol0 = pol.tolist()
        cells.append(dict(mi=mi, ai=ai, a=a, alpha=alpha, peak=realized_peak(sol.pistar),
                          mean=float(np.mean(finals)), finals=finals,
                          pistar=sol.pistar.tolist() if mi == 0 else None, pol0=pol0))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mdp", type=int, default=100)
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--root", type=int, default=ROOT_SEED)
    ap.add_argument("--peak-ref", type=float, default=0.8, help="pin α to RKL's calibration at this peak")
    # PAPER BASELINE = standard α-div normalization (f'(1)=0; (t−1) NOT added). --kl-norm switches to
    # the KL-consistent normalization (f'(1)=1) used only for the before/after diagnostic.
    ap.add_argument("--kl-norm", dest="kl_norm", action="store_true",
                    help="KL-consistent normalization (f'(1)=1); default is standard (paper baseline)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    suffix = "_kln" if args.kl_norm else ""
    out = args.out or f"data/tabular/run_adiv_p{int(round(args.peak_ref * 100))}{suffix}"
    os.makedirs(out, exist_ok=True)
    blocks = [dict(mi=mi, root=args.root, n_seeds=args.n_seeds, peak_ref=args.peak_ref,
                   kl_norm=args.kl_norm) for mi in range(args.n_mdp)]
    print(f"[run] α-div sweep · a={A_GRID}")
    print(f"      {args.n_mdp} MDPs × {len(A_GRID)} a × {args.n_seeds} seeds, off-policy, "
          f"α=RKL@peak{args.peak_ref} (per MDP), jobs={args.jobs}")

    results, t0, done = [], time.time(), 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_mdp, b): b for b in blocks}
        for fut in as_completed(futs):
            results.extend(fut.result()); done += 1
            if done % 10 == 0 or done == len(blocks):
                el = time.time() - t0
                print(f"  [{done}/{len(blocks)} MDPs] {len(results)} cells · "
                      f"elapsed {el/60:.1f}m  eta {el/done*(len(blocks)-done)/60:.1f}m", flush=True)

    manifest = dict(root_seed=args.root, n_mdp=args.n_mdp, n_seeds=args.n_seeds, a_grid=A_GRID,
                    rep_a=REP_A, peak_ref=args.peak_ref, kl_norm=args.kl_norm, regime="off",
                    gamma=GAMMA, eps=EPS, depth=DEPTH, npairs=NPAIRS, steps=STEPS,
                    created_utc=datetime.now(timezone.utc).isoformat())
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({"manifest": manifest, "results": results}, f)
    print(f"[saved] {out}/results.json  ({len(results)} cells)")


if __name__ == "__main__":
    main()
