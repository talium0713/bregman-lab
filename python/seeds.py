"""
seeds.py — reproducible, addressable RNG derivation for the tabular paper experiments.

PROBLEM this fixes:  run_part3.py threads ONE np.random.default_rng(0) through reward draws,
calibration, dataset construction, AND training.  Consequences: the "SEEDS" loop is a mere
repetition index over a shared stream (not addressable), results are ORDER-DEPENDENT (adding an
Ω or changing the n_mc grid perturbs every cell), and a single (Ω, regime, n_mc, seed) cell
cannot be reproduced in isolation.

FIX:  every unit of randomness gets its own independent stream, derived deterministically from a
recorded ROOT_SEED and an integer key that names the unit.  np.random.SeedSequence guarantees the
streams are statistically independent and the mapping (ROOT_SEED, key) -> stream is a pure
function — so results are order-independent, per-cell reproducible, and "100 seeds" is a real
average over 100 independent, named streams.

Roles (the first key component) separate the four sources of randomness so they never collide:
  reward    — the fixed-MDP reward draw         (keyed by mdp index)
  data_off  — off-policy (uniform) preference data  (keyed by mdp, peak)
  data_on   — on-policy (pi*_Omega) preference data (keyed by mdp, peak, reg)
  train     — theta init + inner-term MC + batch order (keyed by mdp, peak, regime, reg, n_mc, seed)

Usage:
    rng = rng_for(ROOT_SEED, "reward", mdp_idx)
    rng = rng_for(ROOT_SEED, "train", mdp_idx, peak_idx, regime_idx, reg_idx, n_mc, seed)
"""
from __future__ import annotations

import numpy as np

# Default root seed, recorded in the paper. Change ONLY to regenerate every result from scratch.
ROOT_SEED = 20260629

# Role codes — the first component of every key, so the four randomness sources never overlap.
ROLE = {"reward": 0, "data_off": 1, "data_on": 2, "train": 3}

# Stable string->int maps so keys stay pure integers (SeedSequence requires int entropy).
REGIME_IDX = {"off": 0, "off_on": 1, "on": 2, "exact": 3}
REGKEY_IDX = {"kl": 0, "adiv": 1, "rkl": 2, "js": 3, "hel": 4, "chi2": 5, "euc": 6}


def rng_for(root: int, role: str, *key) -> np.random.Generator:
    """Independent, reproducible Generator keyed by (root, ROLE[role], *key).

    `key` is any sequence of ints (or things int() accepts) that names this unit of randomness.
    Same (root, role, key) -> identical stream, regardless of call order or what else ran."""
    entropy = [int(root), ROLE[role], *(int(k) for k in key)]
    return np.random.default_rng(np.random.SeedSequence(entropy))
