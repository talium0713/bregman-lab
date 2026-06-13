"""
Layered MDP and the regularized backward-induction solver (port of `solveDP` in math.js).

ENVIRONMENT (matches the JS experiment, `task3-admissibility.js`):
  - DEPTH decision layers, SN states per layer, |A| = 3 actions.
  - Stochastic transitions: action a reaches its "target" next state a with prob 1−ε,
    else uniform over the other states  ->  transP(a, ε)[s'] = (1−ε) if s'==a else ε/2.
  - Rewards r[l][s][a] drawn iid Uniform(−0.8, 0.8) on EVERY (l, s, a).
  - Reference policy π_ref is uniform on every state.

  NOTE vs the paper (Off_policy_admissibility.pdf, §4.1, Fig 2): the paper's environment
  is a *deterministic complete tree* (W=3, H=3, |S|=40), reward 0 on internal nodes and
  leaf rewards ~ N(0,1). Set EPS=0 here to recover deterministic transitions; the tree's
  unique-parent structure is what makes c(s') a clean state function (see inner_term.py,
  c_violation: ε=0 ⇒ zero posterior ambiguity over which a generated s').

The solver returns, per (layer, state), the soft-optimal policy π* (Eq 6), the regularized
value V* (Eq 5/12), the on-policy value V^π, and the Bregman gap B_Ω(π‖π*) (policy-deviation
theorem). The α-optimal policy π* is what trajectory-DPO is asked to recover.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from regularizers import REG, uniform_ref

SN = 3        # states per layer
NA = 3        # actions
DEFAULT_DEPTH = 4
DEFAULT_EPS = 0.2


def resolve_ref(ref, depth: int, na: int = NA, ns: int = SN) -> np.ndarray:
    """
    Normalize a reference-policy spec into a (depth, ns, na) array.
      ref is None            -> uniform over actions on every state  (the default)
      ref has shape (na,)    -> same reference on every state
      ref has shape (depth, ns, na) -> per-state reference, used verbatim
    """
    if ref is None:
        return np.broadcast_to(uniform_ref(na), (depth, ns, na)).copy()
    ref = np.asarray(ref, dtype=float)
    if ref.shape == (na,):
        ref = np.broadcast_to(ref / ref.sum(), (depth, ns, na)).copy()
    elif ref.shape == (depth, ns, na):
        ref = ref / ref.sum(axis=-1, keepdims=True)
    else:
        raise ValueError(f"ref must be None, ({na},), or ({depth},{ns},{na}); got {ref.shape}")
    return ref


def transP(a: int, eps: float) -> np.ndarray:
    """P(s' | a): target state a w.p. 1−ε, else uniform over the remaining states."""
    p = np.full(SN, eps / (SN - 1))
    p[a] = 1 - eps
    return p


def state_occupancy(policy: np.ndarray, eps: float) -> np.ndarray:
    """
    d[l,s] = probability of visiting state s at layer l when rolling out `policy` (depth,SN,NA)
    from a uniform start, under the ε-stochastic transitions. Each layer sums to 1.
    Used for the paper's state-occupancy-weighted Δπ metric (Eq 20).
    """
    depth = policy.shape[0]
    d = np.zeros((depth, SN))
    d[0, :] = 1.0 / SN
    for l in range(depth - 1):
        for s in range(SN):
            if d[l, s] <= 0:
                continue
            for a in range(NA):
                d[l + 1] += d[l, s] * policy[l, s, a] * transP(a, eps)
    return d


def new_rewards(depth: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform(−0.8, 0.8) rewards on every (layer, state, action). Shape (depth, SN, NA)."""
    return rng.uniform(-0.8, 0.8, size=(depth, SN, NA))


def uniform_pis(depth: int) -> np.ndarray:
    """A behaviour policy that is uniform everywhere. Shape (depth, SN, NA)."""
    return np.full((depth, SN, NA), 1.0 / NA)


@dataclass
class DPSolution:
    Qs: np.ndarray        # (depth, SN, NA)  soft-optimal Q*
    pistar: np.ndarray    # (depth, SN, NA)  α-optimal policy π*  (the DPO recovery target)
    Vs: np.ndarray        # (depth, SN)      regularized value V*
    Vp: np.ndarray        # (depth, SN)      on-policy value V^π for the supplied `pis`
    B: np.ndarray         # (depth, SN)      Bregman gap B_Ω(π ‖ π*)
    boundary: bool        # True if any π* touched the simplex boundary


def solve_dp(reg_key: str, r: np.ndarray, pis: np.ndarray, alpha: float,
             gamma: float, eps: float, ref=None, reg=None) -> DPSolution:
    """
    Backward induction for the regularized MDP (port of solveDP).

    For each (l, s):  Q*(l,s,a) = r[l][s][a] + γ E_{s'~P(·|a)}[ V*(l+1, s') ]
                      π*(l,s,·) = argmax_π ⟨π, Q*⟩ − α Ω(π;ref)        (Eq 6)
                      V*(l,s)   = ⟨π*, Q*⟩ − α Ω(π*;ref)               (Eq 5)
    The last layer is terminal (no bootstrap). `ref` is the reference policy π_ref:
    None (uniform over actions, default), a (NA,) vector, or a (depth, SN, NA) array.
    `reg` optionally overrides REG[reg_key] with a custom Regularizer (e.g. make_adiv(a)).
    """
    R = reg if reg is not None else REG[reg_key]
    depth = r.shape[0]
    ref = resolve_ref(ref, depth)
    Qs = np.zeros((depth, SN, NA))
    pistar = np.zeros((depth, SN, NA))
    Vs = np.zeros((depth, SN))
    Vp = np.zeros((depth, SN))
    B = np.zeros((depth, SN))
    boundary = False

    def EV(V, l, a):
        if l == depth - 1:
            return 0.0
        return float(np.dot(transP(a, eps), V[l + 1]))

    # --- optimal pass: build Q*, π*, V* backward ---
    for l in range(depth - 1, -1, -1):
        for s in range(SN):
            q = np.array([r[l, s, a] + gamma * EV(Vs, l, a) for a in range(NA)])
            Qs[l, s] = q
            ps = R.argmax(q, alpha, ref[l, s])
            if ps.min() < 1e-7:
                boundary = True
            pistar[l, s] = ps
            Vs[l, s] = float(np.dot(ps, q) - alpha * R.omega(np.maximum(ps, 1e-12), ref[l, s]))

    # --- evaluation pass for the supplied behaviour policy `pis` + Bregman gap ---
    for l in range(depth - 1, -1, -1):
        for s in range(SN):
            q = np.array([r[l, s, a] + gamma * EV(Vp, l, a) for a in range(NA)])
            p = pis[l, s]
            Vp[l, s] = float(np.dot(p, q) - alpha * R.omega(p, ref[l, s]))
            B[l, s] = R.breg(p, np.maximum(pistar[l, s], 1e-12), ref[l, s])

    return DPSolution(Qs, pistar, Vs, Vp, B, boundary)
