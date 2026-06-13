"""
Trajectory-DPO training under a chosen regularizer (port of `makeDataset` + `trainOne`).

Preference data: a behaviour policy (uniform, i.e. OFF-policy w.r.t. every π*) rolls out
trajectories; a Bradley–Terry oracle on raw discounted returns labels winner/loser pairs.

Training fits tabular logits θ (policy = softmax(θ)) to the trajectory-DPO loss. The implicit
reward of a trajectory is the telescoped sum (Eq 16); the β(s) potential terms cancel inside
each winner−loser difference, leaving two pieces per transition:

    score(τ) = Σ_t [  α·[∇Ω(π(·|s_t))]_{a_t}        # chosen-action term (always exact)
                    + α·C_Ω(π(·|s_{t+1}))  ]         # inner term  (constant for KL; MC for others)

    d = score(τ_w) − score(τ_l),   loss = −log σ(d),   dℓ/dd = −σ(−d).

GRADIENT MODES for the inner term (the experimental knob, see README):
  A (faithful): backprop ∂Ĉ_Ω/∂θ through the SAME sampled a' actions, using Φ'(u).  This is
                the theoretically correct trajectory-DPO gradient.
  B (approx):   stop-grad on ∂Ĉ/∂θ; the inner-term noise enters only through the scalar d.

INNER-TERM SAMPLING:
  on-policy  : draw n_mc fresh actions a' ~ π(·|s')   (Eq 15 estimator; n_mc matters).
  off-policy : use the single logged next action a'_data, no resampling (n_mc ignored).

KL is NOT special-cased. It runs the same estimator as everyone else; because Φ_KL(u)≡1 every
inner draw equals the constant 1 (variance 0) and cancels in d (equal #inner-terms per
fixed-depth trajectory), and Φ'_KL≡0 zeroes its A-direction inner gradient — exactly noise-free
at every n_mc and in both A/B modes, by the math rather than by a flag.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from regularizers import REG, softmax3
from mdp import transP, solve_dp, uniform_pis, resolve_ref, SN, NA


@dataclass
class TrainConfig:
    gamma: float = 0.9
    steps: int = 300
    lr: float = 0.08
    batch: int = 16
    n_mc: int = 2
    grad_mode: str = "A"        # "A" faithful | "B" scale-only
    off_policy: bool = False


def _sample_cat(p: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(p), p=p / p.sum()))


def rollout(rewards: np.ndarray, eps: float, gamma: float, rng: np.random.Generator):
    """One behaviour-policy (uniform-action) trajectory; returns transitions + discounted return."""
    depth = rewards.shape[0]
    s = int(rng.integers(SN))
    trans = []
    Rret = 0.0
    for l in range(depth):
        a = int(rng.integers(NA))
        Rret += (gamma ** l) * rewards[l, s, a]
        sn = _sample_cat(transP(a, eps), rng) if l < depth - 1 else -1
        trans.append((l, s, a, sn))
        s = sn
    return trans, Rret


def make_dataset(rewards: np.ndarray, eps: float, gamma: float, rng: np.random.Generator,
                 n: int = 10000):
    """Bradley–Terry preference pairs over behaviour-policy trajectories."""
    data = []
    for _ in range(n):
        t1, R1 = rollout(rewards, eps, gamma, rng)
        t2, R2 = rollout(rewards, eps, gamma, rng)
        pw = 1.0 / (1.0 + np.exp(-(R1 - R2)))
        if rng.random() < pw:
            data.append((t1, t2))      # (winner, loser)
        else:
            data.append((t2, t1))
    return data


def rollout_policy(rewards, eps, gamma, rng, policy):
    """One trajectory rolled out from `policy` (depth,SN,NA). For §4.3 on-policy data = π*_Ω."""
    depth = rewards.shape[0]
    s = int(rng.integers(SN))
    trans = []
    Rret = 0.0
    for l in range(depth):
        a = int(rng.choice(NA, p=policy[l, s] / policy[l, s].sum()))
        Rret += (gamma ** l) * rewards[l, s, a]
        sn = _sample_cat(transP(a, eps), rng) if l < depth - 1 else -1
        trans.append((l, s, a, sn))
        s = sn
    return trans, Rret


def make_dataset_policy(rewards, eps, gamma, rng, n, policy):
    """Bradley–Terry preference pairs over trajectories rolled out from `policy`
    (§4.3 on-policy = π*_Ω rollouts; the uniform make_dataset is the off-policy = π_ref case)."""
    data = []
    for _ in range(n):
        t1, R1 = rollout_policy(rewards, eps, gamma, rng, policy)
        t2, R2 = rollout_policy(rewards, eps, gamma, rng, policy)
        pw = 1.0 / (1.0 + np.exp(-(R1 - R2)))
        data.append((t1, t2) if rng.random() < pw else (t2, t1))
    return data


def train_one(reg_key: str, rewards: np.ndarray, alpha: float, data, cfg: TrainConfig,
              eps: float, rng: np.random.Generator, ref=None, occ=None):
    """Train tabular θ on the preference data; returns (curve, final_gap, policy).
    `ref` is the reference policy π_ref (None=uniform over actions, (NA,) vector, or
    (depth,SN,NA) array) and is respected everywhere Ω / ∇Ω / C_Ω appear.
    `occ` (depth,SN) weights the TV gap by state-occupancy (paper Eq 20); None ⇒ uniform."""
    R = REG[reg_key]
    depth = rewards.shape[0]
    ref = resolve_ref(ref, depth)
    sol = solve_dp(reg_key, rewards, uniform_pis(depth), alpha, cfg.gamma, eps, ref)
    tgt = sol.pistar                                   # recovery target π*
    W = float(occ.sum()) if occ is not None else (depth * SN)

    th = rng.uniform(-0.05, 0.05, size=(depth, SN, NA))
    m = np.zeros((depth, SN, NA))
    v = np.zeros((depth, SN, NA))

    def pol_at(l, s):
        return softmax3(th[l, s], 1.0)

    def gap():
        g = 0.0
        for l in range(depth):
            for s in range(SN):
                w = 1.0 if occ is None else occ[l, s]
                g += w * 0.5 * np.abs(pol_at(l, s) - tgt[l, s]).sum()
        return g / W

    # ---- inner term C_Ω(s'); returns value and the sampled a' (for the A-direction grad).
    #      NO KL special case: KL flows through the SAME estimator. Φ_KL(u)≡1 makes every draw
    #      equal to the constant 1 (variance 0), and that constant cancels in d = S_w − S_l
    #      because every fixed-depth trajectory has the same number of inner terms. ----
    def draw_inner(pol, a_data, rf):
        u = lambda a: max(pol[a], 1e-12) / rf[a]
        if cfg.off_policy:
            if a_data is None:
                return 0.0, []
            if R.is_euc:
                return float((pol[a_data] - rf[a_data]) - R.omega(pol, rf)), [a_data]
            return float(R.Phi(u(a_data))), [a_data]
        # on-policy: n_mc fresh draws from π(·|s'); vectorized over n_mc (fast for large budgets)
        acts = rng.choice(NA, size=cfg.n_mc, p=pol / pol.sum())
        if R.is_euc:
            e = float(np.mean(pol[acts] - rf[acts]))
            return e - R.omega(pol, rf), acts
        uu = np.maximum(pol[acts], 1e-12) / rf[acts]
        return float(np.mean(R.Phi(uu))), acts

    def score_traj(trans, pols):
        sc = 0.0
        inner = []
        for i, (l, s, a, sn) in enumerate(trans):
            sc += alpha * R.grad_dpo(pols[l][s], ref[l, s])[a]
            if sn >= 0:
                a_data = trans[i + 1][2] if i + 1 < len(trans) else None
                C, acts = draw_inner(pols[l + 1][sn], a_data, ref[l + 1, sn])
                sc += alpha * C
                inner.append((sn, l, acts))
        return sc, inner

    curve = []
    t = 0
    for step in range(cfg.steps):
        pols = [[pol_at(l, s) for s in range(SN)] for l in range(depth)]
        grad = np.zeros((depth, SN, NA))
        for _ in range(cfg.batch):
            tw, tl = data[int(rng.integers(len(data)))]
            sw, iw = score_traj(tw, pols)
            sl, il = score_traj(tl, pols)
            d = sw - sl
            coef = -1.0 / (1.0 + np.exp(d))            # dℓ/dd = −σ(−d)

            # outer gradient: through the chosen-action implicit reward [∇Ω]_a
            def acc_outer(trans, sign):
                for (l, s, a, _sn) in trans:
                    pol = pols[l][s]
                    rf = ref[l, s]
                    for bb in range(NA):
                        grad[l, s, bb] += coef * sign * alpha * R.d_g_dtheta(pol, a, bb, rf)

            acc_outer(tw, +1)
            acc_outer(tl, -1)

            # A-direction: gradient through C_Ω(s') via the SAME a' samples (Φ').
            # No KL guard: KL has Φ'(u)≡0, so its inner gradient is exactly 0 — computed, not skipped.
            if cfg.grad_mode == "A":
                def acc_inner(inner, sign):
                    for (sn, l, acts) in inner:
                        acts = np.asarray(acts)
                        if acts.size == 0:
                            continue
                        pol = pols[l + 1][sn]
                        rf = ref[l + 1, sn]
                        nz = acts.size
                        g = grad[l + 1, sn]                    # view; modified in place
                        # ∂/∂θ_b of α·(1/n)Σ_aj integrand(u_aj):  scatter coeff to aj, minus pol·Σcoeff
                        if R.is_euc:
                            coeff = coef * sign * alpha * (1.0 / nz) * pol[acts]
                            np.add.at(g, acts, coeff)
                            g -= pol * coeff.sum()
                            for bb in range(NA):              # −∂Ω/∂θ_b (over all actions, not samples)
                                dO = sum((pol[a2] - rf[a2]) * pol[a2] * ((1.0 if a2 == bb else 0.0) - pol[bb]) for a2 in range(NA))
                                g[bb] += coef * sign * alpha * (-dO)
                        else:
                            uu = np.maximum(pol[acts], 1e-12) / rf[acts]
                            w = R.dPhi(uu) * (1.0 / rf[acts]) / nz     # ∂Φ(u_aj)/∂π via u=π/ref
                            coeff = coef * sign * alpha * w * pol[acts]
                            np.add.at(g, acts, coeff)
                            g -= pol * coeff.sum()

                acc_inner(iw, +1)
                acc_inner(il, -1)

        # Adam update
        t += 1
        g = grad / cfg.batch
        m[:] = 0.9 * m + 0.1 * g
        v[:] = 0.999 * v + 0.001 * g * g
        mh = m / (1 - 0.9 ** t)
        vh = v / (1 - 0.999 ** t)
        th[:] -= cfg.lr * mh / (np.sqrt(vh) + 1e-8)

        if step % 15 == 0 or step == cfg.steps - 1:
            curve.append(gap())

    pol = np.array([[pol_at(l, s) for s in range(SN)] for l in range(depth)])
    return curve, gap(), pol
