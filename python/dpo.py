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

# Sign of the inner term C_Ω in the trajectory score. Telescoping the Eq-11 implicit reward
# (β(s) = V* − αC_Ω, i.e. inner_term.py Eq 14: r = α[∇Ω]_a + β − γα·E[C_Ω(s')] − γE[β(s')]) puts
# C_Ω into the trajectory score with a NEGATIVE coefficient:
#     score(τ) = Σ_t α[∇Ω(π)]_{a_t}  −  α Σ_t C_Ω(π(·|s_{t+1})).
# (γ dropped: the score is undiscounted, consistent with the chosen-action term — the γ→1 limit
#  of Eq 14.)  KL is unaffected: Φ_KL≡1 is constant ⇒ cancels in d=S_w−S_l, and Φ'_KL≡0 zeroes
#  its inner gradient, so the sign is invisible to KL.
INNER_SIGN = -1.0


@dataclass
class TrainConfig:
    gamma: float = 0.9
    steps: int = 300
    lr: float = 0.08
    batch: int = 16
    n_mc: int = 2
    # inner-term / data regime (see the three categories):
    #   "off"    — category 1 (pure off-policy): use the single logged next action a'_data; no
    #              resampling, n_mc ignored.
    #   "off_on" — category 3 (off-on-policy / Dyna): off-policy data states, but a' is FRESHLY
    #              resampled (n_mc draws) from the current π_θ(·|s') at training time.
    #   "on"     — category 2 (on-policy): a lagged sampler (snapshot of π_θ) rolls out the
    #              trajectories; at each state it draws n_mc actions, STORES them with the chosen
    #              index, and reuses those stored {a_j} both to continue the trajectory and to
    #              estimate C_Ω.  n_mc=1 reduces to an ordinary (s,a,s',a',…) trajectory.
    policy_mode: str = "off_on"
    resample_every: int = 25     # "on": re-snapshot π_θ → sampler and regenerate the batch every k steps
    n_pairs: int = 600           # "on": number of preference pairs regenerated per snapshot


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


def rollout_onpolicy(rewards, eps, gamma, rng, sampler, n_mc):
    """Category-2 on-policy rollout from a (lagged) sampler snapshot `sampler` (depth,SN,NA).
    At each state draw n_mc actions {a_j}~sampler(·|s), pick one (index i) to take, and STORE the
    whole set with i.  The stored {a_j} are reused at training time to estimate C_Ω(π(·|s)).
    Returns steps (l, s, acts, i, sn) and the discounted return.  n_mc=1 ⇒ ordinary trajectory."""
    depth = rewards.shape[0]
    s = int(rng.integers(SN))
    trans = []
    Rret = 0.0
    for l in range(depth):
        p = sampler[l, s] / sampler[l, s].sum()
        acts = rng.choice(NA, size=n_mc, p=p)
        i = int(rng.integers(n_mc))
        a = int(acts[i])
        Rret += (gamma ** l) * rewards[l, s, a]
        sn = _sample_cat(transP(a, eps), rng) if l < depth - 1 else -1
        trans.append((l, s, acts, i, sn))
        s = sn
    return trans, Rret


def make_dataset_onpolicy(rewards, eps, gamma, rng, n, sampler, n_mc):
    """Bradley–Terry pairs over category-2 on-policy trajectories rolled out from a lagged sampler."""
    data = []
    for _ in range(n):
        t1, R1 = rollout_onpolicy(rewards, eps, gamma, rng, sampler, n_mc)
        t2, R2 = rollout_onpolicy(rewards, eps, gamma, rng, sampler, n_mc)
        pw = 1.0 / (1.0 + np.exp(-(R1 - R2)))
        data.append((t1, t2) if rng.random() < pw else (t2, t1))
    return data


def train_one(reg_key: str, rewards: np.ndarray, alpha: float, data, cfg: TrainConfig,
              eps: float, rng: np.random.Generator, ref=None, occ=None, reg=None):
    """Train tabular θ on the preference data; returns (curve, final_gap, policy).
    `ref` is the reference policy π_ref (None=uniform over actions, (NA,) vector, or
    (depth,SN,NA) array) and is respected everywhere Ω / ∇Ω / C_Ω appear.
    `occ` (depth,SN) weights the TV gap by state-occupancy (paper Eq 20); None ⇒ uniform.
    `reg` optionally overrides REG[reg_key] with a custom Regularizer (e.g. make_adiv(a) for the
    α-divergence parameter sweep); the recovery target π* is then solved with that same reg."""
    R = reg if reg is not None else REG[reg_key]
    depth = rewards.shape[0]
    ref = resolve_ref(ref, depth)
    sol = solve_dp(reg_key, rewards, uniform_pis(depth), alpha, cfg.gamma, eps, ref, reg=reg)
    tgt = sol.pistar                                   # recovery target π*
    W = float(occ.sum()) if occ is not None else (depth * SN)

    th = rng.uniform(-0.05, 0.05, size=(depth, SN, NA))
    m = np.zeros((depth, SN, NA))
    v = np.zeros((depth, SN, NA))

    def pol_at(l, s):
        return softmax3(th[l, s], 1.0)

    def snapshot():
        return np.array([[pol_at(l, s) for s in range(SN)] for l in range(depth)])

    def gap():
        g = 0.0
        for l in range(depth):
            for s in range(SN):
                w = 1.0 if occ is None else occ[l, s]
                g += w * 0.5 * np.abs(pol_at(l, s) - tgt[l, s]).sum()
        return g / W

    # trajectory step: 4-tuple (l,s,a,sn) [off / off_on] or 5-tuple (l,s,acts,i,sn) [on].
    # unpack → (l, s, chosen a, sn, stored acts | None).
    def unpack(step):
        if len(step) == 5:
            l, s, acts, i, sn = step
            acts = np.asarray(acts, dtype=int)
            return l, s, int(acts[i]), sn, acts
        l, s, a, sn = step
        return l, s, a, sn, None

    # ---- C_Ω(π(·|s')) over a given action set — the ONLY inner formula (KL→const by Φ_KL≡1). ----
    def inner_value(pol, acts, rf):
        if cfg.policy_mode == "exact":            # closed form: C_Ω = E_{a~π}[Φ(u_a)] over ALL a' (tabular)
            if R.is_euc:
                return float(np.sum(pol * (pol - rf))) - R.omega(pol, rf), None
            uu = np.maximum(pol, 1e-12) / rf
            return float(np.sum(pol * R.Phi(uu))), None
        acts = np.asarray(acts, dtype=int)
        if acts.size == 0:
            return 0.0, acts
        if R.is_euc:
            return float(np.mean(pol[acts] - rf[acts])) - R.omega(pol, rf), acts
        uu = np.maximum(pol[acts], 1e-12) / rf[acts]
        return float(np.mean(R.Phi(uu))), acts

    # ---- which actions feed C at the next state s', per policy_mode ----
    #   exact  : all a' (closed-form C_Ω; no sampling — tabular only)   (A6)
    #   off    : the single logged next action a'_data         (category 1)
    #   on     : the stored {a_j} the sampler drew at s'        (category 2, reused from generation)
    #   off_on : n_mc fresh draws from the CURRENT π_θ(·|s')    (category 3, Dyna)
    def inner_actions(trans, i_t, pol_next):
        if cfg.policy_mode == "exact":
            return list(range(NA))
        if cfg.policy_mode == "off":
            a_next = unpack(trans[i_t + 1])[2] if i_t + 1 < len(trans) else None
            return [] if a_next is None else [a_next]
        if cfg.policy_mode == "on":
            return unpack(trans[i_t + 1])[4]
        return rng.choice(NA, size=cfg.n_mc, p=pol_next / pol_next.sum())

    def score_traj(trans, pols):
        sc = 0.0
        inner = []
        for i_t, step in enumerate(trans):
            l, s, a, sn, _ = unpack(step)
            sc += alpha * R.grad_dpo(pols[l][s], ref[l, s])[a]
            if sn >= 0:
                acts_used = inner_actions(trans, i_t, pols[l + 1][sn])
                C, used = inner_value(pols[l + 1][sn], acts_used, ref[l + 1, sn])
                sc += alpha * INNER_SIGN * C
                inner.append((sn, l, used))
        return sc, inner

    # "on" (category 2): a lagged sampler π_θ rolls out the trajectories; the stored {a_j} are reused
    # for C. Regenerate the batch every `resample_every` steps from a fresh snapshot (the lag);
    # any `data` passed in is ignored in this mode.
    if cfg.policy_mode == "on":
        data = make_dataset_onpolicy(rewards, eps, cfg.gamma, rng, cfg.n_pairs, snapshot(), cfg.n_mc)

    curve = []
    t = 0
    for step in range(cfg.steps):
        if cfg.policy_mode == "on" and cfg.resample_every > 0 and step > 0 and step % cfg.resample_every == 0:
            data = make_dataset_onpolicy(rewards, eps, cfg.gamma, rng, cfg.n_pairs, snapshot(), cfg.n_mc)
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
                for step_ in trans:
                    l, s, a, _sn, _ = unpack(step_)
                    pol = pols[l][s]
                    rf = ref[l, s]
                    for bb in range(NA):
                        grad[l, s, bb] += coef * sign * alpha * R.d_g_dtheta(pol, a, bb, rf)

            acc_outer(tw, +1)
            acc_outer(tl, -1)

            # gradient through C_Ω(s') via the SAME a' samples (Φ') — always computed (the only
            # mode). KL has Φ'(u)≡0, so its inner gradient is exactly 0 — computed, not skipped.
            def acc_inner(inner, sign):
                for (sn, l, acts) in inner:
                    pol = pols[l + 1][sn]
                    rf = ref[l + 1, sn]
                    g = grad[l + 1, sn]
                    if cfg.policy_mode == "exact":     # exact ∂C/∂θ_b = π_b (h_b − E_π[h]), all a'
                        if R.is_euc:                   # C = Σ_a π_a(π_a−ref_a) − Ω ; h_a = 2π_a − ref_a
                            k = 2.0 * pol - rf
                            g += coef * sign * alpha * INNER_SIGN * (pol * (k - float(np.sum(pol * k))))
                            for bb in range(NA):       # −∂Ω/∂θ_b (exact, over all actions)
                                dO = sum((pol[a2] - rf[a2]) * pol[a2] * ((1.0 if a2 == bb else 0.0) - pol[bb])
                                         for a2 in range(NA))
                                g[bb] += coef * sign * alpha * INNER_SIGN * (-dO)
                        else:                          # h_a = Φ(u_a) + π_a Φ'(u_a)/ref_a
                            uu = np.maximum(pol, 1e-12) / rf
                            h = R.Phi(uu) + pol * R.dPhi(uu) / rf
                            g += coef * sign * alpha * INNER_SIGN * (pol * (h - float(np.sum(pol * h))))
                        continue
                    acts = np.asarray(acts)
                    if acts.size == 0:
                        continue
                    nz = acts.size
                    g = grad[l + 1, sn]                    # view; modified in place
                    # ∂/∂θ_b of α·(1/n)Σ_aj integrand(u_aj):  scatter coeff to aj, minus pol·Σcoeff
                    if R.is_euc:
                        coeff = coef * sign * alpha * INNER_SIGN * (1.0 / nz) * pol[acts]
                        np.add.at(g, acts, coeff)
                        g -= pol * coeff.sum()
                        for bb in range(NA):              # −∂Ω/∂θ_b (over all actions, not samples)
                            dO = sum((pol[a2] - rf[a2]) * pol[a2] * ((1.0 if a2 == bb else 0.0) - pol[bb]) for a2 in range(NA))
                            g[bb] += coef * sign * alpha * INNER_SIGN * (-dO)
                    else:
                        uu = np.maximum(pol[acts], 1e-12) / rf[acts]
                        w = R.dPhi(uu) * (1.0 / rf[acts]) / nz     # ∂Φ(u_aj)/∂π via u=π/ref
                        coeff = coef * sign * alpha * INNER_SIGN * w * pol[acts]
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
