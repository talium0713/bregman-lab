"""
THE INNER TERM  C_Ω(s')  —  the heart of the paper, and what you asked to verify.

    r(s,a) = α[∇Ω(π*(·|s))]_a + β(s) − γα·E_{s'}[ C_Ω(π*(·|s'); π_ref(·|s')) ] − γ E_{s'}[β(s')]   (Eq 14)
    C_Ω(π;π_ref) = E_{a~π}[ [∇Ω(π)]_a ] − Ω(π) = E_{a~π}[Φ(u_a)],   Φ(u)=f'(u)−f(u)/u,  u=π/ref   (Eq 13)

FAIRNESS: KL is estimated by the SAME fully-sampled procedure as every other Ω, with NOTHING
hardcoded.  The per-sample integrand Φ(u) = f'(u) − f(u)/u is COMPUTED from f and f'
(regularizers.Regularizer.Phi), so KL is sampled exactly like the rest and its draws merely
happen to evaluate to the constant 1 — the constancy is arithmetic, never a literal.
`is_admissible` verifies the resulting Var_a[Φ]≈0 empirically.

Estimators provided for inspection:
  1. C_exact            closed-form C_Ω (Eq 13).
  2. C_hat_phi          THE (only) estimator: mean_i Φ(u_{a_i}), a_i~π, with Φ computed from f,f'.
                        Fully sample-based for all Ω.  For KL every sampled Φ equals 1
                        (arithmetically) ⇒ variance is exactly 0 — KL is "always sampling, but the
                        value is constant", which is precisely the admissibility signature.
  3. C_hat_offpolicy    extreme off-policy: one logged action a'_data (behaviour policy), no resample.
  4. C_hat_offpolicy    extreme off-policy: one logged action a'_data (behaviour policy), no resample.
  5. c_violation        the "c(s') transformation" probe: how an action-indexed c(a') fails to
                        reduce to a state-indexed c(s') under stochastic (ε>0) transitions.
  6. single_state_std   §4.2 single-state std(Ĉ) vs n, for either estimator.
"""
from __future__ import annotations

import numpy as np

from regularizers import REG, _as_ref
from mdp import transP, SN, NA


def C_exact(reg_key: str, p, ref=None) -> float:
    """Closed-form C_Ω(π) of Eq 13. Comes out 1.0 for KL via the math (no special case)."""
    return REG[reg_key].C(np.asarray(p, float), ref)


def _phi_value(reg_key: str, p, a: int, ref) -> float:
    """Per-sample inner integrand at action a: Φ(u_a) (f-div) or (π_a − ref_a) (euc)."""
    R = REG[reg_key]
    if R.is_euc:
        return float(p[a] - ref[a])
    return float(R.Phi(max(p[a], 1e-12) / ref[a]))


def C_hat_phi(reg_key: str, p, n: int, rng: np.random.Generator, ref=None) -> float:
    """PRIMARY estimator: mean_i Φ(u_{a_i}), a_i~π. KL ⇒ all samples = 1 ⇒ variance 0."""
    R = REG[reg_key]
    p = np.maximum(np.asarray(p, float), 1e-12)
    ref = _as_ref(ref, len(p))
    actions = rng.choice(len(p), size=n, p=p / p.sum())
    vals = np.array([_phi_value(reg_key, p, int(a), ref) for a in actions])
    if R.is_euc:
        return float(vals.mean() - R.omega(p, ref))
    return float(vals.mean())


def C_hat_offpolicy(reg_key: str, p, a_data, ref=None) -> float:
    """Extreme off-policy inner term: the single logged next action a'_data, no resampling."""
    R = REG[reg_key]
    p = np.maximum(np.asarray(p, float), 1e-12)
    ref = _as_ref(ref, len(p))
    if a_data is None:
        return 0.0
    if R.is_euc:
        return float((p[a_data] - ref[a_data]) - R.omega(p, ref))
    return float(R.Phi(p[a_data] / ref[a_data]))


# ──────────────────────────────────────────────────────────────────────────────────────
#  THE "c(s') TRANSFORMATION" PROBE  (port of `cViolation`).
#  Eq 16 folds the action-indexed inner quantity into a state-only c(s'); valid only when s'
#  pins down the generating action a (deterministic, ε=0). Under ε>0, observing s' leaves a
#  posterior post(a|s') ∝ π_behave(a)·P(s'|a), so an action-indexed value c_a retains posterior
#  variance that c(s') cannot represent. Returns the occupancy-weighted RMS of that std vs ε;
#  =0 at ε=0 (admissible), >0 for ε>0.
# ──────────────────────────────────────────────────────────────────────────────────────
def c_violation(eps: float, c_vals=(0.6, 1.0, 1.6), behave=(1 / 3, 1 / 3, 1 / 3)) -> float:
    c_vals = np.asarray(c_vals, float)
    pa = np.asarray(behave, float)
    psp = np.array([sum(pa[a] * transP(a, eps)[sp] for a in range(NA)) for sp in range(SN)])
    V = 0.0
    for sp in range(SN):
        w = np.array([pa[a] * transP(a, eps)[sp] for a in range(NA)])
        Z = w.sum()
        if Z <= 0:
            continue
        post = w / Z
        E = float(np.dot(post, c_vals))
        E2 = float(np.dot(post, c_vals * c_vals))
        V += psp[sp] * max(E2 - E * E, 0.0)
    return float(np.sqrt(V))


def single_state_variance(reg_key: str, ns, n_draws: int = 1500, n_actions: int = 100,
                          scale: float = 3.0, ref=None, seed: int = 0):
    """
    Paper §4.2 single-state experiment (Fig 3). One state, |A|=n_actions, π_ref uniform,
    π = softmax(N(0,1)·scale) (peak ≈0.24 at scale=3). For each n, the n-sample Φ-form estimator
    Ĉ = (1/n)Σ Φ(u_{a_i}); over n_draws draws report mean, std, and the closed-form exact C_Ω.
    KL: mean≡1, std≡0 at every n; non-KL std decays as 1/√n.
    """
    rng = np.random.default_rng(seed)
    R = REG[reg_key]
    logits = rng.standard_normal(n_actions) * scale
    p = np.exp(logits - logits.max()); p = p / p.sum()
    ref = _as_ref(ref, n_actions) if ref is not None else np.full(n_actions, 1.0 / n_actions)
    if R.is_euc:
        phi = p - ref; omega = float(0.5 * np.sum((p - ref) ** 2))
    else:
        phi = np.atleast_1d(R.Phi(p / ref)) * np.ones(n_actions); omega = 0.0
    cexact = R.C(p, ref)
    out = {}
    for n in ns:
        vals = np.empty(n_draws)
        for d in range(n_draws):
            a = rng.choice(n_actions, size=int(n), p=p)
            vals[d] = phi[a].mean() - omega
        out[int(n)] = {"mean": float(vals.mean()), "std": float(vals.std()), "exact": float(cexact)}
    return out


def trajectory_variance(reg_key: str, Hs, n_mc: int = 8, n_real: int = 8000, n_actions: int = 100,
                        scale: float = 1.0, ref=None, seed: int = 0):
    """
    Paper §4.2 trajectory experiment (Fig 4). A horizon-H rollout contributes H−1 independent
    inner-term draws to the per-trajectory implicit reward; each draw is an n_mc-sample Φ-form
    estimate of C_Ω at a peaked policy. We report the mean and per-trajectory std of the sum
    S = Σ_{t=1}^{H-1} Ĉ_Ω over n_real Monte-Carlo realizations, plus the exact (H−1)·C_Ω.
    KL: S = H−1 exactly (std 0); non-KL std grows as √(H−1).
    """
    rng = np.random.default_rng(seed)
    R = REG[reg_key]
    logits = rng.standard_normal(n_actions) * scale
    p = np.exp(logits - logits.max()); p = p / p.sum()
    ref = _as_ref(ref, n_actions) if ref is not None else np.full(n_actions, 1.0 / n_actions)
    if R.is_euc:
        phi = p - ref; omega = float(0.5 * np.sum((p - ref) ** 2))
    else:
        phi = np.atleast_1d(R.Phi(p / ref)) * np.ones(n_actions); omega = 0.0
    cexact = float(R.C(p, ref))
    nstates = max(Hs) - 1
    out = {}
    if nstates <= 0:
        return {H: {"mean": 0.0, "std": 0.0, "exact": 0.0} for H in Hs}
    # Ĉ for n_real realizations × nstates independent next-states (each = mean of n_mc Φ samples)
    acts = rng.choice(n_actions, size=(n_real, nstates, n_mc), p=p)
    chat = phi[acts].mean(axis=2) - omega                    # (n_real, nstates)
    for H in Hs:
        k = max(H - 1, 0)
        if k == 0:
            out[H] = {"mean": 0.0, "std": 0.0, "exact": 0.0}; continue
        S = chat[:, :k].sum(axis=1)
        out[H] = {"mean": float(S.mean()), "std": float(S.std()), "exact": float(k * cexact)}
    return out


def single_state_std(reg_key: str, ns, n_draws: int = 500, n_actions: int = 100,
                     scale: float = 3.0, ref=None, seed: int = 0):
    """
    §4.2 single-state probe with the Φ-form estimator. |A| = n_actions, π = softmax(N(0,1)·scale).
    For each n in `ns`, std over n_draws of the n-sample estimator Ĉ = (1/n)Σ Φ(u_{a_i}).
    `ref` defaults to uniform.  Returns {n: std}.  For KL this is identically 0 (Φ_KL≡1, computed).
    """
    rng = np.random.default_rng(seed)
    R = REG[reg_key]
    logits = rng.standard_normal(n_actions) * scale
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    ref = _as_ref(ref, n_actions) if ref is not None else np.full(n_actions, 1.0 / n_actions)

    if R.is_euc:
        phi = p - ref
        omega = float(0.5 * np.sum((p - ref) ** 2))
    else:
        phi = np.atleast_1d(R.Phi(p / ref)) * np.ones(n_actions)   # Φ computed; KL → constant
        omega = 0.0

    out = {}
    for n in ns:
        vals = np.empty(n_draws)
        for d in range(n_draws):
            a = rng.choice(n_actions, size=int(n), p=p)
            vals[d] = phi[a].mean() - omega
        out[int(n)] = float(vals.std())
    return out
