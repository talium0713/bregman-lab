"""
Seven separable strictly-convex regularizers Ω(π ; π_ref) on the |A|-action simplex.

DESIGN PRINCIPLES (per the two requests):

1. NO KL SHORT-CIRCUIT.  There is no `const_C` / `if KL: return constant` anywhere.  Every
   divergence — KL included — flows through the *same* code: the same soft-argmax, the same
   inner-term integrand Φ, the same Monte-Carlo estimator (see inner_term.py / dpo.py).  KL's
   off-policy admissibility is therefore not asserted; it EMERGES, because KL is the unique f
   whose per-action inner integrand Φ(u) = f'(u) − f(u)/u is constant (≡ 1) and whose Φ'(u) ≡ 0.
   Drawing n_mc samples of a constant function returns the constant with zero variance — the
   "no-op" the paper describes (§4.1).  `is_admissible()` below CHECKS this empirically instead
   of hardcoding it.

2. ARBITRARY π_ref.  `π_ref` defaults to uniform over actions, but every quantity accepts a
   reference vector `ref` (length |A|, sums to 1) and respects it.  With u_a := π_a / ref_a:

        f-divergence:   Ω(π;ref) = Σ_a ref_a f(u_a),   [∇Ω]_a = f'(u_a)
        Bregman (euc):  Ω(π;ref) = ½ Σ_a (π_a − ref_a)²

   The inner integrand Φ(u) is a function of u = π/ref ALONE (because Ω = E_{a~π}[f(u_a)/u_a]),
   so the divergence-specific Φ closed forms below are ref-independent; only u changes with ref.

Paper map (Off_policy_admissibility.pdf):
  omega/grad/breg     Eq (5)/(6) penalty, reward shape, policy-deviation Bregman
  argmax(Q,α,ref)     Eq (6) soft-argmax  π* = argmax ⟨π,Q⟩ − α Ω(π;ref)
  C(π,ref)            Eq (13) inner term  C_Ω = E_{a~π}[f'(u_a)] − Ω = E_{a~π}[Φ(u_a)]
  Phi / dPhi          per-action integrand Φ(u)=f'(u)−f(u)/u  and Φ'(u)
  d_g_dtheta          ∂/∂θ_b of [∇Ω]_a with π = softmax(θ)   (outer DPO gradient)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

EPS_CLAMP = 1e-12


def _lg(x):
    return np.log(np.maximum(x, EPS_CLAMP))


def uniform_ref(na: int) -> np.ndarray:
    return np.full(na, 1.0 / na)


def _as_ref(ref, na: int) -> np.ndarray:
    """Default π_ref = uniform over actions; otherwise validate and normalize the supplied ref."""
    if ref is None:
        return uniform_ref(na)
    ref = np.asarray(ref, dtype=float)
    if ref.shape != (na,):
        raise ValueError(f"ref must have shape ({na},), got {ref.shape}")
    s = ref.sum()
    return ref / s if abs(s - 1.0) > 1e-9 else ref


# ──────────────────────────────────────────────────────────────────────────────────────
# Per-divergence scalar building blocks: f, f', f'' (of t = u), the inverse marginal
# (f')^{-1}(y) = t  used by the unified soft-argmax, and the inner integrand Φ, Φ'.
# `euc` is a Bregman (not f-) divergence and is handled with type="euc".
# ──────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Spec:
    key: str
    label: str
    type: str                              # "fdiv" or "euc"
    f: Optional[Callable] = None
    fp: Optional[Callable] = None
    fpp: Optional[Callable] = None
    inv: Optional[Callable] = None         # (f')^{-1}(y) -> t (>=0), with domain clamps
    nu_bracket: Optional[Callable] = None  # (Q, alpha) -> (nu_lo, nu_hi) for the root search
    # NOTE: there are deliberately NO Phi/dPhi fields. The inner integrand Φ(u)=f'(u)−f(u)/u and
    # its derivative are COMPUTED from f, f', f'' (see Regularizer.Phi/dPhi), so KL's constant
    # Φ≡1 is an arithmetic result, never a hardcoded literal.


# ── α-divergence family, parameterized by a (the "α" of the Amari α-divergence) ──
#   f_a(t) = (t^a − a t + a − 1) / (a(a−1)),   f'_a(t) = (t^{a−1} − 1)/(a−1),   f''_a(t) = t^{a−2}
#   a → 1  ⇒  KL   (t ln t);     a → 0  ⇒  Reverse-KL.   So a ∈ (0,1) sits "between KL and RKL".
# DEFAULT is the midpoint a = 0.5; pass a different value via make_adiv(a) (e.g. the morph sweep).
DEFAULT_ADIV_A = 0.5


def _adiv_inv(y, a):
    """(f'_a)^{-1}(y) = t,  with the domain handled so the soft-argmax root search stays valid."""
    e = a - 1.0
    base = 1.0 + e * y
    if e > 0:                                   # a > 1: base may go negative ⇒ action excluded
        return np.maximum(base, 0.0) ** (1.0 / e)
    base = np.maximum(base, 1e-12)              # a < 1: exponent<0, base→0⁺ ⇒ t→∞ (drives ν up)
    return base ** (1.0 / e)


def _adiv_spec(a: float) -> "Spec":
    if a == 0.0 or a == 1.0:
        raise ValueError("α-div parameter a must avoid 0 (=RKL) and 1 (=KL); use those keys directly.")
    e, norm = a - 1.0, a * (a - 1.0)
    return Spec(
        "adiv", f"α-div (a={a:g})", "fdiv",
        f=lambda t, a=a, n=norm: (t ** a - a * t + a - 1.0) / n,
        fp=lambda t, e=e: (t ** e - 1.0) / e,
        fpp=lambda t, a=a: t ** (a - 2.0),
        inv=lambda y, a=a: _adiv_inv(y, a),
        nu_bracket=None,                        # generic expanding bracket handles any a
    )


_SPECS = {
    "kl": Spec(
        "kl", "KL", "fdiv",
        f=lambda t: t * _lg(t), fp=lambda t: _lg(t) + 1, fpp=lambda t: 1.0 / t,
        inv=lambda y: np.exp(y - 1.0),
    ),
    "adiv": _adiv_spec(DEFAULT_ADIV_A),
    "rkl": Spec(
        "rkl", "Reverse KL", "fdiv",
        f=lambda t: -_lg(t), fp=lambda t: -1.0 / t, fpp=lambda t: 1.0 / (t * t),
        inv=lambda y: -1.0 / np.minimum(y, -1e-12),
        nu_bracket=lambda Q, al: (Q.max() + 1e-10, Q.max() + 1000 * al + 10),
    ),
    "js": Spec(
        "js", "JS", "fdiv",
        f=lambda t: t * _lg(2 * t / (1 + t)) + _lg(2 / (1 + t)), fp=lambda t: _lg(2 * t / (1 + t)),
        fpp=lambda t: 1.0 / (t * (1 + t)),
        inv=lambda y: (lambda e: e / np.maximum(2 - e, EPS_CLAMP))(np.exp(np.minimum(y, 0.6931))),
        nu_bracket=lambda Q, al: (Q.max() - al * np.log(2) + 1e-9, Q.max() + 60 * al),
    ),
    "hel": Spec(
        "hel", "sq. Hellinger", "fdiv",
        f=lambda t: (np.sqrt(t) - 1) ** 2, fp=lambda t: 1 - 1.0 / np.sqrt(t),
        fpp=lambda t: 0.5 * t ** -1.5,
        inv=lambda y: 1.0 / (1.0 - np.minimum(y, 0.999999)) ** 2,
        nu_bracket=lambda Q, al: (Q.max() - al + 1e-9, Q.max() + 60 * al),
    ),
    "chi2": Spec(
        "chi2", "Pearson χ²", "fdiv",
        f=lambda t: (t - 1) ** 2, fp=lambda t: 2 * (t - 1), fpp=lambda t: 2.0,
        inv=lambda y: np.maximum(1 + y / 2, 0.0),
        nu_bracket=lambda Q, al: (Q.min() - 10 * al - 10, Q.max() + 10 * al + 10),
    ),
    "euc": Spec("euc", "sq. Euclidean", "euc"),
}

REGKEYS = ["kl", "adiv", "rkl", "js", "hel", "chi2", "euc"]

# Divergence palette + short labels — identical to the JS suite (src/core/math.js REG table),
# so every figure/HTML keeps the same colour per Ω.
COLORS = {"kl": "#e84393", "adiv": "#c455b8", "rkl": "#9b59d0", "js": "#6f6ae0",
          "hel": "#5585e0", "chi2": "#4fa8d8", "euc": "#5bc8d6"}
SHORT = {"kl": "KL", "adiv": "α-div", "rkl": "RKL", "js": "JS",
         "hel": "Hel", "chi2": "χ²", "euc": "Euc"}


# ──────────────────────────────────────────────────────────────────────────────────────
# Unified soft-argmax: π_a = ref_a · (f')^{-1}((Q_a − ν)/α),  ν chosen so Σ_a π_a = 1.
# (For euc, π_a = max(ref_a + (Q_a − ν)/α, 0).)  Σπ(ν) is monotone ↓ in ν, so we bracket
# and bisect on ν. This replaces all the per-divergence hand-tuned solvers and works for
# ANY ref, KL included (KL also has the closed form π_a ∝ ref_a exp(Q_a/α), used directly).
# ──────────────────────────────────────────────────────────────────────────────────────
def _pi_of_nu(spec: Spec, Q, alpha, ref, nu):
    if spec.type == "euc":
        pi = np.maximum(ref + (Q - nu) / alpha, 0.0)
    else:
        t = spec.inv((Q - nu) / alpha)
        pi = ref * np.maximum(t, 0.0)
    return pi


def _argmax(spec: Spec, Q, alpha, ref):
    Q = np.asarray(Q, dtype=float)
    if spec.key == "kl":                       # closed form (numerically cleanest)
        z = Q / alpha
        w = ref * np.exp(z - z.max())
        return w / w.sum()
    # bracket on ν then bisect Σπ(ν) − 1 = 0
    if spec.nu_bracket is not None:
        lo, hi = spec.nu_bracket(Q, alpha)
    else:
        lo, hi = Q.min() - 10 * alpha - 10, Q.max() + 10 * alpha + 10
    S = lambda nu: _pi_of_nu(spec, Q, alpha, ref, nu).sum() - 1.0
    # expand bracket until it straddles the root (robust to arbitrary ref)
    for _ in range(60):
        if S(lo) > 0:
            break
        lo -= (hi - lo)
    for _ in range(60):
        if S(hi) < 0:
            break
        hi += (hi - lo)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if S(mid) > 0:
            lo = mid
        else:
            hi = mid
    pi = _pi_of_nu(spec, Q, alpha, ref, 0.5 * (lo + hi))
    return pi / pi.sum()


# ──────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Regularizer:
    key: str
    label: str
    spec: Spec

    # ---- helpers ----
    def _u(self, p, ref):
        return np.asarray(p, float) / ref

    # ---- Ω, ∇Ω, Bregman, soft-argmax ----
    def omega(self, p, ref=None):
        p = np.maximum(np.asarray(p, float), EPS_CLAMP)
        ref = _as_ref(ref, len(p))
        if self.spec.type == "euc":
            return float(0.5 * np.sum((p - ref) ** 2))
        return float(np.sum(ref * self.spec.f(p / ref)))

    def grad(self, p, ref=None):
        """[∇_π Ω]_a = f'(u_a)  (for euc: π_a − ref_a). This is also the DPO chosen-action term."""
        p = np.maximum(np.asarray(p, float), EPS_CLAMP)
        ref = _as_ref(ref, len(p))
        if self.spec.type == "euc":
            return p - ref
        return self.spec.fp(p / ref)

    grad_dpo = grad  # the implicit-reward chosen-action term α·[∇Ω]_a is exactly α·grad

    def breg(self, p, q, ref=None):
        p = np.maximum(np.asarray(p, float), EPS_CLAMP)
        q = np.maximum(np.asarray(q, float), EPS_CLAMP)
        ref = _as_ref(ref, len(p))
        if self.spec.type == "euc":
            return float(0.5 * np.sum((p - q) ** 2))
        up, uq = p / ref, q / ref
        return float(np.sum(ref * (self.spec.f(up) - self.spec.f(uq) - self.spec.fp(uq) * (up - uq))))

    def argmax(self, Q, alpha, ref=None):
        Q = np.asarray(Q, float)
        ref = _as_ref(ref, len(Q))
        return _argmax(self.spec, Q, alpha, ref)

    # ---- inner term C_Ω (Eq 13), exact; NOT special-cased for KL ----
    def C(self, p, ref=None):
        p = np.maximum(np.asarray(p, float), 1e-9)
        ref = _as_ref(ref, len(p))
        if self.spec.type == "euc":
            return float(np.dot(p, p - ref) - self.omega(p, ref))
        u = p / ref
        return float(np.sum(p * self.Phi(u)))   # = E_{a~π}[Φ(u_a)]; equals 1 for KL by ARITHMETIC

    # ---- per-action inner integrand Φ(u) and Φ'(u), COMPUTED from f, f', f'' (u = π/ref) ----
    #      Φ(u)  = f'(u) − f(u)/u                       — for KL this evaluates to 1 (not a literal)
    #      Φ'(u) = f''(u) − f'(u)/u + f(u)/u²           — for KL this evaluates to 0
    def Phi(self, u):
        s = self.spec
        return s.fp(u) - s.f(u) / u

    def dPhi(self, u):
        s = self.spec
        return s.fpp(u) - s.fp(u) / u + s.f(u) / (u * u)

    # ---- outer DPO gradient: ∂/∂θ_b of [∇Ω]_a, π = softmax(θ) ----
    def d_g_dtheta(self, p, a, b, ref=None):
        p = np.asarray(p, float)
        ref = _as_ref(ref, len(p))
        delta = 1.0 if a == b else 0.0
        if self.spec.type == "euc":
            return float(p[a] * (delta - p[b]))
        # f'(u_a) with u_a = π_a/ref_a:  ∂/∂θ_b = f''(u_a)·(1/ref_a)·π_a(δ_ab − π_b)
        u_a = max(p[a], EPS_CLAMP) / ref[a]
        return float(self.spec.fpp(u_a) * (1.0 / ref[a]) * p[a] * (delta - p[b]))

    @property
    def is_euc(self):
        return self.spec.type == "euc"


REG: dict[str, Regularizer] = {k: Regularizer(k, s.label, s) for k, s in _SPECS.items()}


def make_adiv(a: float) -> "Regularizer":
    """Build an α-divergence Regularizer for an arbitrary parameter a (a∈(0,1) ⇒ between RKL & KL).
    The default REG['adiv'] uses a = DEFAULT_ADIV_A; use this to sweep a (e.g. the morph figure)."""
    s = _adiv_spec(a)
    return Regularizer("adiv", s.label, s)


# ──────────────────────────────────────────────────────────────────────────────────────
# Admissibility, CHECKED not hardcoded: Ω is off-policy admissible iff its inner integrand
# Φ(u) is constant in u (equivalently Var_{a~π}[Φ] = 0 for every π).  Returns True only for KL.
# ──────────────────────────────────────────────────────────────────────────────────────
def is_admissible(reg_key: str, ref=None, n_trials: int = 200, tol: float = 1e-9,
                  seed: int = 0) -> bool:
    rng = np.random.default_rng(seed)
    R = REG[reg_key]
    na = 3 if ref is None else len(ref)
    ref_v = _as_ref(ref, na)
    spread = 0.0
    for _ in range(n_trials):
        p = rng.dirichlet(np.ones(na))
        if R.is_euc:
            vals = p - ref_v                      # euc per-action integrand
        else:
            u = np.maximum(p, 1e-9) / ref_v
            vals = np.atleast_1d(R.Phi(u)) * np.ones(na)
        spread = max(spread, float(np.max(vals) - np.min(vals)))
    return spread < tol


def softmax3(q, alpha):
    """Plain tempered softmax over logits (used for the trained policy π_θ = softmax(θ))."""
    q = np.asarray(q, float)
    z = (q - q.max()) / alpha
    w = np.exp(z)
    return w / w.sum()
