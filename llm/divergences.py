"""
divergences.py — the 7 regularizers' inner-term integrand Φ(u), in torch, for LLM scale.

Port of python/regularizers.py (numpy) Φ = f'(u) − f(u)/u, u = π_θ/π_ref. Everything is computed
from log_u = logπ_θ − logπ_ref for numerical stability (§5: never materialize π/π_ref directly).

  Ω            Φ(u)                         const?   admissible
  RKL  (kl)    1                            yes      YES  (f = u ln u)
  FKL  (rkl)   (ln u − 1)/u                 no
  JS   (js)    (1/u)·ln((1+u)/2)            no
  Hel  (hel)   u^(−1/2) − u^(−1)            no
  χ²   (chi2)  u − 1/u                      no       (not DPO-inducing — confound)
  α-div        (u^a − 1)/(a u)   a=0.5      no       (a=0.5 ⇒ Φ = 2·Φ_Hel)
  Euc          π_a − π_ref,a                no       (NOT a function of u alone — see phi_euc)

KEYS/labels/colors mirror python/regularizers.py so figures stay consistent across tabular + LLM.
"""
from __future__ import annotations

import math
import torch

KEYS = ["kl", "adiv", "rkl", "js", "hel", "chi2", "euc"]     # = tabular REGKEYS order (α-div 2nd; matches COLORS)
SHORT = {"kl": "RKL", "rkl": "FKL", "js": "JS", "hel": "Hel",
         "chi2": "χ²", "adiv": "α-div", "euc": "Euc"}
COLORS = {"kl": "#EE008D", "adiv": "#BE3EC5", "rkl": "#4065E9", "js": "#037CF2",
          "hel": "#00BCD4", "chi2": "#12AE5A", "euc": "#008B83"}   # chi2 green: kept in sync with python/regularizers.py
DEFAULT_ADIV_A = 0.5
LN2 = math.log(2.0)

# Generator normalization (Appendix E). Adding s·(t−1) to the generator is affine, so it leaves Ω/π*
# unchanged but shifts the inner term: f'→f'+s, Φ→Φ+s/u, C→C+s (E_π[1/u]=1). The shift is chosen to
# hit a target f'(1); the modes below mirror python/regularizers.py exactly.
#   natural         : s=0            (each divergence's own generator; RKL f'(1)=1, adiv/js/hel/chi2 0, FKL −1)
#   amari|standard  : f'(1)=0        s = −f'(1)          (RKL ⇒ u ln u−(u−1), Φ=1−1/u)
#   kln             : f'(1)=1        s = 1−f'(1)         (KL-consistent; RKL is a no-op)
#   canon           : f'(1)=f''(1)   s = f''(1)−f'(1)    (tabular canonical; RKL/FKL/adiv=1, js/hel=0.5, χ²=2)
# euc is excluded from all normalization (Bregman, no f(u) generator).
FP1  = {"kl": 1.0, "rkl": -1.0, "adiv": 0.0, "js": 0.0, "hel": 0.0, "chi2": 0.0}   # f'(1)
FPP1 = {"kl": 1.0, "rkl":  1.0, "adiv": 1.0, "js": 0.5, "hel": 0.5, "chi2": 2.0}   # f''(1)


def norm_shift(key: str, norm: str | None) -> float:
    """The additive generator shift s for normalization mode `norm` (see table above)."""
    if norm in (None, "natural"):
        return 0.0
    if norm in ("amari", "standard"):
        return -FP1[key]
    if norm == "kln":
        return 1.0 - FP1[key]
    if norm == "canon":
        return FPP1[key] - FP1[key]
    raise ValueError(f"unknown normalization {norm!r} (want natural|amari|kln|canon)")


KLN_SHIFT = {k: 1.0 - FP1[k] for k in FP1}   # back-compat: == the old {"kl":0,"rkl":2,"adiv":1,...:1}


def _norm_arg(kln: bool, norm: str | None) -> str:
    """Resolve the legacy kln=bool and the new norm=str into one mode (norm wins if given)."""
    return norm if norm is not None else ("kln" if kln else "natural")


def phi_from_logu(key: str, log_u: torch.Tensor, adiv_a: float = DEFAULT_ADIV_A,
                  kln: bool = False, norm: str | None = None) -> torch.Tensor:
    """Φ(u) for f-divergences, from log_u = log(π_θ/π_ref). Returns same shape as log_u (float32).
    Normalization: pass norm∈{natural,amari,kln,canon} (or legacy kln=True ⇒ norm='kln'); adds s/u
    with s=norm_shift(key,norm). NOT for 'euc'."""
    lu = log_u if log_u.dtype == torch.float64 else log_u.float()   # keep fp64; upcast bf16/fp16
    inv_u = torch.exp(-lu)                # 1/u
    if key == "kl":
        base = torch.ones_like(lu)
    elif key == "rkl":                    # (ln u − 1)/u
        base = (lu - 1.0) * inv_u
    elif key == "chi2":                   # u − 1/u
        base = torch.exp(lu) - inv_u
    elif key == "hel":                    # u^(−1/2) − u^(−1)
        base = torch.exp(-0.5 * lu) - inv_u
    elif key == "js":                     # (1/u)·ln((1+u)/2) = (log1p(u) − ln2)/u
        base = (torch.log1p(torch.exp(lu)) - LN2) * inv_u
    elif key == "adiv":                   # (u^a − 1)/(a u)
        base = (torch.exp(adiv_a * lu) - 1.0) / adiv_a * inv_u
    else:
        raise ValueError(f"unknown / non-scalar key {key!r}")
    s = norm_shift(key, _norm_arg(kln, norm))
    return base + s * inv_u if s != 0.0 else base              # Φ_norm = Φ + s/u


def fprime_from_logu(key: str, log_u: torch.Tensor, adiv_a: float = DEFAULT_ADIV_A,
                     kln: bool = False, norm: str | None = None) -> torch.Tensor:
    """f'(u) at logged tokens — the DPO chosen-action term [∇Ω]_a — from log_u = log(π_θ/π_ref).
    Normalization: norm∈{natural,amari,kln,canon} (or legacy kln=True); adds the constant s=norm_shift.
    NOT for 'euc'.

    Identity used by the RKL anchor: f'(u) − Φ(u) = f(u)/u, and for RKL f/u = ln u, so the
    single-sample score Σ[f'−Φ] = Σ ln u = the standard DPO log-ratio (see stage_b_train)."""
    lu = log_u if log_u.dtype == torch.float64 else log_u.float()
    inv_u = torch.exp(-lu)
    if key == "kl":                       # ln u + 1
        base = lu + 1.0
    elif key == "rkl":                    # −1/u
        base = -inv_u
    elif key == "chi2":                   # 2(u − 1)
        base = 2.0 * (torch.exp(lu) - 1.0)
    elif key == "hel":                    # 1 − u^(−1/2)
        base = 1.0 - torch.exp(-0.5 * lu)
    elif key == "js":                     # ln(2u/(1+u)) = ln2 + ln u − log1p(u)
        base = LN2 + lu - torch.log1p(torch.exp(lu))
    elif key == "adiv":                   # (u^(a−1) − 1)/(a − 1)
        base = (torch.exp((adiv_a - 1.0) * lu) - 1.0) / (adiv_a - 1.0)
    else:
        raise ValueError(f"unknown / non-scalar key {key!r}")
    return base + norm_shift(key, _norm_arg(kln, norm))       # f'_norm = f' + s


def phi_euc(p_policy: torch.Tensor, p_ref: torch.Tensor) -> torch.Tensor:
    """Φ_euc at a token = π_θ(a) − π_ref(a). Needs the actual probabilities, not just u."""
    return (p_policy - p_ref).float()


def exact_C(key: str, log_pi: torch.Tensor, log_ref: torch.Tensor,
            adiv_a: float = DEFAULT_ADIV_A, chunk: int = 256,
            dtype: torch.dtype = torch.float64, kln: bool = False,
            norm: str | None = None) -> torch.Tensor:
    """Closed-form inner term C_Ω(π_θ(·|s)) = Σ_a π_θ(a)·Φ(u_a), summed over the FULL vocab.
    log_pi, log_ref: full-vocab log-probs, shape [T, V]. Returns [T] (per position).
    RKL comes out exactly 1 by arithmetic; euc uses its own Bregman form.

    `dtype`: reduction precision. Stage A measurement uses float64 (the heavy-tailed χ²/FKL lose
    accuracy in float32). Stage B TRAINING passes float32 — here each term π_θ(a)·Φ(u_a) is bounded
    (→ −f(0⁺)·π_ref(a) as u→0, the π_θ weight cancelling the 1/u tail), so fp32 is accurate and
    half the memory for backprop. CHUNKED over the token dim to bound the full-vocab temporaries."""
    if log_pi.shape[0] == 0:
        return log_pi.new_zeros(0, dtype=dtype)
    outs = []
    for i in range(0, log_pi.shape[0], chunk):
        lp = log_pi[i:i + chunk].to(dtype); lr = log_ref[i:i + chunk].to(dtype)
        pi = torch.exp(lp)
        if key == "euc":                  # <π, π−π_ref> − ½‖π−π_ref‖²
            d = pi - torch.exp(lr)
            outs.append((pi * d).sum(-1) - 0.5 * (d * d).sum(-1))
        else:
            outs.append((pi * phi_from_logu(key, lp - lr, adiv_a)).sum(-1))
    C = torch.cat(outs)
    s = 0.0 if key == "euc" else norm_shift(key, _norm_arg(kln, norm))
    return C + s                                                 # C_norm = C + s (since E_π[1/u]=1)
