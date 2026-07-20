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

KEYS = ["kl", "rkl", "js", "hel", "chi2", "adiv", "euc"]     # = tabular REGKEYS (reordered)
SHORT = {"kl": "RKL", "rkl": "FKL", "js": "JS", "hel": "Hel",
         "chi2": "χ²", "adiv": "α-div", "euc": "Euc"}
COLORS = {"kl": "#EE008D", "adiv": "#BE3EC5", "rkl": "#4065E9", "js": "#037CF2",
          "hel": "#0088E1", "chi2": "#008CB9", "euc": "#008B83"}
DEFAULT_ADIV_A = 0.5
LN2 = math.log(2.0)


def phi_from_logu(key: str, log_u: torch.Tensor, adiv_a: float = DEFAULT_ADIV_A) -> torch.Tensor:
    """Φ(u) for f-divergences, from log_u = log(π_θ/π_ref). Returns same shape as log_u (float32).
    NOT for 'euc' (Φ_euc needs the raw probabilities, see phi_euc)."""
    lu = log_u if log_u.dtype == torch.float64 else log_u.float()   # keep fp64; upcast bf16/fp16
    inv_u = torch.exp(-lu)                # 1/u
    if key == "kl":
        return torch.ones_like(lu)
    if key == "rkl":                      # (ln u − 1)/u
        return (lu - 1.0) * inv_u
    if key == "chi2":                     # u − 1/u
        return torch.exp(lu) - inv_u
    if key == "hel":                      # u^(−1/2) − u^(−1)
        return torch.exp(-0.5 * lu) - inv_u
    if key == "js":                       # (1/u)·ln((1+u)/2) = (log1p(u) − ln2)/u
        return (torch.log1p(torch.exp(lu)) - LN2) * inv_u
    if key == "adiv":                     # (u^a − 1)/(a u)
        return (torch.exp(adiv_a * lu) - 1.0) / adiv_a * inv_u
    raise ValueError(f"unknown / non-scalar key {key!r}")


def phi_euc(p_policy: torch.Tensor, p_ref: torch.Tensor) -> torch.Tensor:
    """Φ_euc at a token = π_θ(a) − π_ref(a). Needs the actual probabilities, not just u."""
    return (p_policy - p_ref).float()


def exact_C(key: str, log_pi: torch.Tensor, log_ref: torch.Tensor,
            adiv_a: float = DEFAULT_ADIV_A) -> torch.Tensor:
    """Closed-form inner term C_Ω(π_θ(·|s)) = Σ_a π_θ(a)·Φ(u_a), summed over the FULL vocab.
    log_pi, log_ref: full-vocab log-probs, shape [..., V]. Returns [...] (per position). fp32.
    RKL comes out exactly 1 by arithmetic; euc uses its own Bregman form.
    Reduced in float64 — the heavy-tailed members (χ², FKL) lose accuracy summing in float32."""
    log_pi = log_pi.double(); log_ref = log_ref.double()
    pi = torch.exp(log_pi)
    if key == "euc":                      # <π, π−π_ref> − ½‖π−π_ref‖²
        pr = torch.exp(log_ref)
        d = pi - pr
        return (pi * d).sum(-1) - 0.5 * (d * d).sum(-1)
    log_u = log_pi - log_ref              # [..., V]
    phi = phi_from_logu(key, log_u, adiv_a)
    return (pi * phi).sum(-1)
