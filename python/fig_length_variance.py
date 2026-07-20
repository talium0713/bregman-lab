"""
fig_length_variance.py — §3 length→variance experiment (off-policy, DPO-vs-f-DPO regime).

Claim (§3): with the standard SUM reward (no length normalization), the pairwise score gap
d = S_w − S_l has variance that GROWS with the total length T_w+T_l for every non-admissible Ω,
and is EXACTLY 0 for the admissible RKL — at every length.

Setup (matches the tabular `off` regime, π_ref uniform):
  * one inner term per non-terminal token: Ĉ_Ω(s') = Φ(u_{a'}), single logged next token a'.
  * off-policy: a' ~ behaviour = π_ref (uniform).  π = softmax(N(0,1)·scale), so u = |A|·π.
  * a length-T trajectory contributes T−1 independent inner draws: S = Σ_{t=1}^{T-1} Φ(u_{a'_t}).
  * α = 1 (std[d] scales linearly in α).
Theory:  Var[d] = α²(T_w+T_l−2)·σ²_pool,  σ²_pool = Var_{state, a'~π_ref}[Φ].  RKL: σ_pool=0.

Left  : std[d] vs T (equal lengths T_w=T_l=T). Markers = Monte-Carlo, lines = theory σ_pool√(2(T−1)).
        RKL pinned at the floor (exact 0). Straight lines on log-log confirm the √T law.
Right : empirical heatmap std[d](T_w,T_l) for FKL. Iso-lines are anti-diagonals ⇒ depends only on
        T_w+T_l, exactly as the formula predicts (not on how the length is split).

Run:  python fig_length_variance.py   →   figs/length_variance.png
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regularizers import REG, COLORS, SHORT

FIGDIR = os.path.join(os.path.dirname(__file__), "figs")
KEYS = ["kl", "adiv", "rkl", "js", "hel", "chi2"]
NA = 100
SCALE = 1.0
FLOOR = 1e-2


def phi_pool(reg_key: str, n_states: int = 20000, seed: int = 0) -> np.ndarray:
    """Pool of Φ(u_a) over random states × all actions; a'~π_ref(uniform) ≡ uniform pick from pool."""
    rng = np.random.default_rng(seed)
    lg = rng.standard_normal((n_states, NA)) * SCALE
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return REG[reg_key].Phi(p * NA).ravel()          # u = |A|·π (π_ref uniform)


def std_d(pool: np.ndarray, Tw: int, Tl: int, n_real: int, rng) -> float:
    def S(T):
        if T < 2:
            return np.zeros(n_real)
        idx = rng.integers(0, pool.size, size=(n_real, T - 1))
        return pool[idx].sum(1)
    return float((S(Tw) - S(Tl)).std())


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 5.0))
    pools = {k: phi_pool(k) for k in KEYS}
    sigma = {k: float(pools[k].std()) for k in KEYS}

    # ---- Left: std[d] vs T (equal lengths), empirical vs theory ----
    Ts = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    rng = np.random.default_rng(1)
    for k in KEYS:
        emp = [max(std_d(pools[k], T, T, 120000, rng), FLOOR) for T in Ts]
        axL.plot(Ts, emp, "o", color=COLORS[k], ms=5,
                 zorder=9 if k == "kl" else 4, label=SHORT[k])
        th = np.maximum(sigma[k] * np.sqrt(2 * (np.array(Ts) - 1)), FLOOR)
        axL.plot(Ts, th, "-", color=COLORS[k], lw=2.6 if k == "kl" else 1.5,
                 alpha=0.9, zorder=8 if k == "kl" else 3)
    axL.axhline(FLOOR, color="0.6", ls=":", lw=1)
    axL.text(2.1, FLOOR * 1.15, "RKL ≡ 0 (floored)", fontsize=8, color="0.4", va="bottom")
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel(r"trajectory length $T$  ($T_w=T_l=T$)")
    axL.set_ylabel(r"$\mathrm{std}[d]=\mathrm{std}[S_w-S_l]$   ($\alpha=1$)")
    axL.set_title(r"std[d] vs length: markers=MC, lines=theory $\sigma_{\rm pool}\sqrt{2(T-1)}$")
    axL.legend(ncol=2, fontsize=9, framealpha=0.9)
    axL.grid(True, which="both", alpha=0.15)

    # ---- Right: empirical heatmap std[d](T_w,T_l) for FKL ----
    grid = list(range(2, 41, 4))
    Z = np.zeros((len(grid), len(grid)))
    rng2 = np.random.default_rng(2)
    pool_fkl = pools["rkl"]  # SHORT['rkl']='FKL' (code key rkl = forward KL)
    for i, Tw in enumerate(grid):
        for j, Tl in enumerate(grid):
            Z[i, j] = std_d(pool_fkl, Tw, Tl, 30000, rng2)
    im = axR.imshow(Z, origin="lower", cmap="viridis",
                    extent=[grid[0], grid[-1], grid[0], grid[-1]], aspect="auto")
    cs = axR.contour(grid, grid, Z, levels=6, colors="w", linewidths=0.8, alpha=0.7)
    axR.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    axR.set_xlabel(r"$T_l$ (loser length)")
    axR.set_ylabel(r"$T_w$ (winner length)")
    axR.set_title("FKL: std[d]($T_w,T_l$) — contours are anti-diagonals\n"
                  r"$\Rightarrow$ depends only on $T_w+T_l$, not the split")
    fig.colorbar(im, ax=axR, label="std[d]")

    fig.suptitle("§3 length→variance (off-policy, standard SUM reward): RKL noise-free at every "
                 "length; non-admissible std[d] ∝ √(T_w+T_l−2)", fontsize=10.5, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "length_variance.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)

    print("\nσ_pool[Φ] (off-policy, a'~π_ref):")
    for k in KEYS:
        print("  %-5s %9.3g" % (SHORT[k], sigma[k]))
    print("\nstd[d] off-policy (α=1), MC:")
    rng3 = np.random.default_rng(7)
    print("           Tw=8,Tl=8   Tw=8,Tl=4   Tw=32,Tl=4")
    for k in ["kl", "rkl", "chi2", "js"]:
        row = [std_d(pools[k], tw, tl, 300000, rng3) for (tw, tl) in [(8, 8), (8, 4), (32, 4)]]
        print("  %-5s   " % SHORT[k] + "   ".join("%9.3g" % v for v in row))


if __name__ == "__main__":
    main()
