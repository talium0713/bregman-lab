"""
fig_paper_layered_v2.py — revised t04 and t06, both recast as Amari vs canonical.

  figs/t04_state_anatomy[_pN]   2x7: rows = the seven regularizers, columns = Amari | canonical.
                                Each panel is pi* (dashed) against the recovered pi_theta (solid)
                                over the twelve (layer, state) cells x 3 actions. Euclidean has no
                                f(u) generator, so its canonical cell is blank.
  figs/t06_alpha_sweep          the alpha family under both normalizations, one panel per peak.

Text is kept to the minimum the panel needs; everything else belongs in the caption.
Reads only cached layered runs - no training. Run from python/:  python fig_paper_layered_v2.py
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regularizers import REGKEYS, COLORS, SHORT
from mdp import SN, NA, solve_dp, uniform_pis

DEPTH = 4
GAMMA, EPS = 0.9, 0.2
ARMS = [("std", r"Amari   $f'(1)=0$"), ("canon", r"canonical   $f'(1)=f''(1)$")]


def _save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return f"{stem}.png"


def _kl_kw(rk, lw=1.6):
    return dict(lw=2.6 if rk == "kl" else lw, zorder=8 if rk == "kl" else 3)


# ───────────────────────────────────────────────── t04: 2x7 Amari vs canonical
def fig_anatomy(path, suffix):
    d = json.load(open(path)); pc = d["panel_cache"]; peak = d["peak"]
    block = SN * NA
    centers = [l * block + (block - 1) / 2 for l in range(DEPTH)]
    fig, axes = plt.subplots(len(REGKEYS), 2, figsize=(9.6, 11.4), sharex=True, sharey=True)
    gaps = {}
    rewards = np.asarray(pc["rewards"]); alphas = pc["alphas"]
    for r, rk in enumerate(REGKEYS):
        # pi* is the REGULARIZED optimum, recomputed from the cached reward draw and calibrated
        # weight - not any arm's recovered policy (they are what we are comparing against it).
        star = solve_dp(rk, rewards, uniform_pis(DEPTH), float(alphas[rk]), GAMMA, EPS).pistar
        for c, (arm, _) in enumerate(ARMS):
            ax = axes[r, c]
            pol = pc["pols"][arm].get(rk)
            if pol is None:                                    # euc has no canonical representative
                ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#eee",
                                           hatch="///", edgecolor="#bbb", lw=0))
                ax.text(0.5, 0.5, "no canonical form exists", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7.5, color="#777")
                ax.set_xticks(centers); continue
            pol = np.asarray(pol)
            tgt = np.asarray(d["gap"][arm][rk])
            idx = np.arange(star.size)
            ax.plot(idx, star.reshape(-1), "--", color="#444", lw=0.9, marker="o", ms=1.6, zorder=2)
            ax.plot(idx, pol.reshape(-1), "-", color=COLORS[rk], **_kl_kw(rk))
            for l in range(1, DEPTH):
                ax.axvline(l * block - 0.5, color="#e9e9e9", lw=0.6)
            ax.text(0.985, 0.97, rf"$\Delta_\pi$={tgt[0]:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.5, color=COLORS[rk], zorder=9,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.5))
            gaps.setdefault(arm, {})[rk] = float(tgt[0])
            ax.set_xticks(centers)
            if r == len(REGKEYS) - 1:
                ax.set_xticklabels([rf"$\ell_{l}$" for l in range(DEPTH)], fontsize=7)
        axes[r, 0].set_ylabel(SHORT[rk], color=COLORS[rk], fontsize=10)
        axes[r, 0].set_ylim(0, 1.05)
    for c, (_, lab) in enumerate(ARMS):
        axes[0, c].set_title(lab, fontsize=10)
    fig.tight_layout()
    return _save(fig, f"figs/t04_state_anatomy{suffix}"), peak, gaps


# ───────────────────────────────────────────────── t06: alpha family, Amari vs canonical
def _agg_adiv(path):
    R = json.load(open(path)); man = R["manifest"]; res = R["results"]
    a_grid = man["a_grid"]
    out = {}
    for a in a_grid:
        f = np.concatenate([np.asarray(c["finals"]) for c in res if abs(c["a"] - a) < 1e-9])
        out[a] = (float(f.mean()), float(f.std(ddof=1) / np.sqrt(len(f)) * 1.96))
    return man, a_grid, out


def fig_alpha():
    peaks = []
    for p in sorted(glob.glob("data/tabular/run_adiv_p*0/results.json")):
        pk = int(re.search(r"run_adiv_p(\d+)", p).group(1))
        kln = p.replace(f"_p{pk}/", f"_p{pk}_kln/")
        if os.path.exists(kln):
            peaks.append((pk, p, kln))
    if not peaks:
        return None, {}
    fig, axes = plt.subplots(1, len(peaks), figsize=(4.6 * len(peaks), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    C_AM, C_CA = "#6b7280", "#b0224b"
    tab = {}
    for ax, (pk, p_am, p_ca) in zip(axes, peaks):
        _, A, am = _agg_adiv(p_am)
        _, _, ca = _agg_adiv(p_ca)
        keep = [a for a in A if not (0.9 < a < 1.1 and abs(a - 1.0) > 1e-9)]   # drop dense probes
        Ak = np.array(keep)
        for d, col, lab in ((am, C_AM, r"Amari  $f'(1)=0$"), (ca, C_CA, r"canonical  $f'(1)=f''(1)$")):
            m = np.array([d[a][0] for a in keep]); ci = np.array([d[a][1] for a in keep])
            ax.plot(Ak, m, marker="o", ms=3.5, lw=1.8, color=col, label=lab)
            ax.fill_between(Ak, m - ci, m + ci, color=col, alpha=0.15)
        ax.axvline(1.0, color=COLORS["kl"], ls=":", lw=1.1, alpha=0.7)
        ax.set_title(f"peak {pk/100:.1f}", fontsize=10)
        ax.set_xlabel(r"$\alpha$"); ax.grid(alpha=0.2)
        tab[pk / 100] = {"amari_at_1": am[1.0][0], "canon_at_1": ca[1.0][0]}
    axes[0].set_ylabel(r"$\Delta_\pi$ = mean $\mathrm{TV}(\pi_\theta\,\|\,\pi^\star)$")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].set_ylim(0, None)
    fig.tight_layout()
    return _save(fig, "figs/t06_alpha_sweep"), tab


def main():
    os.makedirs("figs", exist_ok=True)
    made = []
    # t04 — one file per cached peak; the un-suffixed one is the headline peak 0.7
    for path in sorted(glob.glob("data/tabular/canon_2x7_p*.json")):
        pk = re.search(r"canon_2x7_p(\d+)", path).group(1)
        suffix = "" if pk == "7" else f"_p{pk}"
        p, peak, gaps = fig_anatomy(path, suffix)
        made.append(p)
        print(f"\nt04 [peak {peak}]  Delta_pi per arm")
        for rk in REGKEYS:
            a = gaps.get("std", {}).get(rk); c = gaps.get("canon", {}).get(rk)
            print(f"  {SHORT[rk]:7s} amari {a:.3f}" + (f"   canonical {c:.3f}" if c is not None
                                                       else "   canonical —"))
    p, tab = fig_alpha()
    if p:
        made.append(p)
        print("\nt06  alpha=1 (RKL) gap per peak")
        for pk, v in sorted(tab.items()):
            print(f"  peak {pk}:  amari {v['amari_at_1']:.3f}   canonical {v['canon_at_1']:.3f}")
    print("\n[saved]" + "".join(f"\n  {m} (+ .pdf)" for m in made))


if __name__ == "__main__":
    main()
