"""
fig_paper_layered.py — the paper's legacy tree figures, regenerated on the LAYERED MDP.

Produces (all from the cached 100-MDP layered runs in data/tabular/, no retraining):
  figs/t03_recovery_vs_nmc   policy gap vs the MC budget, on-policy and off-policy panels
  figs/t04_state_anatomy     recovered policy vs target at each of the twelve (layer, state)
                             cells, plus the per-cell TV the gap integrates over
  figs/t07_calibration       policy peak vs the regularization weight, with the calibrated
                             anchors, plus the calibrated-weight table (median and range)

Configuration is read from the run manifest and asserted against the paper's block, so a
silent divergence is impossible. Run from python/:  python fig_paper_layered.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

from regularizers import REGKEYS, COLORS, SHORT, SHORT_TEX
from mdp import SN, NA, new_rewards
from seeds import ROOT_SEED, rng_for
from experiments import calibrate, peakiness
from fig_tabular import load, aggregate

MAIN = "data/tabular/run_20260629_182331/results.json"   # peak 0.7, 100 MDP, nmc 1..32
PEAKS_RUN = "data/tabular/run_ablation_peaks/results.json"  # peaks 0.6 / 0.8
GAMMA, EPS, DEPTH = 0.9, 0.2, 4

# the configuration the paper quotes verbatim — asserted, never assumed
EXPECTED_ENV = {"gamma": 0.9, "eps": 0.2, "depth": 4, "SN": 3, "NA": 3,
                "reward": "Uniform(-0.8,0.8)"}

REGIME_LABEL = {"on": "on-policy (fresh rollouts)",
                "off": "off-policy (single logged $a'$)",
                "off_on": "resampled (logged states, fresh $a'$)"}


def _kl_kw(rk, lw=1.8):
    return dict(lw=3.2 if rk == "kl" else lw, zorder=8 if rk == "kl" else 3)


def _save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return f"{stem}.png"


def check_env(man):
    env = man["env"]
    bad = {k: (env.get(k), v) for k, v in EXPECTED_ENV.items() if env.get(k) != v}
    if bad:
        raise SystemExit(f"ENV MISMATCH vs the paper's configuration block: {bad}")
    print(f"[env ok] {env}")
    return env


# ─────────────────────────────────────────────────────────────────────────────
# t03 — policy gap vs MC budget, on-policy and off-policy
# ─────────────────────────────────────────────────────────────────────────────
def fig_recovery_vs_nmc(agg_p, nmc, n_mdp):
    regimes = ["on", "off"]
    fig, axes = plt.subplots(1, len(regimes), figsize=(11.0, 4.3), sharey=True)
    ymax = 0.0
    for ax, reg in zip(axes, regimes):
        avail = sorted(agg_p[reg][REGKEYS[0]].keys())
        flat = len(avail) == 1          # off-policy: the single logged a' — n_mc never enters
        for rk in REGKEYS:
            if flat:
                v, ci = agg_p[reg][rk][avail[0]][0], agg_p[reg][rk][avail[0]][1]
                c = np.full(len(nmc), v); ci = np.full(len(nmc), ci)
                ax.plot(nmc, c, color=COLORS[rk], label=SHORT_TEX[rk], **_kl_kw(rk))
            else:
                c = np.array([agg_p[reg][rk][n][0] for n in nmc])
                ci = np.array([agg_p[reg][rk][n][1] for n in nmc])
                ax.plot(nmc, c, marker="o", ms=4, color=COLORS[rk], label=SHORT_TEX[rk], **_kl_kw(rk))
            ax.fill_between(nmc, c - ci, c + ci, color=COLORS[rk], alpha=0.13, zorder=2)
            ymax = max(ymax, float((c + ci).max()))
        if flat:
            ax.text(0.5, 0.055, "$n_{\\mathrm{mc}}$ does not enter this estimator:\n"
                                "the inner term is the single logged $a'$",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color="#444",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="#ccc", pad=1.4))
        ax.set_xscale("log", base=2); ax.set_xticks(nmc); ax.set_xticklabels(nmc, fontsize=8)
        ax.set_xlabel("MC budget  $n_{\\mathrm{mc}}$")
        ax.set_title(REGIME_LABEL[reg], fontsize=10)
        ax.grid(alpha=0.2)
        # the gap the paper quotes: value at the largest budget, with its CI
        ref = avail[0] if flat else nmc[-1]
        txt = "\n".join(f"{SHORT[rk]} {agg_p[reg][rk][ref][0]:.3f}$\\pm${agg_p[reg][rk][ref][1]:.3f}"
                        for rk in REGKEYS)
        ax.text(0.985, 0.97, f"$\\Delta_\\pi$ at $n_{{\\mathrm{{mc}}}}$={ref}\n{txt}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.0, zorder=9,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.8))
    axes[0].set_ylabel(r"policy gap  $\Delta_\pi$ = mean $\mathrm{TV}(\pi_\theta\,\|\,\pi^\star)$")
    axes[0].legend(fontsize=7.5, ncol=2, loc="lower left")
    for ax in axes:
        ax.set_ylim(0, ymax * 1.30)
    fig.suptitle(f"Recovery vs Monte-Carlo budget — layered MDP, {n_mdp} MDPs, $\\pm$95% CI",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    return _save(fig, "figs/t03_recovery_vs_nmc")


# ─────────────────────────────────────────────────────────────────────────────
# t04 — per (layer, state) anatomy  [design option 1]
# ─────────────────────────────────────────────────────────────────────────────
def fig_state_anatomy(results, peak, regime="off", nm=1):
    cells = {c["rk"]: c for c in results
             if c["peak"] == peak and c["regime"] == regime and c["nm"] == nm
             and c["mi"] == 0 and c.get("pol0") is not None}
    missing = [rk for rk in REGKEYS if rk not in cells]
    if missing:
        raise SystemExit(f"no cached MDP-0 policy for {missing} at ({peak}, {regime}, n_mc={nm})")

    block = SN * NA
    centers = [l * block + (block - 1) / 2 for l in range(DEPTH)]
    fig, axes = plt.subplots(len(REGKEYS), 2, figsize=(11.0, 12.2),
                             gridspec_kw={"width_ratios": [2.6, 1.0]})
    for r, rk in enumerate(REGKEYS):
        c = cells[rk]
        star = np.asarray(c["pistar"]);  pol = np.asarray(c["pol0"])
        axL, axR = axes[r, 0], axes[r, 1]

        # left: the policy itself over the 36 (layer, state, action) entries
        idx = np.arange(star.size)
        axL.plot(idx, star.reshape(-1), "--", color="#444", lw=1.0, marker="o", ms=2.0, zorder=2)
        axL.plot(idx, pol.reshape(-1), "-", color=COLORS[rk], **_kl_kw(rk, 1.7))
        for l in range(1, DEPTH):
            axL.axvline(l * block - 0.5, color="#e6e6e6", lw=0.7)
        axL.set_ylim(0, 1.05); axL.set_xticks(centers)
        axL.set_xticklabels([rf"$\ell_{l}$" for l in range(DEPTH)], fontsize=7.5)
        axL.set_ylabel(SHORT_TEX[rk], color=COLORS[rk], fontsize=10)

        # right: the per-cell TV the gap integrates over (12 cells)
        tv = 0.5 * np.abs(pol - star).sum(-1)                      # (DEPTH, SN)
        flat = tv.reshape(-1)
        axR.bar(np.arange(flat.size), flat, color=COLORS[rk], alpha=0.85,
                edgecolor="#222", linewidth=0.4)
        axR.axhline(flat.mean(), color="#222", ls="--", lw=1.0)
        axR.set_ylim(0, 1.0); axR.set_xticks(np.arange(0, DEPTH * SN, SN) + (SN - 1) / 2)
        axR.set_xticklabels([rf"$\ell_{l}$" for l in range(DEPTH)], fontsize=7.5)
        axR.text(0.97, 0.93, rf"$\Delta_\pi$={flat.mean():.3f}", transform=axR.transAxes,
                 ha="right", va="top", fontsize=7.5, color=COLORS[rk], zorder=9,
                 bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.6))
        if r == 0:
            axL.set_title(r"$\pi^\star$ (dashed) vs $\pi_\theta$ (solid) over the 12 cells $\times$ 3 actions",
                          fontsize=10)
            axR.set_title(r"per-cell $\mathrm{TV}(\pi_\theta\|\pi^\star)$  (dashed = mean = $\Delta_\pi$)",
                          fontsize=10)
    fig.suptitle(f"Per-(layer, state) anatomy — {REGIME_LABEL[regime]}, $n_{{\\mathrm{{mc}}}}$={nm}, "
                 f"peak {peak}, MDP 0", fontsize=11, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    return _save(fig, "figs/t04_state_anatomy"), {rk: float((0.5 * np.abs(
        np.asarray(cells[rk]["pol0"]) - np.asarray(cells[rk]["pistar"])).sum(-1)).mean())
        for rk in REGKEYS}


# ─────────────────────────────────────────────────────────────────────────────
# t07 — calibration: peak vs the regularization weight + the calibrated-weight table
# ─────────────────────────────────────────────────────────────────────────────
def _tr(ax):
    """x in axes fraction, y in data coords."""
    return blended_transform_factory(ax.transAxes, ax.transData)


def fig_calibration(results_by_peak, targets):
    rewards = new_rewards(DEPTH, rng_for(ROOT_SEED, "reward", 0))     # MDP 0, the swept one
    grid = np.logspace(np.log10(0.02), np.log10(40), 60)
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    for rk in REGKEYS:
        pk = [peakiness(rk, rewards, a, GAMMA, EPS) for a in grid]
        ax.plot(grid, pk, color=COLORS[rk], label=SHORT_TEX[rk], **_kl_kw(rk, 1.6))
    for t in targets:
        al = calibrate(rewards, t, GAMMA, EPS)
        ax.axhline(t, color="#888", ls="--", lw=0.9)
        # sit just above each dashed line and inside the axes, not spilling off the right edge
        ax.text(0.012, t, f"target {t}", transform=_tr(ax), va="bottom", ha="left",
                fontsize=8, color="#555")
        for rk in REGKEYS:
            ax.scatter([al[rk]], [t], color=COLORS[rk], s=30, zorder=9 if rk == "kl" else 5,
                       edgecolor="#222", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"regularization weight  $\alpha$   (calibrated per $\Omega$, not fixed)")
    ax.set_ylabel(r"policy peak   $\mathrm{mean}_s \max_a \pi^\star(a|s)$")
    ax.grid(alpha=0.2); ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    p = _save(fig, "figs/t07_calibration")

    # table: calibrated weight per Omega, median and range over the 100 MDPs (from cache)
    table = {}
    for peak, res in results_by_peak.items():
        per = {rk: sorted({c["mi"]: c["alpha"] for c in res
                           if c["rk"] == rk and c["peak"] == peak}.values()) for rk in REGKEYS}
        table[peak] = {rk: {"median": float(np.median(v)), "min": float(v[0]), "max": float(v[-1]),
                            "n_mdp": len(v)} for rk, v in per.items()}
    return p, table


def main():
    os.makedirs("figs", exist_ok=True)
    man, results, _ = load([MAIN])
    env = check_env(man)
    agg, peaks, mdps = aggregate(results)
    peak = peaks[0]
    nmc = sorted({c["nm"] for c in results if c["regime"] == "on"})
    print(f"[data] {MAIN}: {len(mdps)} MDPs · peak {peak} · n_mc {nmc} · "
          f"regimes {sorted({c['regime'] for c in results})}")

    p3 = fig_recovery_vs_nmc(agg[peak], nmc, len(mdps))
    p4, gaps04 = fig_state_anatomy(results, peak, regime="off", nm=1)

    by_peak = {peak: results}
    if os.path.exists(PEAKS_RUN):
        _, res2, _ = load([PEAKS_RUN])
        for pk in sorted({c["peak"] for c in res2}):
            by_peak[pk] = res2
    targets = sorted(by_peak)
    p7, caltab = fig_calibration(by_peak, targets)

    print("\nTABLE 3 — off-policy per-cell gap at n_mc=1 (MDP 0, the t04 panel)")
    for rk in REGKEYS:
        print(f"  {SHORT[rk]:7s} {gaps04[rk]:.3f}" + ("   <- admissible" if rk == "kl" else ""))

    print(f"\nTABLE 4 — calibrated regularization weight (median [min, max] over 100 MDPs)")
    print(f"  {'Omega':7s}" + "".join(f"{f'peak {t}':>26s}" for t in targets))
    for rk in REGKEYS:
        row = "".join(f"{caltab[t][rk]['median']:>10.3f} [{caltab[t][rk]['min']:.2f},"
                      f"{caltab[t][rk]['max']:.2f}]" for t in targets)
        print(f"  {SHORT[rk]:7s}" + row)

    json.dump({"env": env, "peak": peak, "n_mdp": len(mdps), "nmc": nmc,
               "t04_gaps_off_nmc1_mdp0": gaps04, "t07_calibrated_weight": caltab},
              open("figs/layered_paper_tables.json", "w"), indent=1)
    print(f"\n[saved]\n  {p3} (+ .pdf)\n  {p4} (+ .pdf)\n  {p7} (+ .pdf)\n"
          f"  figs/layered_paper_tables.json")


if __name__ == "__main__":
    main()
