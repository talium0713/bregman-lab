"""
fig_tabular.py — figures + tables from run_tabular.py results.json (the reproducible runs).

Aggregation pools all finals across MDPs for a population mean ± 95% CI: under the
100-independent-MDP design each MDP is one i.i.d. draw, so the CI honestly includes MDP + data +
optimization variance.  Everything is keyed by the calibration peak, so multiple results.json
(e.g. a peak={0.6,0.8} ablation merged with the peak=0.7 headline) coexist and produce per-peak
figures + a cross-peak comparison.

Produces (per peak P present in the data):
  figs/tabular_headline_p{P}.png   3 panels — off (bars), off_on & on (Δπ vs n_mc, ±95%CI)
  figs/tabular_offpolicy_peaks.png the off-policy punchline across all peaks (RKL alone recovers π*)
  figs/tabular_alpha_sweep.png     deterministic peak(α) sweep (MDP 0) with the calibrated anchors
  figs/tabular_calibration.csv/.tex  calibrated α per Ω per peak, peakiness, C_Ω, admissibility

Run:  python fig_tabular.py [results.json ...]   (default: newest data/tabular/run_*/, merges all given)
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regularizers import (REGKEYS as _REGKEYS, COLORS as _COLORS, SHORT as _SHORT,
                          REGIME_LABEL, REGIME_SUB)
from mdp import new_rewards
from experiments import peakiness, calibrate
from seeds import rng_for


def load(paths=None):
    """Load + merge one or more results.json. Returns (manifest, merged_results, out_dir)."""
    if paths is None:
        runs = sorted(glob.glob("data/tabular/run_*/results.json"))
        if not runs:
            sys.exit("no data/tabular/run_*/results.json — run run_tabular.py first")
        paths = [runs[-1]]
    elif isinstance(paths, str):
        paths = [paths]
    man, results = None, []
    for p in paths:
        R = json.load(open(p))
        man = man or R["manifest"]
        results.extend(R["results"])
    return man, results, os.path.dirname(paths[0])


def _kl_kw(rk, lw=1.7):
    """The admissible Ω (code key 'kl', plotted as RKL) is drawn thick and on top."""
    return dict(lw=3.4 if rk == "kl" else lw, zorder=8 if rk == "kl" else 3)


def aggregate(results):
    """out[peak][regime][rk][nm] = (center, ci95, per_mdp_means). Pools finals across MDPs."""
    bucket = {}
    for r in results:
        bucket.setdefault((r["peak"], r["regime"], r["rk"], r["nm"]), {}) \
              .setdefault(r["mi"], []).extend(r["finals"])
    out = {}
    for (peak, reg, rk, nm), by_mdp in bucket.items():
        pooled = np.concatenate([np.asarray(by_mdp[mi], float) for mi in by_mdp])
        n = len(pooled)
        center = float(pooled.mean())
        ci = float(pooled.std(ddof=1) / np.sqrt(n) * 1.96) if n > 1 else 0.0
        perm = np.array([float(np.mean(by_mdp[mi])) for mi in sorted(by_mdp)])
        out.setdefault(peak, {}).setdefault(reg, {}).setdefault(rk, {})[nm] = (center, ci, perm)
    peaks = sorted(out)
    mdps = sorted({r["mi"] for r in results})
    return out, peaks, mdps


def _design_str(man):
    nm, ns = man["config"]["n_mdp"], man["config"]["n_seeds"]
    return f"{nm} independent MDPs" if ns == 1 else f"{nm} fixed MDP × {ns} training seeds"


def fig_headline(man, agg_p, peak, nmc):
    """One peak's 3-panel headline: off (bars), off_on & on (Δπ vs n_mc, ±95%CI)."""
    COLORS, SHORT = _COLORS, _SHORT   # canonical palette (colours are cosmetic; don't freeze the manifest's)
    regimes = [r for r in ("off", "off_on", "on") if r in agg_p]
    titles = {reg: f"{REGIME_LABEL[reg]}  ({REGIME_SUB[reg]})" for reg in regimes}
    fig, axes = plt.subplots(1, len(regimes), figsize=(5.2 * len(regimes), 4.4))
    axes = np.atleast_1d(axes)
    ymax = 0
    for ax, reg in zip(axes, regimes):
        if reg == "off":
            x = np.arange(len(REGKEYS))
            for i, rk in enumerate(REGKEYS):
                c, ci, perm = agg_p[reg][rk][1]
                ax.bar(x[i], c, 0.74, yerr=ci, capsize=3, color=COLORS[rk],
                       edgecolor="#111" if rk == "kl" else "none",
                       linewidth=2.0 if rk == "kl" else 0, zorder=3)
                if len(perm) <= 10:
                    ax.plot([x[i]] * len(perm), perm, "o", ms=3, color="#222", alpha=0.5, zorder=5)
                ymax = max(ymax, c + ci)
            ax.set_xticks(x); ax.set_xticklabels([SHORT[rk] for rk in REGKEYS], fontsize=8)
            ax.set_ylabel("policy gap  Δπ  (mean TV vs π*)")
        else:
            for rk in REGKEYS:
                cs = np.array([agg_p[reg][rk][n][0] for n in nmc])
                ci = np.array([agg_p[reg][rk][n][1] for n in nmc])
                ax.plot(nmc, cs, marker="o", ms=4, color=COLORS[rk], label=SHORT[rk], **_kl_kw(rk))
                ax.fill_between(nmc, cs - ci, cs + ci, color=COLORS[rk], alpha=0.13, zorder=2)
                ymax = max(ymax, (cs + ci).max())
            ax.set_xscale("log", base=2); ax.set_xticks(nmc); ax.set_xticklabels(nmc, fontsize=8)
            ax.set_xlabel("MC budget  $n$")
        ax.set_title(titles[reg], fontsize=10); ax.grid(alpha=0.2)
    for ax in axes:
        ax.set_ylim(0, ymax * 1.08)
    axes[-1].legend(fontsize=8, ncol=2, title="Ω (RKL = permissible)")
    fig.tight_layout()
    p = f"figs/tabular_headline_p{int(round(peak*100))}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close()
    return p


def fig_offpolicy_peaks(man, agg, peaks):
    """The off-policy punchline across all peaks: grouped bars per Ω, one bar per peak."""
    COLORS, SHORT = _COLORS, _SHORT   # canonical palette (colours are cosmetic; don't freeze the manifest's)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(REGKEYS)); w = 0.8 / len(peaks)
    for j, peak in enumerate(peaks):
        for i, rk in enumerate(REGKEYS):
            c, ci, _ = agg[peak]["off"][rk][1]
            ax.bar(x[i] + (j - (len(peaks) - 1) / 2) * w, c, w, yerr=ci, capsize=2,
                   color=COLORS[rk], alpha=0.45 + 0.55 * j / max(len(peaks) - 1, 1),
                   edgecolor="#111" if rk == "kl" else "none", linewidth=1.2 if rk == "kl" else 0,
                   label=f"peak {peak}" if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[rk] for rk in REGKEYS])
    ax.set_ylabel("off-policy gap  Δπ  (mean TV vs π*)")   # caption: {design} · ±95% CI · RKL lowest at every peak
    ax.grid(alpha=0.2, axis="y"); ax.legend(fontsize=8, title="bar shade = peak")
    fig.tight_layout()
    p = "figs/tabular_offpolicy_peaks.png"
    fig.savefig(p, dpi=140); plt.close()
    return p


def fig_alpha_sweep(man, peaks):
    """Deterministic peak(α) sweep for the representative MDP 0, with the calibrated α anchors for
    every peak present marked.  No randomness: solve_dp is a pure function of the reward draw."""
    COLORS, SHORT = _COLORS, _SHORT   # canonical palette (colours are cosmetic; don't freeze the manifest's)
    root, env = man["root_seed"], man["env"]
    gamma, eps, depth = env["gamma"], env["eps"], env["depth"]
    rew = new_rewards(depth, rng_for(root, "reward", 0))
    grid = np.logspace(np.log10(0.05), np.log10(6), 60)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for rk in REGKEYS:
        pk = [peakiness(rk, rew, a, gamma, eps) for a in grid]
        ax.plot(grid, pk, color=COLORS[rk], label=SHORT[rk], **_kl_kw(rk, 1.6))
    for peak in peaks:
        al = calibrate(rew, peak, gamma, eps)
        ax.axhline(peak, color="#888", ls="--", lw=0.9)
        for rk in REGKEYS:
            ax.scatter([al[rk]], [peak], color=COLORS[rk], s=26, zorder=9 if rk == "kl" else 4)
        ax.text(grid[-1], peak, f" peak {peak}", va="center", fontsize=8, color="#555")
    ax.set_xscale("log"); ax.set_xlabel(r"regularization temperature $\beta$")   # temperature (C0: β, not the family α)
    ax.set_ylabel("peak   mean_s max_a π*(a|s)")   # caption: MDP 0, deterministic; per-Ω β calibrated to each target peak
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.2)
    fig.tight_layout()
    p = "figs/tabular_alpha_sweep.png"
    fig.savefig(p, dpi=140); plt.close()
    return p


def table_calibration(man, results, peaks):
    """Calibrated α per Ω per peak + peakiness/C_Ω/admissibility (peak-independent). CSV + LaTeX."""
    SHORT = man["short"]
    by = {}  # by[rk][peak] = alpha ; plus C_exact/admissible from any cell
    meta = {}
    for r in results:
        if r["regime"] == "off" and r["nm"] == 1 and r["mi"] == 0:
            by.setdefault(r["rk"], {})[r["peak"]] = r["alpha"]
            meta[r["rk"]] = (r["C_exact"], r["admissible"])
    pcols = ",".join(f"alpha@{p}" for p in peaks)
    csv = ["Omega," + pcols + ",C_exact,permissible"]
    tex = [r"\begin{tabular}{l" + "r" * len(peaks) + "rc}", r"\toprule",
           "$\\Omega$ & " + " & ".join(f"$\\beta_{{{p}}}$" for p in peaks)
           + r" & $C_\Omega(\pi^*)$ & permissible \\", r"\midrule"]
    for rk in REGKEYS:
        ce, adm = meta[rk]
        als = [by[rk].get(p, float("nan")) for p in peaks]
        csv.append(f"{SHORT[rk]}," + ",".join(f"{a:.4f}" for a in als) + f",{ce:.4f},{int(adm)}")
        tex.append(f"{SHORT[rk]} & " + " & ".join(f"{a:.3f}" for a in als)
                   + f" & {ce:.3f} & {'yes' if adm else 'no'} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open("figs/tabular_calibration.csv", "w") as f:
        f.write("\n".join(csv) + "\n")
    with open("figs/tabular_calibration.tex", "w") as f:
        f.write("\n".join(tex) + "\n")
    return "figs/tabular_calibration.csv", "figs/tabular_calibration.tex"


# module-level palette/labels (results.json carries copies, but these are the canonical source)
REGKEYS, COLORS, SHORT = _REGKEYS, _COLORS, _SHORT


def main():
    paths = sys.argv[1:] or None
    man, results, _ = load(paths)
    os.makedirs("figs", exist_ok=True)
    agg, peaks, mdps = aggregate(results)
    nmc = list(man["config"]["nmc"])
    print(f"[data] {len(results)} cells · {len(mdps)} MDPs · peaks={peaks} · "
          f"root={man['root_seed']} git={man['git_commit'][:8]}")
    outs = []
    for peak in peaks:
        outs.append(fig_headline(man, agg[peak], peak, nmc))
    if len(peaks) > 1:
        outs.append(fig_offpolicy_peaks(man, agg, peaks))
    outs.append(fig_alpha_sweep(man, peaks))
    outs += list(table_calibration(man, results, peaks))
    print("[saved]\n  " + "\n  ".join(outs))


if __name__ == "__main__":
    main()
