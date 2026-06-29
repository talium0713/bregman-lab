"""
fig_eps.py — ε (transition-noise) ablation figures from the fixed-α ε sweep.

Reads the ε-ablation run (ε ∈ {0,0.1,0.3,0.4,0.5}, α held fixed via calib_eps=0.2) and merges the
ε=0.2 point from the headline run (same α calibration), giving a complete Δπ-vs-ε curve.  α is the
SAME for every ε, so the curves isolate the effect of transition noise alone — not an α artifact.

Produces:
  figs/tabular_eps_offpolicy.png   off-policy Δπ vs ε per Ω (the clean admissibility-vs-noise story)
  figs/tabular_eps_regimes.png     3 panels off / off_on / on (off_on,on at a chosen n_mc) Δπ vs ε

Run:  python fig_eps.py [eps_results.json headline_results.json ...]   (defaults to the standard dirs)
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

from regularizers import REGKEYS, COLORS, SHORT

DEFAULT_EPS_RUN = "data/tabular/run_eps_ablation/results.json"
DEFAULT_HEADLINE = "data/tabular/run_20260629_182331/results.json"   # supplies the ε=0.2 point


def load_many(paths):
    man, results = None, []
    for p in paths:
        if not os.path.exists(p):
            print(f"  (skip missing {p})"); continue
        R = json.load(open(p))
        man = man or R["manifest"]
        results.extend(R["results"])
    if not results:
        sys.exit("no results found")
    return man, results


def agg_eps(results):
    """out[regime][rk][nm][eps] = (mean, ci95).  Pools finals across MDPs; cells without an 'eps'
    field (the headline run) default to 0.2."""
    bucket = {}
    for r in results:
        eps = float(r.get("eps", 0.2))
        bucket.setdefault((r["regime"], r["rk"], r["nm"], eps), []).extend(r["finals"])
    out = {}
    for (reg, rk, nm, eps), fin in bucket.items():
        f = np.asarray(fin, float)
        ci = float(f.std(ddof=1) / np.sqrt(len(f)) * 1.96) if len(f) > 1 else 0.0
        out.setdefault(reg, {}).setdefault(rk, {}).setdefault(nm, {})[eps] = (float(f.mean()), ci)
    return out


def _kl_kw(rk, lw=1.7):
    return dict(lw=3.4 if rk == "kl" else lw, zorder=8 if rk == "kl" else 3)


def _plot_eps(ax, agg, reg, nm, eps_mark=0.2):
    epss = sorted({e for rk in REGKEYS for e in agg.get(reg, {}).get(rk, {}).get(nm, {})})
    for rk in REGKEYS:
        d = agg[reg][rk][nm]
        xs = [e for e in epss if e in d]
        mu = np.array([d[e][0] for e in xs]); ci = np.array([d[e][1] for e in xs])
        ax.plot(xs, mu, marker="o", ms=4, color=COLORS[rk], label=SHORT[rk], **_kl_kw(rk))
        ax.fill_between(xs, mu - ci, mu + ci, color=COLORS[rk], alpha=0.13, zorder=2)
    ax.axvline(eps_mark, color="#999", ls="--", lw=0.9)
    ax.set_xlabel("transition noise  ε"); ax.grid(alpha=0.2)
    return epss


def fig_offpolicy(agg, man):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _plot_eps(ax, agg, "off", 1)
    ax.set_ylabel("off-policy gap  Δπ  (mean TV vs π*)")
    ax.set_title("Off-policy recovery vs transition noise ε (α fixed at the ε=0.2 calibration)\n"
                 "RKL stays lowest at every ε · 100 independent MDPs · ±95% CI", fontsize=10)
    ax.legend(fontsize=8, ncol=2, title="Ω (RKL = admissible)")
    fig.tight_layout(); p = "figs/tabular_eps_offpolicy.png"; fig.savefig(p, dpi=140); plt.close()
    return p


def fig_regimes(agg, man, nm_pick=1):
    regimes = [r for r in ("off", "off_on", "on") if r in agg]
    titles = {"off": "off-policy (n_mc irrelevant)",
              "off_on": f"off-on-policy / Dyna (n_mc={nm_pick})",
              "on": f"on-policy (n_mc={nm_pick})"}
    fig, axes = plt.subplots(1, len(regimes), figsize=(5.2 * len(regimes), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, reg in zip(axes, regimes):
        _plot_eps(ax, agg, reg, 1 if reg == "off" else nm_pick)
        ax.set_title(titles[reg], fontsize=10)
    axes[0].set_ylabel("policy gap  Δπ  (mean TV vs π*)")
    axes[-1].legend(fontsize=8, ncol=2, title="Ω (RKL = admissible)")
    fig.suptitle("ε ablation — Δπ vs transition noise across regimes (α fixed) · 100 MDPs, ±95% CI",
                 fontsize=11, y=1.0)
    fig.tight_layout(); p = "figs/tabular_eps_regimes.png"; fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(); return p


def main():
    paths = sys.argv[1:] or [DEFAULT_EPS_RUN, DEFAULT_HEADLINE]
    man, results = load_many(paths)
    os.makedirs("figs", exist_ok=True)
    agg = agg_eps(results)
    epss = sorted({float(r.get("eps", 0.2)) for r in results})
    print(f"[data] {len(results)} cells · ε values present={epss} · "
          f"MDPs={len({r['mi'] for r in results})}")
    outs = [fig_offpolicy(agg, man), fig_regimes(agg, man, nm_pick=1)]
    print("[saved]\n  " + "\n  ".join(outs))


if __name__ == "__main__":
    main()
