"""
fig_paper_layered_t05t06.py — draws t05 (peaked reference) and t06 (alpha family vs MC budget)
from the JSON written by run_t05_peaked.py / run_t06_alpha_nmc.py. No training here.

  figs/t05_peaked_reference   same two panels as t03, but pi_ref = (0.6, 0.2, 0.2)
  figs/t06_alpha_sweep        policy gap along the alpha family against the MC budget

Run from python/:  python fig_paper_layered_t05t06.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regularizers import REGKEYS, COLORS, SHORT, SHORT_TEX

T05 = "data/tabular/run_t05_peaked.json"
T06 = "data/tabular/run_t06_alpha_nmc.json"


def _kl_kw(rk, lw=1.8):
    return dict(lw=3.2 if rk == "kl" else lw, zorder=8 if rk == "kl" else 3)


def _save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return f"{stem}.png"


def _mean_ci(xs):
    a = np.asarray(xs, float)
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a))


# ─────────────────────────────────────────────────────────────── t05
def fig_t05():
    d = json.load(open(T05)); cfg = d["config"]
    nmc = cfg["nmc"]; ref = cfg["peaked_ref"]; n_mdp = cfg["n_mdp"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
    ymax = 0.0
    for ax, reg in zip(axes, ["on", "off"]):
        flat = reg == "off"
        for rk in REGKEYS:
            if flat:
                m, ci = _mean_ci(d["off"][rk])
                c = np.full(len(nmc), m); cib = np.full(len(nmc), ci)
                ax.plot(nmc, c, color=COLORS[rk], label=SHORT_TEX[rk], **_kl_kw(rk))
            else:
                stats = [_mean_ci(d["on"][str(n)][rk]) for n in nmc]
                c = np.array([s[0] for s in stats]); cib = np.array([s[1] for s in stats])
                ax.plot(nmc, c, marker="o", ms=4, color=COLORS[rk], label=SHORT_TEX[rk], **_kl_kw(rk))
            ax.fill_between(nmc, c - cib, c + cib, color=COLORS[rk], alpha=0.13, zorder=2)
            ymax = max(ymax, float((c + cib).max()))
        ax.set_xscale("log", base=2); ax.set_xticks(nmc); ax.set_xticklabels(nmc, fontsize=8)
        ax.set_xlabel("MC budget  $n_{\\mathrm{mc}}$"); ax.grid(alpha=0.2)
        ax.set_title("on-policy (fresh rollouts)" if not flat else
                     "off-policy (single logged $a'$)", fontsize=10)
        src = (lambda rk: _mean_ci(d["off"][rk])) if flat else \
              (lambda rk: _mean_ci(d["on"][str(nmc[-1])][rk]))
        txt = "\n".join(f"{SHORT[rk]} {src(rk)[0]:.3f}$\\pm${src(rk)[1]:.3f}" for rk in REGKEYS)
        ax.text(0.985, 0.97, f"$\\Delta_\\pi$ at $n_{{\\mathrm{{mc}}}}$={1 if flat else nmc[-1]}\n{txt}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.0, zorder=9,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.8))
        if flat:
            ax.text(0.5, 0.055, "$n_{\\mathrm{mc}}$ does not enter this estimator:\n"
                                "the inner term is the single logged $a'$",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color="#444",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="#ccc", pad=1.4))
    axes[0].set_ylabel(r"policy gap  $\Delta_\pi$ = mean $\mathrm{TV}(\pi_\theta\,\|\,\pi^\star)$")
    axes[0].legend(fontsize=7.5, ncol=2, loc="lower left")
    for ax in axes:
        ax.set_ylim(0, ymax * 1.30)
    fig.suptitle(f"Recovery under a PEAKED reference  $\\pi_{{\\mathrm{{ref}}}}$={tuple(ref)} — "
                 f"layered MDP, {n_mdp} MDPs, $\\pm$95% CI", fontsize=11, y=1.0)
    fig.tight_layout()
    p = _save(fig, "figs/t05_peaked_reference")
    tab = {rk: {"on_max_budget": _mean_ci(d["on"][str(nmc[-1])][rk]),
                "off": _mean_ci(d["off"][rk])} for rk in REGKEYS}
    return p, tab


# ─────────────────────────────────────────────────────────────── t06
def fig_t06():
    d = json.load(open(T06)); cfg = d["config"]
    A = np.asarray(cfg["a_grid"], float); nmc = cfg["nmc"]
    # drop the dense near-1 probe points (kept only to resolve the old knife-edge); keep a=1
    keep = [i for i, a in enumerate(A) if not (0.9 < a < 1.1 and abs(a - 1.0) > 1e-9)]
    Ak = A[keep]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    cmap = plt.get_cmap("viridis")
    for j, nm in enumerate(nmc):
        arr = np.asarray(d["on"][str(nm)], float)[:, keep]        # (mdp, a)
        m = arr.mean(0); ci = 1.96 * arr.std(0, ddof=1) / np.sqrt(arr.shape[0])
        col = cmap(0.15 + 0.6 * j / max(len(nmc) - 1, 1))
        ax.plot(Ak, m, marker="o", ms=3.5, lw=1.7, color=col,
                label=f"on-policy, $n_{{\\mathrm{{mc}}}}$={nm}")
        ax.fill_between(Ak, m - ci, m + ci, color=col, alpha=0.15)
    arr = np.asarray(d["off"], float)[:, keep]
    m = arr.mean(0); ci = 1.96 * arr.std(0, ddof=1) / np.sqrt(arr.shape[0])
    ax.plot(Ak, m, marker="s", ms=3.5, lw=2.2, color="#c0392b", ls="--",
            label="off-policy (single logged $a'$)")
    ax.fill_between(Ak, m - ci, m + ci, color="#c0392b", alpha=0.13)

    ax.axvline(1.0, color=COLORS["kl"], ls=":", lw=1.2, alpha=0.75)
    ax.annotate(r"$\alpha=1$: RKL", xy=(1.0, 0.06), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points", fontsize=8.5,
                color=COLORS["kl"], va="bottom", ha="left")
    ax.set_xlabel(r"divergence-family parameter  $\alpha$   "
                  r"($\alpha\!\to\!0$: FKL · $\alpha\!=\!1$: RKL · $\alpha\!=\!2$: $\chi^2$)")
    ax.set_ylabel(r"policy gap  $\Delta_\pi$ = mean $\mathrm{TV}(\pi_\theta\,\|\,\pi^\star)$")
    ax.grid(alpha=0.2); ax.legend(fontsize=8, loc="upper center", ncol=2)
    ax.set_title(f"$\\alpha$-family vs Monte-Carlo budget — layered MDP, {cfg['n_mdp']} MDPs, "
                 f"$\\pm$95% CI (temperature pinned to RKL@peak {cfg['peak_ref']})", fontsize=10)
    fig.tight_layout()
    p = _save(fig, "figs/t06_alpha_sweep")
    i1 = int(np.argmin(np.abs(Ak - 1.0)))
    tab = {"a_at_1": float(Ak[i1]),
           "off_at_1": float(np.asarray(d["off"], float)[:, keep][:, i1].mean()),
           "off_mean_off1": float(np.delete(m, i1).mean())}
    return p, tab


def main():
    os.makedirs("figs", exist_ok=True)
    made = []
    if os.path.exists(T05):
        p, tab = fig_t05(); made.append(p)
        print("\nTABLE 5 — peaked reference (pi_ref = 0.6/0.2/0.2), mean ± 95% CI over 100 MDPs")
        print(f"  {'Omega':7s}{'on (max budget)':>22s}{'off-policy':>22s}")
        for rk in REGKEYS:
            (a, ac), (b, bc) = tab[rk]["on_max_budget"], tab[rk]["off"]
            print(f"  {SHORT[rk]:7s}{a:>13.3f}±{ac:.3f}{b:>15.3f}±{bc:.3f}"
                  + ("   <- no off-policy penalty" if rk == "kl" else ""))
    if os.path.exists(T06):
        p, tab = fig_t06(); made.append(p)
        print(f"\nTABLE 6 — alpha family: off-policy gap at alpha=1 (RKL) {tab['off_at_1']:.3f} "
              f"vs mean over the rest of the family {tab['off_mean_off1']:.3f}")
    print("\n[saved]" + "".join(f"\n  {m} (+ .pdf)" for m in made) if made else "[nothing to draw]")


if __name__ == "__main__":
    main()
