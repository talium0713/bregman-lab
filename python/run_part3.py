"""
run_part3.py — the FULL "Part 3 / Off-Policy Admissibility" experiment, end to end.

Reproduces every panel of the JS Part-3 lab with the shared palette, at real settings:
  1. α–peak sweep            peak(α) for all 7 Ω, target line + calibrated α markers
  2. C_Ω statistics          per-Ω spread across states + 1-sample MC std (fully-sampled Φ-form)
  3. training curves         policy gap Δπ vs step, on-policy, n_mc = 1
  4. policy-gap bars         final Δπ over the n_mc sweep {1,4,16}, ± seed std  (on-policy)
  5. π* vs π_θ panels        recovered policy vs target, per divergence
  6. extreme off-policy      on-policy vs single-logged-a' off-policy final Δπ
  7. single-state std(Ĉ)     Φ-form (the only estimator): KL≡0, others ∝ 1/√n
  8. α-div morph             same α, sweep α-div parameter a: policy goes RKL → KL

Writes figs/part3_*.png and results_data.js (read by review.html).  Run:
    python run_part3.py            # full run (~3 min)
    QUICK=1 python run_part3.py    # fast smoke (fewer seeds/steps)
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regularizers import REG, REGKEYS, is_admissible, make_adiv, COLORS, SHORT
from mdp import solve_dp, uniform_pis, new_rewards, state_occupancy, SN, NA
from inner_term import C_exact, c_violation, single_state_variance, trajectory_variance
from dpo import TrainConfig, make_dataset, make_dataset_policy, train_one
from experiments import calibrate, peakiness, c_stats, run_sweep, mean_std

QUICK = bool(os.environ.get("QUICK"))
GAMMA, EPS, DEPTH, BATCH = 0.9, 0.2, 4, 16
PEAKS = [0.6]                    # §4.3 at the calibration anchor (extend to 0.7/0.8 if desired)
NMC_SWEEP = [1, 2] if QUICK else [1, 2, 4, 8, 16, 32, 64, 128, 256]   # §4.3 Monte-Carlo budget sweep
SEEDS = 2 if QUICK else 3
STEPS = 80 if QUICK else 400
NPAIRS = 400 if QUICK else 6000  # preference-pair count. Larger ⇒ data noise ↓ ⇒ KL's noise-free
                                 # inner term shows its advantage (matches the JS N_pairs→10000 trend).


def _kl_kw(rk, base_lw=1.6):
    """KL is drawn thicker and always in front (higher zorder) than the other divergences."""
    return dict(lw=3.4 if rk == "kl" else base_lw, zorder=8 if rk == "kl" else 3)



# ──────────────────────────── figures (KL emphasized, per-peak suffix) ────────────────────────────
def fig_alpha_sweep(rewards, alphas, target, sfx):
    al = np.logspace(np.log10(0.05), np.log10(6), 44)
    plt.figure(figsize=(7.5, 4.5))
    sweep = {}
    for rk in REGKEYS:
        pk = [peakiness(rk, rewards, a, GAMMA, EPS) for a in al]
        sweep[rk] = pk
        plt.plot(al, pk, color=COLORS[rk], label=REG[rk].label, **_kl_kw(rk, 1.5))
        plt.axvline(alphas[rk], color=COLORS[rk], ls=":", lw=0.9, alpha=0.5)
        plt.scatter([alphas[rk]], [target], color=COLORS[rk], s=26, zorder=9 if rk == "kl" else 4)
    plt.axhline(target, color="#888", ls="--", lw=1.1, label=f"target peak={target}")
    plt.xscale("log"); plt.xlabel("regularization weight α"); plt.ylabel("peak  mean_s max_a π*(a|s)")
    plt.title(f"α–peak sweep (target peak = {target})")
    plt.legend(fontsize=7.5, ncol=2); plt.tight_layout()
    plt.savefig(f"figs/part3_alpha_sweep{sfx}.png", dpi=130); plt.close()
    return {"alpha_grid": list(al), "peak": sweep}


def fig_cstats(stats, target, sfx):
    plt.figure(figsize=(7.5, 4))
    x = np.arange(len(REGKEYS)); w = 0.38; floor = 1e-4
    ss = np.array([max(stats[rk]["state_std"], 0) for rk in REGKEYS])
    mc = np.array([max(stats[rk]["mc_sd"], 0) for rk in REGKEYS])
    for i, rk in enumerate(REGKEYS):
        plt.bar(x[i] - w/2, max(ss[i], floor), w, color=COLORS[rk], alpha=0.95)
        plt.bar(x[i] + w/2, max(mc[i], floor), w, color=COLORS[rk], alpha=0.4, edgecolor="w", linewidth=0.6)
    plt.yscale("log"); plt.ylim(floor, max(ss.max(), mc.max(), 1e-2) * 2)
    plt.xticks(x, [SHORT[rk] for rk in REGKEYS]); plt.ylabel("C_Ω spread (log)")
    plt.title(f"C_Ω stats (peak {target}) — solid: std across states · faint: 1-sample Φ-form MC std. KL≈0")
    plt.tight_layout(); plt.savefig(f"figs/part3_cstats{sfx}.png", dpi=130); plt.close()


def fig_curves(out, target, sfx, tag="on-policy"):
    plt.figure(figsize=(6.4, 3.4))
    for rk in REGKEYS:
        curves = out[1][rk]["curves"]
        n = min(len(c) for c in curves)
        m = np.mean([c[:n] for c in curves], axis=0)
        plt.plot(np.linspace(0, STEPS, n), m, color=COLORS[rk], label=REG[rk].label, **_kl_kw(rk, 1.6))
    plt.xlabel("SGD step"); plt.ylabel("policy gap Δπ")
    plt.title(f"Training curves (n_mc=1, {tag}, peak {target})", fontsize=10)
    plt.legend(fontsize=6.5, ncol=2); plt.tight_layout()
    plt.savefig(f"figs/part3_curves{sfx}.png", dpi=120); plt.close()


def fig_policy_panels(rewards, alphas, out, target, sfx, tag="on-policy"):
    # x index = flattened (layer ℓ, state s, action a): DEPTH×SN×NA = 36 entries, grouped by layer.
    fig, axes = plt.subplots(4, 2, figsize=(8.6, 9)); axes = axes.ravel()
    block = SN * NA
    centers = [l * block + (block - 1) / 2 for l in range(DEPTH)]
    for i, rk in enumerate(REGKEYS):
        ax = axes[i]
        sol = solve_dp(rk, rewards, uniform_pis(DEPTH), alphas[rk], GAMMA, EPS)
        pols = out[1][rk]["pols"]
        star, est = [], []
        for l in range(DEPTH):
            for s in range(SN):
                for a in range(NA):
                    star.append(sol.pistar[l, s, a]); est.append(np.mean([P[l, s, a] for P in pols]))
        idx = np.arange(len(star))
        ax.plot(idx, star, "--", color="#444", lw=1.1, marker="o", ms=2.5, label="π* target", zorder=2)
        ax.plot(idx, est, "-", color=COLORS[rk], lw=2.6 if rk == "kl" else 1.8, marker="s", ms=2.6,
                label="π_θ learned", zorder=4)
        for l in range(1, DEPTH):
            ax.axvline(l * block - 0.5, color="#ccc", lw=0.8)
        ax.set_title(f"{REG[rk].label}  (α={alphas[rk]:.2f})", fontsize=10, color=COLORS[rk])
        ax.set_ylim(0, 1); ax.set_xticks(centers); ax.set_xticklabels([f"ℓ{l}" for l in range(DEPTH)])
        ax.tick_params(labelsize=7.5)
        if i == 0:
            ax.legend(fontsize=7)
    axes[-1].axis("off")
    fig.suptitle(f"π* (dashed) vs π_θ (solid) — x = each (layer ℓ, state, action) entry · n_mc=1, {tag}, peak {target}",
                 y=1.0, fontsize=11)
    fig.tight_layout(); fig.savefig(f"figs/part3_policy_panels{sfx}.png", dpi=115); plt.close()


def fig_on_off_bars(out_on, out_off, target, sfx):
    plt.figure(figsize=(8, 4))
    x = np.arange(len(REGKEYS)); w = 0.38
    for i, rk in enumerate(REGKEYS):
        on_mu, on_sd = mean_std(out_on[1][rk]["finals"])
        of_mu, of_sd = mean_std(out_off[1][rk]["finals"])
        ekw = dict(edgecolor="#111", linewidth=2.0) if rk == "kl" else {}
        plt.bar(x[i] - w/2, on_mu, w, yerr=on_sd, color=COLORS[rk], alpha=0.95, capsize=2, **ekw)
        plt.bar(x[i] + w/2, of_mu, w, yerr=of_sd, color=COLORS[rk], alpha=0.4, edgecolor="w", capsize=2)
    plt.xticks(x, [SHORT[rk] for rk in REGKEYS]); plt.ylabel("final gap Δπ (n_mc=1)")
    plt.title(f"solid: on-policy (n_mc=1) · faint: extreme off-policy (1 logged a′), peak {target}. KL lowest")
    plt.tight_layout(); plt.savefig(f"figs/part3_gap_on_off{sfx}.png", dpi=130); plt.close()


def fig_single_state_var(ssv, ns):
    """Paper §4.2 Fig 3: left = estimator value ±1σ (dashed=exact), right = std vs n. KL≡const/0."""
    x = np.log2(ns)
    fig, (axV, axS) = plt.subplots(1, 2, figsize=(11, 4))
    for rk in REGKEYS:
        d = ssv[rk]
        mu = np.array([d[n]["mean"] for n in ns]); sd = np.array([d[n]["std"] for n in ns])
        ex = np.array([d[n]["exact"] for n in ns])
        axV.plot(x, mu, color=COLORS[rk], **_kl_kw(rk, 1.6), label=REG[rk].label)
        axV.fill_between(x, mu - sd, mu + sd, color=COLORS[rk], alpha=0.10)
        axV.plot(x, ex, ls="--", lw=0.8, color=COLORS[rk], alpha=0.7)
        axS.plot(x, sd, color=COLORS[rk], marker="s", ms=4, **_kl_kw(rk, 1.6), label=REG[rk].label)
    axV.set_title("Single state — estimator value Ĉ_Ω (±1σ, dashed=exact)", fontsize=10)
    axV.set_xlabel("log2 n"); axV.set_ylabel("Ĉ_Ω(π; ref)")
    axS.set_title("Single state — estimator std vs n (KL≡0, others ∝ 1/√n)", fontsize=10)
    axS.set_xlabel("log2 n"); axS.set_ylabel("std of Ĉ_Ω"); axS.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig("figs/part3_single_state_var.png", dpi=120); plt.close()


def fig_trajectory_var(tjv, Hs):
    """Paper §4.2 Fig 4: left = inner-term sum Σ_t Ĉ_Ω ±1σ (n_mc=8), right = per-traj std vs H."""
    fig, (axV, axS) = plt.subplots(1, 2, figsize=(11, 4))
    for rk in REGKEYS:
        d = tjv[rk]
        mu = np.array([d[H]["mean"] for H in Hs]); sd = np.array([d[H]["std"] for H in Hs])
        ex = np.array([d[H]["exact"] for H in Hs])
        axV.plot(Hs, mu, color=COLORS[rk], **_kl_kw(rk, 1.6), label=REG[rk].label)
        axV.fill_between(Hs, mu - sd, mu + sd, color=COLORS[rk], alpha=0.10)
        axV.plot(Hs, ex, ls="--", lw=0.8, color=COLORS[rk], alpha=0.7)
        axS.plot(Hs, sd, color=COLORS[rk], marker="s", ms=4, **_kl_kw(rk, 1.6), label=REG[rk].label)
    axV.set_title("Trajectory — inner-term sum Σ_t Ĉ_Ω (±1σ, n_mc=8)", fontsize=10)
    axV.set_xlabel("horizon H"); axV.set_ylabel("Σ_t Ĉ_Ω(π(·|s_{t+1}))")
    axS.set_title("Trajectory — per-traj std vs H (KL≡0, others ∝ √(H−1))", fontsize=10)
    axS.set_xlabel("horizon H"); axS.set_ylabel("std of Σ_t Ĉ_Ω"); axS.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig("figs/part3_trajectory_var.png", dpi=120); plt.close()


def fig_cprobe():
    epsg = np.linspace(0, 0.6, 31)
    plt.figure(figsize=(6.5, 4))
    plt.plot(epsg, [c_violation(e) for e in epsg], color="#9b59d0", lw=2.5, label="action-indexed c(a′)")
    plt.axhline(0, color="#58c08a", lw=2.5, label="state-indexed c(s′)")
    plt.axvline(EPS, ls="--", color="gray", alpha=0.6)
    plt.xlabel("transition noise ε"); plt.ylabel("inner-term violation")
    plt.title("c(s′) admissible at ε=0; action-indexing leaks variance as ε grows")
    plt.legend(); plt.tight_layout(); plt.savefig("figs/c_probe.png", dpi=130); plt.close()
    return {"eps": list(epsg), "violation": [c_violation(e) for e in epsg],
            "eps_mark": EPS, "viol_at_eps": c_violation(EPS)}


def fig_adiv_morph(rewards):
    """α-div parameter sweep at fixed α; π* over (l,s,a). Gradient anchored on RKL & KL colours;
    legend marks the family members it recovers (a→0 RKL, 0.5 Hellinger, →1 KL, 2 χ²). No ground truth."""
    from matplotlib.colors import LinearSegmentedColormap
    alpha = 0.5
    specs = [(0.02, "a→0   (Reverse-KL)"), (0.25, "a=0.25"),
             (0.5,  "a=0.5  (sq. Hellinger)"), (0.75, "a=0.75"),
             (0.999, "a→1   (KL)"), (1.5, "a=1.5"), (2.0, "a=2   (Pearson χ²)")]
    a_min, a_max = 0.0, 2.0
    cmap = LinearSegmentedColormap.from_list("rkl_kl_ext", [(0.0, COLORS["rkl"]), (0.5, COLORS["kl"]), (1.0, "#f39c12")])
    plt.figure(figsize=(12, 4.3)); series = {}
    for a, lab in specs:
        sol = solve_dp("adiv", rewards, uniform_pis(DEPTH), alpha, GAMMA, EPS, reg=make_adiv(a))
        pol = sol.pistar.reshape(-1); series[f"{a}"] = {"label": lab, "pol": pol.tolist()}
        c = cmap((a - a_min) / (a_max - a_min))
        lw = 3.0 if ("KL)" in lab or "Reverse" in lab) else 2.0
        plt.plot(np.arange(len(pol)), pol, marker="s", ms=4.5, lw=lw, color=c, label=lab, zorder=3)
    for l in range(1, DEPTH):
        plt.axvline(l * SN * NA - 0.5, color="#ddd", lw=0.8, zorder=0)
    plt.xlabel("policy index  (layer, state, action)")
    plt.ylabel(f"reach prob  π*(a|s),   fixed α={alpha}")
    plt.title("α-divergence morph: π* sweeps RKL → Hellinger → KL → χ² as the parameter a grows (same α)")
    plt.grid(alpha=0.22); plt.legend(fontsize=8.5, ncol=2, loc="upper right")
    plt.tight_layout(); plt.savefig("figs/part3_adiv_morph.png", dpi=130); plt.close()
    return {"alpha": alpha, "a_min": a_min, "a_max": a_max, "series": series}


# ──────────────────────── §4.3 : trained-policy gap vs Monte-Carlo budget n_mc ────────────────────────
def run_section43(peak, rng, n_mdp=3):
    """
    Paper §4.3 (Fig 5). At the calibrated α for `peak`, sweep the per-step MC budget n_mc and
    measure the trained-policy gap Δπ (mean TV over states), AVERAGED over `n_mdp` reward draws
    (the paper's "leaf-reward draws") for clean bands. Two data regimes (both n_mc-resample inner):
      on-policy  : preference data rolled out from each Ω's own π*_Ω
      off-policy : preference data rolled out from π_ref (uniform)
    KL is flat & lowest in n_mc (admissible, zero inner-term noise); non-KL start high at n_mc=1
    and decay toward KL as the budget grows.  Returns alphas0, rewards0 (MDP 0, for the side
    panels), agg[on/off][rk][nm]=(mean,std) over (MDP×seed), and the n_mc=1 policies of MDP 0.
    """
    raw = {"on": {rk: {nm: [] for nm in NMC_SWEEP} for rk in REGKEYS},
           "off": {rk: {nm: [] for nm in NMC_SWEEP} for rk in REGKEYS}}
    pol1 = {"on": {}, "off": {}}
    alphas0 = rewards0 = None
    for mi in range(n_mdp):
        rewards = new_rewards(DEPTH, rng)
        alphas = calibrate(rewards, peak, GAMMA, EPS)
        data_off = make_dataset(rewards, EPS, GAMMA, rng, NPAIRS)        # π_ref rollouts
        if mi == 0:
            alphas0, rewards0 = alphas, rewards
        for rk in REGKEYS:
            sol = solve_dp(rk, rewards, uniform_pis(DEPTH), alphas[rk], GAMMA, EPS)
            data_on = make_dataset_policy(rewards, EPS, GAMMA, rng, NPAIRS, sol.pistar)   # π*_Ω rollouts
            for nm in NMC_SWEEP:
                for sd in range(SEEDS):
                    cfg = TrainConfig(gamma=GAMMA, steps=STEPS, batch=BATCH, grad_mode="A", n_mc=nm)
                    _, gon, pon = train_one(rk, rewards, alphas[rk], data_on, cfg, EPS, rng)
                    _, gof, pof = train_one(rk, rewards, alphas[rk], data_off, cfg, EPS, rng)
                    raw["on"][rk][nm].append(gon); raw["off"][rk][nm].append(gof)
                    if mi == 0 and nm == 1 and sd == 0:
                        pol1["on"][rk] = pon; pol1["off"][rk] = pof
        print(f"    MDP {mi+1}/{n_mdp} done")
    agg = {key: {rk: {nm: mean_std(raw[key][rk][nm]) for nm in NMC_SWEEP} for rk in REGKEYS}
           for key in ("on", "off")}
    return alphas0, rewards0, agg, pol1


def fig_nmc_sweep(sweep, n_mc_list, peak, sfx):
    fig, (axOn, axOff) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, key, title in [(axOn, "on", "on-policy (π*_Ω)"), (axOff, "off", "off-policy (π_ref)")]:
        for rk in REGKEYS:
            mu = np.array([sweep[key][rk][nm][0] for nm in n_mc_list])
            sd = np.array([sweep[key][rk][nm][1] for nm in n_mc_list])
            ax.plot(n_mc_list, mu, color=COLORS[rk], marker="o", ms=4, label=REG[rk].label, **_kl_kw(rk, 1.7))
            ax.fill_between(n_mc_list, mu - sd, mu + sd, color=COLORS[rk], alpha=0.12)
        ax.set_xscale("log", base=2); ax.set_xticks(n_mc_list); ax.set_xticklabels(n_mc_list, fontsize=8)
        ax.set_xlabel("n_mc (per-step MC budget)"); ax.set_title(f"{title} · peak {peak}", fontsize=11)
        ax.grid(alpha=0.2)
    axOn.set_ylabel("Δπ  (mean TV over states)"); axOff.legend(fontsize=7.5, ncol=2)
    fig.tight_layout(); fig.savefig(f"figs/part3_nmc_sweep{sfx}.png", dpi=120); plt.close()


def fig_panels_single(pols, alphas, rewards, peak, sfx, tag):
    """π* (dashed) vs π_θ (solid) per divergence, single n_mc=1 run. pols[rk] = one policy array."""
    fig, axes = plt.subplots(4, 2, figsize=(8.6, 9)); axes = axes.ravel()
    block = SN * NA
    centers = [l * block + (block - 1) / 2 for l in range(DEPTH)]
    for i, rk in enumerate(REGKEYS):
        ax = axes[i]
        sol = solve_dp(rk, rewards, uniform_pis(DEPTH), alphas[rk], GAMMA, EPS)
        P = pols[rk]
        star = [sol.pistar[l, s, a] for l in range(DEPTH) for s in range(SN) for a in range(NA)]
        est = [P[l, s, a] for l in range(DEPTH) for s in range(SN) for a in range(NA)]
        idx = np.arange(len(star))
        ax.plot(idx, star, "--", color="#444", lw=1.1, marker="o", ms=2.5, label="π* target", zorder=2)
        ax.plot(idx, est, "-", color=COLORS[rk], lw=2.6 if rk == "kl" else 1.8, marker="s", ms=2.6, label="π_θ", zorder=4)
        for l in range(1, DEPTH):
            ax.axvline(l * block - 0.5, color="#ccc", lw=0.8)
        ax.set_title(f"{REG[rk].label}  (α={alphas[rk]:.2f})", fontsize=10, color=COLORS[rk])
        ax.set_ylim(0, 1); ax.set_xticks(centers); ax.set_xticklabels([f"ℓ{l}" for l in range(DEPTH)]); ax.tick_params(labelsize=7.5)
        if i == 0:
            ax.legend(fontsize=7)
    axes[-1].axis("off")
    fig.suptitle(f"π* (dashed) vs π_θ (solid) — n_mc=1, {tag}, peak {peak}", y=1.0, fontsize=11)
    fig.tight_layout(); fig.savefig(f"figs/part3_panels{sfx}.png", dpi=115); plt.close()


# ──────────────────────────────────────── main ────────────────────────────────────────
def main():
    os.makedirs("figs", exist_ok=True)
    rng = np.random.default_rng(0)
    print(f"[settings] peaks={PEAKS} n_mc_sweep={NMC_SWEEP} seeds={SEEDS} steps={STEPS} depth={DEPTH} eps={EPS}")
    rewards = new_rewards(DEPTH, rng)                          # MDP for the §4.2 global panels
    N_MDP = 1 if QUICK else 3                                  # §4.3 averages over this many reward draws

    results = {"meta": {"gamma": GAMMA, "eps": EPS, "depth": DEPTH, "peaks": PEAKS,
                        "n_mc_sweep": NMC_SWEEP, "seeds": SEEDS, "steps": STEPS, "npairs": NPAIRS,
                        "n_mdp": N_MDP, "batch": BATCH, "lr": TrainConfig().lr,
                        "colors": COLORS, "short": SHORT},
               "global": {}, "peaks": {}}

    # ---- GLOBAL panels (peak-independent): §4.2 inner-term variance experiments ----
    print("[global] §4.2 single-state + trajectory inner-term variance")
    ns = [2**k for k in range(2, 11)]
    ssv = {rk: single_state_variance(rk, ns, seed=1) for rk in REGKEYS}
    fig_single_state_var(ssv, ns)
    Hs = list(range(1, 9))
    tjv = {rk: trajectory_variance(rk, Hs, seed=2) for rk in REGKEYS}
    fig_trajectory_var(tjv, Hs)
    results["global"]["single_state_var"] = {"ns": ns, "data": ssv}
    results["global"]["trajectory_var"] = {"Hs": Hs, "data": tjv}
    results["global"]["figs"] = {"single_state_var": "figs/part3_single_state_var.png",
                                 "trajectory_var": "figs/part3_trajectory_var.png"}

    # ---- PER-PEAK : §4.3 n_mc sweep, averaged over N_MDP reward draws ----
    for peak in PEAKS:
        sfx = f"_p{int(round(peak*10))}"
        print(f"\n=== target peak {peak} : §4.3 n_mc sweep (avg over {N_MDP} MDP draws) ===")
        alphas0, rewards0, sweep, pol1 = run_section43(peak, rng, n_mdp=N_MDP)
        sub = {"calibration": {rk: {"label": REG[rk].label, "alpha": alphas0[rk],
                                    "peak": peakiness(rk, rewards0, alphas0[rk], GAMMA, EPS),
                                    "C_exact": C_exact(rk, solve_dp(rk, rewards0, uniform_pis(DEPTH), alphas0[rk], GAMMA, EPS).pistar[0, 0]),
                                    "admissible": is_admissible(rk)} for rk in REGKEYS}}
        sub["alpha_sweep"] = fig_alpha_sweep(rewards0, alphas0, peak, sfx)
        stats = c_stats(rewards0, alphas0, GAMMA, EPS, rng=rng); sub["cstats"] = stats; fig_cstats(stats, peak, sfx)
        fig_nmc_sweep(sweep, NMC_SWEEP, peak, sfx)
        fig_panels_single(pol1["on"], alphas0, rewards0, peak, sfx + "_on", "on-policy (π*_Ω)")
        fig_panels_single(pol1["off"], alphas0, rewards0, peak, sfx + "_off", "off-policy (π_ref)")
        sub["nmc_sweep"] = {"n_mc": NMC_SWEEP,
                            "on": {rk: {nm: dict(zip(("mean", "std"), sweep["on"][rk][nm])) for nm in NMC_SWEEP} for rk in REGKEYS},
                            "off": {rk: {nm: dict(zip(("mean", "std"), sweep["off"][rk][nm])) for nm in NMC_SWEEP} for rk in REGKEYS}}
        sub["figs"] = {"alpha_sweep": f"figs/part3_alpha_sweep{sfx}.png", "cstats": f"figs/part3_cstats{sfx}.png",
                       "nmc_sweep": f"figs/part3_nmc_sweep{sfx}.png",
                       "panels_on": f"figs/part3_panels{sfx}_on.png", "panels_off": f"figs/part3_panels{sfx}_off.png"}
        results["peaks"][f"{peak}"] = sub
        print(f"  Δπ on-policy  n_mc={NMC_SWEEP[0]} → {NMC_SWEEP[-1]}:")
        for rk in REGKEYS:
            lo = sweep["on"][rk][NMC_SWEEP[0]][0]; hi = sweep["on"][rk][NMC_SWEEP[-1]][0]
            print(f"    {SHORT[rk]:5s} {lo:.3f} → {hi:.3f}")

    with open("results_data.js", "w") as f:
        f.write("window.RESULTS = " + json.dumps(results, indent=1) + ";\n")
    print("\n[saved] figs/part3_*.png (per-peak) + results_data.js  →  open/refresh review.html")


if __name__ == "__main__":
    main()
