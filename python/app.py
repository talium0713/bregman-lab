"""
Streamlit interactive console for the off-policy-admissibility experiments.

Run:   python3 -m pip install streamlit        # once
       streamlit run app.py                     # opens http://localhost:8501

The browser can't run Python natively, so this serves a tiny local Python process and renders
widgets in the browser. Heavy runs are cached / gated behind buttons with progress bars.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # find sibling modules from any CWD

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st

import regularizers as rg
from regularizers import REG, REGKEYS, COLORS, SHORT, make_adiv, is_admissible
from mdp import new_rewards, uniform_pis, solve_dp, transP, SN, NA
from inner_term import C_exact, single_state_variance, trajectory_variance
from dpo import TrainConfig, make_dataset, make_dataset_policy, train_one
from experiments import calibrate, peakiness, mean_std

GAMMA = 0.9
ADIV_A = 0.5                       # fixed α-div member for the non-morph tabs (between RKL & KL)
rg.REG["adiv"] = make_adiv(ADIV_A)
st.set_page_config(page_title="Off-policy admissibility lab", layout="wide")


import io


def kl_kw(rk, base=1.6):
    return dict(lw=3.2 if rk == "kl" else base, zorder=8 if rk == "kl" else 3)


def show(fig, container=None):
    """Render a matplotlib fig as a PNG image (small by default; Streamlit's hover
    fullscreen icon enlarges it on click)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=115, bbox_inches="tight")
    plt.close(fig)
    (container or st).image(buf.getvalue(), width="stretch")


# ───────────────────────── sidebar settings ─────────────────────────
st.sidebar.title("⚙ Settings")
peak = st.sidebar.slider("target peak  (mean over states of max_a π*)", 0.40, 0.90, 0.60, 0.05)
eps = st.sidebar.slider("ε  (transition noise)", 0.0, 0.5, 0.20, 0.05)
depth = st.sidebar.slider("depth (layers)", 2, 5, 4)
npairs = st.sidebar.slider("N_pairs (preference data)", 3000, 10000, 6000, 1000)
steps = st.sidebar.select_slider("SGD steps", [100, 200, 400, 800], value=400)
seeds = st.sidebar.slider("seeds  (tabular needs many for stable estimates)", 1, 100, 3)
batch = st.sidebar.select_slider("batch size", [4, 8, 16, 32], value=16)
st.sidebar.caption("Heavy runs are gated behind buttons with progress bars. "
                   "(α-divergence parameter a is swept in its own tab, so it's not a global setting.)")

st.title("Off-policy admissibility — live experiment console")
st.markdown(r"""KL is the unique **off-policy admissible** regularizer: its inner term
$C_\Omega(\pi;\pi_{\mathrm{ref}})=\mathbb{E}_{a\sim\pi}[\nabla_\pi\Omega]_a-\Omega(\pi)$ is **constant**,
so the implicit reward is computable from a transition $(s,a,s')$ alone — no next-state sample
$a'\sim\pi(\cdot|s')$ needed. Every other $\Omega$ inherits a Monte-Carlo inner term whose variance
compounds along trajectories.""")


# ───────────────────────── cached compute ─────────────────────────
@st.cache_data(show_spinner=False)
def rewards_for(depth, mdp_seed=0):
    return new_rewards(depth, np.random.default_rng(mdp_seed))


@st.cache_data(show_spinner=False)
def calib(peak, eps, depth, mdp_seed=0):
    rew = rewards_for(depth, mdp_seed)
    return calibrate(rew, peak, GAMMA, eps)


@st.cache_data(show_spinner="computing α–peak sweep…")
def alpha_peak_curves(eps, depth, mdp_seed=0):
    rew = rewards_for(depth, mdp_seed)
    al = np.logspace(np.log10(0.05), np.log10(6), 40)
    return list(al), {rk: [peakiness(rk, rew, a, GAMMA, eps) for a in al] for rk in REGKEYS}


@st.cache_data(show_spinner="running §4.2 single-state variance…")
def ss_var():
    ns = [2 ** k for k in range(2, 11)]
    return ns, {rk: single_state_variance(rk, ns, seed=1) for rk in REGKEYS}


@st.cache_data(show_spinner="running §4.2 trajectory variance…")
def tj_var():
    Hs = list(range(1, 9))
    return Hs, {rk: trajectory_variance(rk, Hs, seed=2) for rk in REGKEYS}


@st.cache_data(show_spinner="sampling inner values from Dirichlet…")
def c_ablation(peak, eps, depth, conc, n_draws, mdp_seed=0):
    """
    Ablation: replace the actual inner integrand with ARBITRARY inner values sampled from a
    Dirichlet, and measure the n-sample MC std of the inner term under a'~π(·|s').
      c(s')   = one value per state  (state-indexed)  ⇒ no a' dependence ⇒ std ≡ 0 (admissible).
      c(s',a')= one value per state-action            ⇒ a' dependence    ⇒ std ∝ 1/√n (non-admissible).
    Averages over `n_draws` Dirichlet draws across ALL states / state-actions of the MDP. The
    sampling policy π(·|s') is the calibrated admissible-divergence π* at the chosen peak.
    """
    ns = [2 ** k for k in range(0, 9)]               # 1 … 256
    rng = np.random.default_rng(123)
    rew = rewards_for(depth, mdp_seed)
    al = calibrate(rew, peak, GAMMA, eps)
    pol = solve_dp("kl", rew, uniform_pis(depth), al["kl"], GAMMA, eps).pistar   # a'~π*(·|s')
    states = [(l, s) for l in range(depth) for s in range(SN)]
    per_draw_var = []
    for _ in range(n_draws):
        v = 0.0
        for (l, s) in states:
            p = pol[l, s]
            c = rng.dirichlet(np.full(NA, conc))     # c(s',a') ~ Dirichlet over this state's actions
            m = float(np.dot(p, c)); v += float(np.dot(p, c * c) - m * m)   # Var_{a'~π}[c]
        per_draw_var.append(v / len(states))
    var_sa = float(np.mean(per_draw_var))
    res_sa = {n: float(np.sqrt(var_sa / n)) for n in ns}   # c(s',a') MC std
    res_s = {n: 0.0 for n in ns}                           # c(s')  ≡ 0 (state-indexed)
    return ns, res_sa, res_s


def nmc_run(peak, eps, depth, npairs, steps, seeds, batch, nmc_list, n_mdp):
    """ONE shared training run: trains every Ω over the n_mc list, on-policy (π*_Ω rollouts) and
    off-policy (π_ref rollouts), averaged over n_mdp reward draws (mi=0 = the displayed seed-0 MDP).
    Returns the final-gap aggregate AND the trained policies / curves of MDP-0·seed-0 so the SAME
    run feeds both the gap-vs-n_mc plot and the policy-recovery panels."""
    raw = {k: {rk: {nm: [] for nm in nmc_list} for rk in REGKEYS} for k in ("on", "off")}
    pols = {k: {rk: {} for rk in REGKEYS} for k in ("on", "off")}        # MDP0/seed0 policy per nm
    curves = {k: {rk: {} for rk in REGKEYS} for k in ("on", "off")}      # MDP0/seed0 training curve
    total = n_mdp * len(REGKEYS) * len(nmc_list); done = 0
    prog = st.session_state.get("_prog")
    for mi in range(n_mdp):
        rng = np.random.default_rng(mi)                                  # mi=0 → seed 0 (shown MDP)
        rew = new_rewards(depth, rng)
        al = calibrate(rew, peak, GAMMA, eps)
        data_off = make_dataset(rew, eps, GAMMA, rng, npairs)
        for rk in REGKEYS:
            sol = solve_dp(rk, rew, uniform_pis(depth), al[rk], GAMMA, eps)
            data_on = make_dataset_policy(rew, eps, GAMMA, rng, npairs, sol.pistar)
            for nm in nmc_list:
                for sd in range(seeds):
                    cfg = TrainConfig(gamma=GAMMA, steps=steps, batch=batch, n_mc=nm)
                    con, gon, pon = train_one(rk, rew, al[rk], data_on, cfg, eps, rng)
                    cof, gof, pof = train_one(rk, rew, al[rk], data_off, cfg, eps, rng)
                    raw["on"][rk][nm].append(gon); raw["off"][rk][nm].append(gof)
                    if mi == 0 and sd == 0:
                        pols["on"][rk][nm], pols["off"][rk][nm] = pon, pof
                        curves["on"][rk][nm], curves["off"][rk][nm] = con, cof
                done += 1
                if prog is not None:
                    prog.progress(done / total, f"MDP {mi+1}/{n_mdp} · {SHORT[rk]} · n_mc={nm}  ({done}/{total})")
    agg = {k: {rk: {nm: mean_std(raw[k][rk][nm]) for nm in nmc_list} for rk in REGKEYS} for k in ("on", "off")}
    return agg, pols, curves


def fig_recovery(pols_k, nm, alphas, rew, tag):
    """π* (dashed) vs π_θ (solid) over (ℓ,s,a), per Ω, from a stored run's policy at budget nm."""
    depth = rew.shape[0]
    fig, axes = plt.subplots(4, 2, figsize=(9, 8)); axes = axes.ravel()
    for i, rk in enumerate(REGKEYS):
        sol = solve_dp(rk, rew, uniform_pis(depth), alphas[rk], GAMMA, eps)
        pol = pols_k[rk][nm]
        star = [sol.pistar[l, s, a] for l in range(depth) for s in range(SN) for a in range(NA)]
        est = [pol[l, s, a] for l in range(depth) for s in range(SN) for a in range(NA)]
        g = 0.5 * np.mean([abs(star[j] - est[j]) for j in range(len(star))]) * NA
        ax = axes[i]
        ax.plot(star, "--", color="#444", marker="o", ms=2, lw=1.0, label="π*")
        ax.plot(est, "-", color=COLORS[rk], marker="s", ms=2, lw=2.4 if rk == "kl" else 1.5, label="π_θ")
        ax.set_title(f"{REG[rk].label} (α={alphas[rk]:.2f})", fontsize=9, color=COLORS[rk]); ax.set_ylim(0, 1)
        if i == 0:
            ax.legend(fontsize=7)
    axes[-1].axis("off")
    fig.suptitle(f"{tag} · n_mc={nm} · x=(layer,state,action)", y=1.0, fontsize=10)
    fig.tight_layout(); return fig


# ───────────────────────── MDP instance (always shown on top) ─────────────────────────
def fig_mdp(rewards, eps):
    depth = rewards.shape[0]
    act_c = ["#1f9bb0", "#d4699f", "#7b6fcc"]
    fig, ax = plt.subplots(figsize=(11, 2.9))
    for l in range(depth):
        for s in range(SN):
            if l < depth - 1:
                for a in range(NA):
                    tpa = transP(a, eps)
                    for s2 in range(SN):
                        if tpa[s2] < 1e-6:
                            continue
                        ax.plot([l, l + 1], [s, s2], color=act_c[a], lw=0.6 + 2.6 * tpa[s2],
                                alpha=0.2 + 0.4 * tpa[s2], zorder=1, solid_capstyle="round")
    for l in range(depth):
        for s in range(SN):
            ax.scatter([l], [s], s=420, c="white", edgecolors="#333", linewidths=1.2, zorder=3)
            ax.text(l, s, f"s{s}", ha="center", va="center", fontsize=8, color="#222", zorder=4)
            r = rewards[l, s]
            ax.text(l, s - 0.34, f"[{r[0]:+.1f} {r[1]:+.1f} {r[2]:+.1f}]", ha="center",
                    fontsize=6.0, color="#777", zorder=4)
    for a, c in enumerate(act_c):
        ax.plot([], [], color=c, lw=3, label=f"action a{a+1}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=8, frameon=False)
    ax.set_xticks(range(depth)); ax.set_xticklabels([f"layer ℓ{l}" for l in range(depth)], fontsize=8)
    ax.set_yticks(range(SN)); ax.set_yticklabels([f"s{s}" for s in range(SN)], fontsize=8)
    ax.set_ylim(-0.7, SN - 0.4); ax.set_xlim(-0.4, depth - 0.6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    return fig


with st.container():
    st.markdown("#### MDP instance (seed 0) — used by tabs ①②④⑤")
    rew0 = rewards_for(depth)
    show(fig_mdp(rew0, eps))
    st.caption(r"Layered MDP: depth $\times$ 3 states $\times$ 3 actions. Action $a$ reaches its target "
               r"state w.p. $1-\varepsilon$, else uniform over the rest. Node label = the 3 action "
               r"rewards $[r_{a_1}, r_{a_2}, r_{a_3}]\sim\mathrm{Uniform}(-0.8,0.8)$; reference policy "
               r"$\pi_{\mathrm{ref}}$ uniform. (§4.3 averages over a few fresh draws.)")

st.divider()

# ───────────────────────── tabs ─────────────────────────
t1, t2, t3, t4, t5 = st.tabs(
    ["① Calibration", "② §4.2 variance", "③ §4.3 training (gap + recovery)", "④ α-div morph",
     "⑤ c(s') vs c(s',a')"])

# ── ① calibration table + α–peak sweep plot ──
with t1:
    st.subheader("α calibration & admissibility")
    alphas = calib(peak, eps, depth)
    rew = rewards_for(depth)
    rows = []
    for rk in REGKEYS:
        c0 = C_exact(rk, solve_dp(rk, rew, uniform_pis(depth), alphas[rk], GAMMA, eps).pistar[0, 0])
        rows.append({"Ω": REG[rk].label, "α": round(alphas[rk], 3),
                     "peak": round(peakiness(rk, rew, alphas[rk], GAMMA, eps), 3),
                     "C_Ω(π*)": round(c0, 3), "admissible": "✓" if is_admissible(rk) else "—"})
    st.dataframe(rows, width="stretch", hide_index=True)
    st.markdown(r"Each $\Omega$ has a different geometric scale, so $\alpha$ is **calibrated per "
                r"divergence** by bisection until the peak $\bar p=\mathbb{E}_s[\max_a\pi^*_\Omega(a|s)]$ "
                r"matches the target — a fair cross-$\Omega$ comparison. `admissible` "
                r"($C_\Omega$ constant $\Rightarrow$ no next-state sample) is **checked, not hardcoded** — only KL.")
    st.markdown("**α–peak sweep** (paper Fig.): $\\bar p(\\alpha)$ per $\\Omega$, target line, calibrated $\\alpha$ marked.")
    al, curves = alpha_peak_curves(eps, depth)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for rk in REGKEYS:
        ax.plot(al, curves[rk], color=COLORS[rk], label=REG[rk].label, **kl_kw(rk, 1.5))
        ax.axvline(alphas[rk], color=COLORS[rk], ls=":", lw=0.8, alpha=0.5)
        ax.scatter([alphas[rk]], [peak], color=COLORS[rk], s=26, zorder=9 if rk == "kl" else 4)
    ax.axhline(peak, color="#888", ls="--", lw=1.1, label=f"target peak = {peak}")
    ax.set_xscale("log"); ax.set_xlabel("regularization weight α")
    ax.set_ylabel(r"peak  $\mathbb{E}_s[\max_a \pi^*(a|s)]$")
    ax.set_title("α–peak sweep — calibrated α (dots) all hit the common peak"); ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout(); show(fig)

# ── ② §4.2 variance ──
with t2:
    st.subheader("§4.2 — inner-term estimator variance (no training)")
    st.markdown(r"$\mathrm{Var}[\hat C_\Omega]$ is the most direct admissibility signature: **KL $\equiv 0$** "
                r"at every $n$ / horizon; non-KL decay as $1/\sqrt{n}$ (single state) and grow as "
                r"$\sqrt{H-1}$ (trajectory).")
    ns, ssv = ss_var(); Hs, tjv = tj_var()
    c1, c2 = st.columns(2)
    with c1:
        fig, (a, b) = plt.subplots(2, 1, figsize=(5.5, 6))
        for rk in REGKEYS:
            mu = np.array([ssv[rk][n]["mean"] for n in ns]); sd = np.array([ssv[rk][n]["std"] for n in ns])
            a.plot(np.log2(ns), mu, color=COLORS[rk], label=REG[rk].label, **kl_kw(rk))
            a.fill_between(np.log2(ns), mu - sd, mu + sd, color=COLORS[rk], alpha=0.1)
            b.plot(np.log2(ns), [ssv[rk][n]["std"] for n in ns], color=COLORS[rk], marker="s", ms=3, **kl_kw(rk))
        a.set_title(r"single state — $\hat C_\Omega$ (±1σ)"); a.set_xlabel("log2 n"); a.legend(fontsize=6, ncol=2)
        b.set_title(r"single state — std vs n (RKL $\equiv$ 0)"); b.set_xlabel("log2 n")
        fig.tight_layout(); show(fig)
    with c2:
        fig, (a, b) = plt.subplots(2, 1, figsize=(5.5, 6))
        for rk in REGKEYS:
            mu = np.array([tjv[rk][H]["mean"] for H in Hs]); sd = np.array([tjv[rk][H]["std"] for H in Hs])
            a.plot(Hs, mu, color=COLORS[rk], label=REG[rk].label, **kl_kw(rk))
            a.fill_between(Hs, mu - sd, mu + sd, color=COLORS[rk], alpha=0.1)
            b.plot(Hs, [tjv[rk][H]["std"] for H in Hs], color=COLORS[rk], marker="s", ms=3, **kl_kw(rk))
        a.set_title(r"trajectory — $\sum_t \hat C_\Omega$ (±1σ)"); a.set_xlabel("horizon H"); a.legend(fontsize=6, ncol=2)
        b.set_title(r"trajectory — std vs H ($\sqrt{H-1}$)"); b.set_xlabel("horizon H")
        fig.tight_layout(); show(fig)

# ── ③ §4.3 training: ONE run → gap-vs-n_mc + policy recovery (shared) ──
with t3:
    st.subheader("§4.3 — one training run feeds the n_mc gap curves AND policy recovery")
    st.markdown(r"Trains every $\Omega$ over the $n_{mc}$ list, **on-policy** ($\pi^*_\Omega$ rollouts) "
                r"and **off-policy** ($\pi_{\mathrm{ref}}$ rollouts). One run produces both the "
                r"$\Delta_\pi$-vs-$n_{mc}$ curves and the $\pi^*$-vs-$\pi_\theta$ panels (pick $n_{mc}$ below).")
    with st.expander("What is n_mc, and why does it apply to off-policy too?"):
        st.markdown(r"""
**on/off-policy = the behaviour policy that *collected the data***, not whether rollouts happen
(off-policy still rolls out — from $\pi_{\mathrm{ref}}$). **$n_{mc}$ is a training-time quantity:**
to evaluate the non-KL implicit reward at a logged $(s,a,s')$ you must estimate the inner term by
drawing $n_{mc}$ next-state actions $a'\sim\pi_\theta(\cdot|s')$ at each logged $s'$ — in **both**
regimes. KL avoids it ($C_{\mathrm{KL}}\equiv 1$), so KL is flat in $n_{mc}$ while non-KL improve.""")
    cc1, cc2 = st.columns(2)
    nmc_max = cc1.select_slider("n_mc max", [4, 8, 16], value=16)
    n_mdp = cc2.slider("MDP draws to average (mi=0 = the MDP shown above)", 1, 20, 1)
    nmc_list = [2 ** k for k in range(int(np.log2(nmc_max)) + 1)]
    st.caption(f"This run trains 7 Ω × {len(nmc_list)} n_mc × {seeds} seeds × 2 regimes × {n_mdp} MDP "
               f"= {7*len(nmc_list)*seeds*2*n_mdp} fits. Python is much slower than the JS standalone "
               f"(browser JIT), so large `seeds` is heavy. Re-press to recompute.")
    if st.button("▶ Run training", type="primary"):
        st.session_state["_prog"] = st.progress(0.0, text="starting…")
        agg, pols, curves = nmc_run(peak, eps, depth, npairs, steps, seeds, batch, nmc_list, n_mdp)
        st.session_state.pop("_prog", None)
        st.session_state["run"] = (agg, pols, curves, nmc_list)
    if "run" in st.session_state:
        agg, pols, curves, nmc_list = st.session_state["run"]
        rew = rewards_for(depth); al = calib(peak, eps, depth)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
        for ax, key, ttl in [(axes[0], "on", r"on-policy ($\pi^*_\Omega$)"),
                             (axes[1], "off", r"off-policy ($\pi_{\mathrm{ref}}$)")]:
            for rk in REGKEYS:
                mu = np.array([agg[key][rk][nm][0] for nm in nmc_list])
                sd = np.array([agg[key][rk][nm][1] for nm in nmc_list])
                ax.plot(nmc_list, mu, color=COLORS[rk], marker="o", ms=4, label=REG[rk].label, **kl_kw(rk, 1.7))
                ax.fill_between(nmc_list, mu - sd, mu + sd, color=COLORS[rk], alpha=0.12)
            ax.set_xscale("log", base=2); ax.set_xticks(nmc_list); ax.set_xticklabels(nmc_list, fontsize=8)
            ax.set_xlabel("n_mc"); ax.set_title(ttl); ax.grid(alpha=0.2)
        axes[0].set_ylabel(r"$\Delta_\pi$ (mean TV over states)"); axes[1].legend(fontsize=7, ncol=2)
        fig.tight_layout(); show(fig)
        st.markdown("**Policy recovery from this run** — π* (dashed) vs π_θ (solid). Pick the budget:")
        nm_sel = st.select_slider("n_mc for the recovery panels", nmc_list, value=nmc_list[0])
        rc1, rc2 = st.columns(2)
        rc1.markdown(r"**on-policy ($\pi^*_\Omega$)**"); show(fig_recovery(pols["on"], nm_sel, al, rew, "on-policy"), rc1)
        rc2.markdown(r"**off-policy ($\pi_{\mathrm{ref}}$)**"); show(fig_recovery(pols["off"], nm_sel, al, rew, "off-policy"), rc2)
        with st.expander("training curves (gap vs SGD step) at the selected n_mc"):
            fig, ax = plt.subplots(figsize=(7, 3.6))
            for rk in REGKEYS:
                cv = curves["on"][rk][nm_sel]
                ax.plot(np.linspace(0, steps, len(cv)), cv, color=COLORS[rk], label=REG[rk].label, **kl_kw(rk, 1.5))
            ax.set_xlabel("SGD step"); ax.set_ylabel("Δπ"); ax.set_title(f"on-policy training curves · n_mc={nm_sel}")
            ax.legend(fontsize=6.5, ncol=2); fig.tight_layout(); show(fig)
    else:
        st.info("Set n_mc max / MDP draws, then click **Run training** — one run fills the gap curves and the recovery panels.")


# ── ④ α-div morph ──
with t4:
    st.subheader("α-divergence morph — same α, sweep the parameter a")
    st.markdown(r"The α-divergence family recovers four of the seven: $a\to 0$ = Reverse-KL, "
                r"$a=0.5$ = sq. Hellinger, $a\to 1$ = KL, $a=2$ = Pearson $\chi^2$. "
                r"Colour is a gradient anchored on the KL & RKL palette.")
    rew = rewards_for(depth)
    alpha = 0.5
    specs = [(0.02, "a→0 (RKL)"), (0.25, "a=0.25"), (0.5, "a=0.5 (Hellinger)"), (0.75, "a=0.75"),
             (0.999, "a→1 (KL)"), (1.5, "a=1.5"), (2.0, "a=2 (χ²)")]
    cmap = LinearSegmentedColormap.from_list("m", [(0.0, COLORS["rkl"]), (0.5, COLORS["kl"]), (1.0, "#f39c12")])
    fig, ax = plt.subplots(figsize=(11, 4))
    for a, lab in specs:
        sol = solve_dp("adiv", rew, uniform_pis(depth), alpha, GAMMA, eps, reg=make_adiv(a))
        pol = sol.pistar.reshape(-1)
        ax.plot(pol, marker="s", ms=4, lw=3 if ("KL)" in lab or "RKL" in lab) else 2,
                color=cmap(a / 2.0), label=lab)
    ax.set_xlabel("policy index (layer, state, action)")
    ax.set_ylabel(r"$\pi^*(a|s)$, fixed $\alpha=%.1f$" % alpha)
    ax.set_title("α-div recovers FKL (a→0), Hellinger (a=0.5), RKL (a→1), χ² (a=2)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.2)
    fig.tight_layout(); show(fig)

# ── ⑤ c(s') vs c(s',a') admissibility ablation (Dirichlet-sampled inner values) ──
with t5:
    st.subheader("c(s') vs c(s',a') — admissibility ablation")
    st.markdown(r"""
Instead of hand-picking the inner values, we **sample them from a Dirichlet** over all states
(for a state-indexed $c(s')$) or all state-actions (for an action-indexed $c(s',a')$), and measure
the $n$-sample Monte-Carlo std of the inner term under $a'\sim\pi^*(\cdot|s')$:

- **$c(s')$** (state-indexed): no dependence on $a'$ ⇒ the estimator is **exactly constant** ⇒ std $\equiv 0$ — *admissible* (computable from $(s,a,s')$ alone).
- **$c(s',a')$** (action-indexed): depends on the next action ⇒ you must draw $a'\sim\pi^*(\cdot|s')$ ⇒ std $\propto 1/\sqrt{n}$ — *non-admissible*.

The admissible regularizer is admissible precisely because its inner term is a (constant) $c(s')$;
this ablation shows the principle holds for **any** Dirichlet-sampled $c$ — a state-indexed inner
term costs zero MC variance, a state-action-indexed one costs $\propto 1/\sqrt{n}$, regardless of
the specific values.
""")
    cc1, cc2 = st.columns(2)
    conc = cc1.slider("Dirichlet concentration  (smaller = more spread-out c)", 0.1, 5.0, 1.0, 0.1)
    n_draws = cc2.slider("Dirichlet draws to average", 10, 500, 100, 10)
    ns, sa, ss = c_ablation(peak, eps, depth, conc, n_draws)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(np.log2(ns), [sa[n] for n in ns], marker="s", ms=5, lw=2.6, color="#c0392b",
            label="c(s',a') — action-indexed (non-admissible)")
    ax.plot(np.log2(ns), [ss[n] for n in ns], marker="o", ms=5, lw=2.6, color="#27ae60",
            label="c(s') — state-indexed (admissible)")
    ax.set_xlabel(r"log2 n   ($n$ = MC samples $a'\sim\pi^*(\cdot|s')$)")
    ax.set_ylabel("single-state MC std of the inner term")
    ax.set_title(r"$c(s')$ needs no $a'$ sample (std $\equiv$ 0); $c(s',a')$ incurs MC variance $\propto 1/\sqrt{n}$")
    ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout(); show(fig)
    st.caption("c-values are sampled per state / state-action from a Dirichlet (averaged over the draws "
               "above), across every state of the MDP; sampling policy is the calibrated admissible π* "
               "at the current peak/ε.")
