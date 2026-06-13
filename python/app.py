"""
Streamlit interactive console for the off-policy-admissibility experiments.

Run:   python3 -m pip install streamlit        # once
       streamlit run app.py                     # opens http://localhost:8501

Why Streamlit: the browser can't run Python natively, so this serves a small local Python
process and renders widgets in the browser. Heavy runs are cached with @st.cache_data, so
moving a slider only recomputes what actually changed. (The static review.html stays as the
written-up work log; this app is the live experiment console.)
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
from mdp import new_rewards, uniform_pis, solve_dp, SN, NA
from inner_term import C_exact, single_state_variance, trajectory_variance, c_violation
from dpo import TrainConfig, make_dataset, make_dataset_policy, train_one
from experiments import calibrate, peakiness, c_stats, mean_std

GAMMA = 0.9
st.set_page_config(page_title="Off-policy admissibility lab", layout="wide")


def kl_kw(rk, base=1.6):
    return dict(lw=3.2 if rk == "kl" else base, zorder=8 if rk == "kl" else 3)


# ───────────────────────── sidebar settings ─────────────────────────
st.sidebar.title("⚙ Settings")
peak = st.sidebar.slider("target peak  (mean_s max_a π*)", 0.40, 0.90, 0.60, 0.05)
eps = st.sidebar.slider("ε  (transition noise)", 0.0, 0.5, 0.20, 0.05)
depth = st.sidebar.slider("depth (layers)", 2, 5, 4)
adiv_a = st.sidebar.slider("α-div parameter a  (0→RKL, 0.5→Hellinger, 1→KL, 2→χ²)", 0.1, 2.0, 0.5, 0.1)
npairs = st.sidebar.select_slider("N_pairs (preference data)", [600, 2000, 6000], value=6000)
steps = st.sidebar.select_slider("SGD steps", [100, 200, 400, 800], value=400)
seeds = st.sidebar.slider("seeds", 1, 5, 3)
batch = st.sidebar.select_slider("batch size", [4, 8, 16, 32], value=16)
st.sidebar.caption("Heavy runs are cached — changing a slider only recomputes what depends on it.")

# keep the α-divergence member consistent with the slider on every rerun (cache keys include adiv_a)
rg.REG["adiv"] = make_adiv(adiv_a)

st.title("Off-policy admissibility — live experiment console")
st.caption("KL is the unique off-policy admissible regularizer: its inner term C_Ω is constant, "
           "so it needs no next-state Monte-Carlo sample. Tune the settings and run the panels below.")


# ───────────────────────── cached compute ─────────────────────────
@st.cache_data(show_spinner=False)
def rewards_for(depth, mdp_seed=0):
    return new_rewards(depth, np.random.default_rng(mdp_seed))


@st.cache_data(show_spinner=False)
def calib(peak, eps, depth, adiv_a, mdp_seed=0):
    rg.REG["adiv"] = make_adiv(adiv_a)
    rew = rewards_for(depth, mdp_seed)
    return calibrate(rew, peak, GAMMA, eps)


@st.cache_data(show_spinner="Running §4.2 inner-term variance…")
def ss_var(adiv_a, n_actions=100):
    rg.REG["adiv"] = make_adiv(adiv_a)
    ns = [2 ** k for k in range(2, 11)]
    return ns, {rk: single_state_variance(rk, ns, n_actions=n_actions, seed=1) for rk in REGKEYS}


@st.cache_data(show_spinner="Running §4.2 trajectory variance…")
def tj_var(adiv_a):
    rg.REG["adiv"] = make_adiv(adiv_a)
    Hs = list(range(1, 9))
    return Hs, {rk: trajectory_variance(rk, Hs, seed=2) for rk in REGKEYS}


def nmc_sweep(peak, eps, depth, npairs, steps, seeds, batch, adiv_a, nmc_list, n_mdp, progress_cb=None):
    rg.REG["adiv"] = make_adiv(adiv_a)
    raw = {key: {rk: {nm: [] for nm in nmc_list} for rk in REGKEYS} for key in ("on", "off")}
    pol1 = {"on": {}, "off": {}}
    a0 = r0 = None
    total = n_mdp * len(REGKEYS) * len(nmc_list)
    done = 0
    for mi in range(n_mdp):
        rng = np.random.default_rng(10 + mi)
        rew = new_rewards(depth, rng)
        al = calibrate(rew, peak, GAMMA, eps)
        data_off = make_dataset(rew, eps, GAMMA, rng, npairs)
        if mi == 0:
            a0, r0 = al, rew
        for rk in REGKEYS:
            sol = solve_dp(rk, rew, uniform_pis(depth), al[rk], GAMMA, eps)
            data_on = make_dataset_policy(rew, eps, GAMMA, rng, npairs, sol.pistar)
            for nm in nmc_list:
                for sd in range(seeds):
                    cfg = TrainConfig(gamma=GAMMA, steps=steps, batch=batch, grad_mode="A", n_mc=nm)
                    _, gon, pon = train_one(rk, rew, al[rk], data_on, cfg, eps, rng)
                    _, gof, pof = train_one(rk, rew, al[rk], data_off, cfg, eps, rng)
                    raw["on"][rk][nm].append(gon); raw["off"][rk][nm].append(gof)
                    if mi == 0 and nm == nmc_list[0] and sd == 0:
                        pol1["on"][rk] = pon; pol1["off"][rk] = pof
                done += 1
                if progress_cb:
                    progress_cb(done / total, f"MDP {mi+1}/{n_mdp} · {SHORT[rk]} · n_mc={nm}  ({done}/{total})")
    agg = {key: {rk: {nm: mean_std(raw[key][rk][nm]) for nm in nmc_list} for rk in REGKEYS}
           for key in ("on", "off")}
    return a0, r0, agg, pol1


# ───────────────────────── tabs ─────────────────────────
t1, t2, t3, t4, t5 = st.tabs(
    ["① Calibration", "② §4.2 variance", "③ §4.3 n_mc sweep", "④ Policy recovery", "⑤ α-div morph"])

# ── ① calibration + admissibility ──
with t1:
    st.subheader("α calibration & admissibility (at the chosen peak)")
    alphas = calib(peak, eps, depth, adiv_a)
    rew = rewards_for(depth)
    rows = []
    for rk in REGKEYS:
        c0 = C_exact(rk, solve_dp(rk, rew, uniform_pis(depth), alphas[rk], GAMMA, eps).pistar[0, 0])
        rows.append({"Ω": REG[rk].label, "α": round(alphas[rk], 3),
                     "peak": round(peakiness(rk, rew, alphas[rk], GAMMA, eps), 3),
                     "C_Ω(π*)": round(c0, 3), "admissible": "✓" if is_admissible(rk) else "—"})
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption("α is calibrated per divergence so all share the same policy peak (fair comparison). "
               "admissible (C_Ω constant ⇒ no next-state sample needed) is CHECKED, not hardcoded — only KL.")

# ── ② §4.2 variance ──
with t2:
    st.subheader("§4.2 — inner-term estimator variance (no training)")
    st.caption("Var[Ĉ_Ω]: KL ≡ 0 at every n / horizon; non-KL decay as 1/√n (single state) and grow "
               "as √(H−1) (trajectory). The most direct admissibility signature.")
    ns, ssv = ss_var(adiv_a)
    Hs, tjv = tj_var(adiv_a)
    c1, c2 = st.columns(2)
    with c1:
        fig, (a, b) = plt.subplots(2, 1, figsize=(5.5, 6))
        for rk in REGKEYS:
            mu = np.array([ssv[rk][n]["mean"] for n in ns]); sd = np.array([ssv[rk][n]["std"] for n in ns])
            a.plot(np.log2(ns), mu, color=COLORS[rk], label=REG[rk].label, **kl_kw(rk))
            a.fill_between(np.log2(ns), mu - sd, mu + sd, color=COLORS[rk], alpha=0.1)
            b.plot(np.log2(ns), [ssv[rk][n]["std"] for n in ns], color=COLORS[rk], marker="s", ms=3, **kl_kw(rk))
        a.set_title("single state — Ĉ_Ω (±1σ)"); a.set_xlabel("log2 n"); a.legend(fontsize=6, ncol=2)
        b.set_title("single state — std vs n (KL≡0)"); b.set_xlabel("log2 n")
        fig.tight_layout(); st.pyplot(fig)
    with c2:
        fig, (a, b) = plt.subplots(2, 1, figsize=(5.5, 6))
        for rk in REGKEYS:
            mu = np.array([tjv[rk][H]["mean"] for H in Hs]); sd = np.array([tjv[rk][H]["std"] for H in Hs])
            a.plot(Hs, mu, color=COLORS[rk], label=REG[rk].label, **kl_kw(rk))
            a.fill_between(Hs, mu - sd, mu + sd, color=COLORS[rk], alpha=0.1)
            b.plot(Hs, [tjv[rk][H]["std"] for H in Hs], color=COLORS[rk], marker="s", ms=3, **kl_kw(rk))
        a.set_title("trajectory — Σ_t Ĉ_Ω (±1σ)"); a.set_xlabel("horizon H"); a.legend(fontsize=6, ncol=2)
        b.set_title("trajectory — std vs H (KL≡0, others √(H−1))"); b.set_xlabel("horizon H")
        fig.tight_layout(); st.pyplot(fig)

# ── ③ §4.3 n_mc sweep ──
with t3:
    st.subheader("§4.3 — trained-policy gap Δπ vs Monte-Carlo budget n_mc")
    st.caption("on-policy = π*_Ω rollouts · off-policy = π_ref (uniform) rollouts; both resample n_mc "
               "inner samples. KL stays flat & lowest; non-KL start high at n_mc=1 and decay toward KL.")
    cc1, cc2 = st.columns(2)
    nmc_max = cc1.select_slider("n_mc max (2^k)", [4, 16, 64, 256], value=64)
    n_mdp = cc2.slider("MDP draws to average", 1, 3, 1)
    nmc_list = [2 ** k for k in range(int(np.log2(nmc_max)) + 1)]
    if st.button("▶ Run n_mc sweep", type="primary"):
        pb = st.progress(0.0, text="starting…")
        a0, r0, sweep, pol1 = nmc_sweep(peak, eps, depth, npairs, steps, seeds, batch, adiv_a, nmc_list, n_mdp,
                                        progress_cb=lambda f, t: pb.progress(f, text=t))
        pb.empty()
        st.session_state["sweep"] = (sweep, nmc_list, a0, r0, pol1)
    if "sweep" in st.session_state:
        sweep, nmc_list, a0, r0, pol1 = st.session_state["sweep"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
        for ax, key, ttl in [(axes[0], "on", "on-policy (π*_Ω)"), (axes[1], "off", "off-policy (π_ref)")]:
            for rk in REGKEYS:
                mu = np.array([sweep[key][rk][nm][0] for nm in nmc_list])
                sd = np.array([sweep[key][rk][nm][1] for nm in nmc_list])
                ax.plot(nmc_list, mu, color=COLORS[rk], marker="o", ms=4, label=REG[rk].label, **kl_kw(rk, 1.7))
                ax.fill_between(nmc_list, mu - sd, mu + sd, color=COLORS[rk], alpha=0.12)
            ax.set_xscale("log", base=2); ax.set_xticks(nmc_list); ax.set_xticklabels(nmc_list, fontsize=8)
            ax.set_xlabel("n_mc"); ax.set_title(ttl); ax.grid(alpha=0.2)
        axes[0].set_ylabel("Δπ (mean TV over states)"); axes[1].legend(fontsize=7, ncol=2)
        fig.tight_layout(); st.pyplot(fig)
    else:
        st.info("Set n_mc max / MDP draws, then click **Run n_mc sweep**. (n_mc=256 × seeds × MDPs is the slow part.)")

# ── ④ policy recovery ──
with t4:
    st.subheader("Policy recovery — π* (target) vs π_θ (DPO-trained)")
    nmc_one = st.select_slider("n_mc", [1, 2, 4, 8, 16, 64, 256], value=1)
    if st.button("▶ Train & compare", type="primary"):
        rg.REG["adiv"] = make_adiv(adiv_a)
        rng = np.random.default_rng(0)
        rew = rewards_for(depth)
        al = calib(peak, eps, depth, adiv_a)
        data = make_dataset(rew, eps, GAMMA, rng, npairs)
        pb = st.progress(0.0, text="training…")
        fig, axes = plt.subplots(4, 2, figsize=(11, 10)); axes = axes.ravel()
        for i, rk in enumerate(REGKEYS):
            pb.progress(i / len(REGKEYS), text=f"training {REG[rk].label}  ({i+1}/{len(REGKEYS)}) · n_mc={nmc_one}")
            sol = solve_dp(rk, rew, uniform_pis(depth), al[rk], GAMMA, eps)
            cfg = TrainConfig(gamma=GAMMA, steps=steps, batch=batch, grad_mode="A", n_mc=nmc_one)
            _, g, pol = train_one(rk, rew, al[rk], data, cfg, eps, rng)
            star = [sol.pistar[l, s, a] for l in range(depth) for s in range(SN) for a in range(NA)]
            est = [pol[l, s, a] for l in range(depth) for s in range(SN) for a in range(NA)]
            ax = axes[i]
            ax.plot(star, "--", color="#444", marker="o", ms=2.5, lw=1.0, label="π*")
            ax.plot(est, "-", color=COLORS[rk], marker="s", ms=2.5, lw=2.4 if rk == "kl" else 1.6, label="π_θ")
            ax.set_title(f"{REG[rk].label}  (α={al[rk]:.2f})  Δπ={g:.3f}", fontsize=9, color=COLORS[rk])
            ax.set_ylim(0, 1)
            if i == 0:
                ax.legend(fontsize=7)
        axes[-1].axis("off")
        pb.empty()
        fig.suptitle(f"x = (layer, state, action) · n_mc={nmc_one} · peak={peak}", y=1.0)
        fig.tight_layout(); st.pyplot(fig)
    else:
        st.info("Pick n_mc, then click **Train & compare**.")

# ── ⑤ α-div morph ──
with t5:
    st.subheader("α-divergence morph — same α, sweep the parameter a (RKL → Hellinger → KL → χ²)")
    rew = rewards_for(depth)
    al = calib(peak, eps, depth, adiv_a)
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
    ax.set_xlabel("policy index (layer, state, action)"); ax.set_ylabel(f"π*(a|s), fixed α={alpha}")
    ax.set_title("α-div recovers RKL (a→0), Hellinger (a=0.5), KL (a→1), χ² (a=2)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.2)
    fig.tight_layout(); st.pyplot(fig)
    rg.REG["adiv"] = make_adiv(adiv_a)   # restore the sidebar's a
