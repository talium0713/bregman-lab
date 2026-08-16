# Off-policy permissibility — Python port (verifiable)

A clean Python reimplementation of the experiment currently written in JS
(`src/core/math.js` + `src/tasks/task3-admissibility.js`), so you can read it against the
paper `Off_policy_admissibility.pdf` and confirm it does what you intend — **especially the
inner term `C_Ω(s')`**.

```
python/
  regularizers.py   the 7 divergences: f, f', Ω, ∇Ω, Bregman, soft-argmax, Φ, Φ', C_Ω
  mdp.py            layered ε-stochastic MDP + regularized backward induction (solve_dp)
  inner_term.py     ★ C_Ω(s'): exact, Eq-15 estimator, Φ-form estimator, off-policy, c(s') probe
  dpo.py            preference data + trajectory-DPO training (A/B grad modes, on/off-policy)
  experiments.py    quick demo of the §4 questions (fast iteration)
  run_part3.py      ★ the FULL Part-3 experiment: every JS-lab panel at real settings
  verify.py         7 numerical checks (regression test — all must PASS)
  review.html       cumulative answer log (git-tree of rounds) + static results panel
  app.py            ★ Streamlit live console — sliders + Run buttons, runs the experiments in-browser
```

### Live console (Streamlit)

The browser can't run Python natively, so this serves a tiny local Python process and renders
widgets in the browser; heavy runs are cached so a slider only recomputes what changed.

```bash
python3 -m pip install --user streamlit      # once
python3 -m streamlit run app.py              # opens http://localhost:8501
```

Tabs: ① calibration & permissibility · ② §4.2 inner-term variance · ③ §4.3 n_mc sweep (on/off,
behind a Run button — the heavy one) · ④ policy recovery π* vs π_θ · ⑤ α-div morph. Sidebar
sliders: peak, ε, depth, α-div parameter a, N_pairs, steps, seeds, batch. (`review.html` stays as
the written-up work log; `app.py` is the interactive console.)

Run (no deps beyond numpy + matplotlib):

```bash
python run_part3.py          # FULL run (~3 min): figs/part3_*.png + results_data.js
QUICK=1 python run_part3.py  # fast smoke (2 seeds / 80 steps)
python verify.py             # 7 checks, must all PASS
open review.html             # read answers + live results (reads results_data.js on file://)
```

`run_part3.py` produces: α–peak sweep, C_Ω statistics, training curves, policy-gap bars over the
`n_mc` sweep, π* vs π_θ panels, the extreme off-policy comparison, and the KL-variance proof —
all with the shared palette. Tune `SEEDS/STEPS/NMCS/EPS/DEPTH/TARGET` at the top of the file.

---

## How `C_Ω(s')` is implemented — and how to read it

The implicit reward (Eq 14) is
```
r(s,a) = α[∇Ω(π*(·|s))]_a + β(s) − γα·E_{s'}[ C_Ω(π*(·|s')) ] − γ E_{s'}[β(s')]
                                        └──── inner term ────┘
C_Ω(π;π_ref) = E_{a~π}[ [∇Ω(π)]_a ] − Ω(π) = E_{a~π}[ Φ(u_a) ],  Φ(u)=f'(u)−f(u)/u, u=π/ref  (Eq 13)
```

### No KL short-circuit — and no hardcoded constant either

There is **no `const_C` flag, no `if KL: return constant`, and no `Phi = lambda u: 1.0`
literal** anywhere. The inner integrand and its derivative are **computed from `f, f', f''`**
for every Ω (`regularizers.Regularizer.Phi/dPhi`):

```
Φ(u)  = f'(u) − f(u)/u            # KL → (ln u + 1) − (u ln u)/u = 1     (arithmetic, not a literal)
Φ'(u) = f''(u) − f'(u)/u + f(u)/u²   # KL → 0
```

KL is sampled by the **identical** fully-sampled procedure as the other six; each draw evaluates
`Φ(u_{a_i})`, and for KL that arithmetic simply lands on `1` every time. So KL is *"always
sampling, but the value is constant"* — exactly the fair behaviour you asked for. Verified: the
generic `Φ/Φ'` match the old closed forms to ~1e-15, `Φ_KL ≡ 1` (spread ~0), `Φ'_KL ≈ 0` (~1e-15
residual, *not* a literal 0). `regularizers.is_permissible(rk)` then CHECKS `Var_a[Φ]≈0` over
random policies — `True` for `kl` only.

### Why the (single, fully-sampled) Φ-form estimator gives KL zero variance

There is exactly **one** estimator, used everywhere (`C_hat_phi`, `dpo.draw_inner`): it samples
the **whole integrand** `Ĉ = (1/n)Σ_i Φ(u_{a_i})` with `Φ = f' − f/u`. For KL each draw is
`(ln u_{a_i}+1) − ln u_{a_i} = 1`, so the per-sample fluctuation cancels *within* each draw →
variance exactly 0. KL is sampled identically to the rest; it just lands on the constant. For
non-KL, `Φ` varies with `u`, so the variance is `Var_π[Φ]/n > 0`. This shows up directly in the
C_Ω-statistics panel (`mc_sd` column: KL≈0) and the single-state `std(Ĉ)` figure (KL≡0, others
∝ 1/√n). `verify.py` CHECK 5 asserts both.

(The earlier "freeze Ω exact and sample only f'" variant — which *would* give KL nonzero
variance — has been removed to avoid confusion. You always sample the whole integrand.)

### The "c(s') transformation" — `c_violation` in `inner_term.py`

Eq 16 reduces the action-indexed inner quantity to a single number per next state, `c(s')`.
That reduction is exact **only when `s'` identifies the action that produced it** — i.e. for
deterministic transitions (the paper's tree, `ε=0`). The JS adds ε-stochastic transitions and
probes what breaks:
```
post(a | s') ∝ π_behave(a)·P(s'|a)        # posterior over which a generated s'
violation(ε) = sqrt( Σ_{s'} P(s')·Var_{a|s'}[ c_a ] )
```
At `ε=0` each `s'` has a one-action posterior ⇒ `violation=0` ⇒ `c(s')` is well-defined
(**permissible**). As `ε` grows, an *action-indexed* `c(a')` can no longer be folded into a
state function and leaks irreducible variance. Verified: `violation(0)=0.000`,
`violation(0.2)=0.293`. This is the JS author's geometric illustration of Theorem 2's
permissibility boundary; `c_probe.png` reproduces it.

### Arbitrary reference policy `π_ref`

`π_ref` is no longer hardcoded to uniform. It **defaults to uniform over actions**, but every
function accepts a `ref` argument and respects it (`u_a := π_a / ref_a`):

- `regularizers.py`: `omega/grad/breg/C/argmax/d_g_dtheta(... , ref=None)`. The soft-argmax is a
  single ν-bisection solver `π_a = ref_a·(f')⁻¹((Q_a−ν)/α)` that works for any `ref` (KL uses
  the closed form `π_a ∝ ref_a·exp(Q_a/α)`).
- `mdp.solve_dp(..., ref=None)`, `dpo.train_one(..., ref=None)`: pass `ref` as
  `None` (uniform), a `(NA,)` vector (same reference on every state), or a `(depth, SN, NA)`
  array (per-state reference). `mdp.resolve_ref` normalizes all three.

Sanity-checked: with a non-uniform `ref` the whole pipeline (calibration → solve → train) runs,
and **`C_KL = 1` for *any* `ref` and any policy** — KL stays permissible regardless of the
reference, exactly as Theorem 2 says (`c(s')` depends on `s'` only through `π_ref(·|s')`).

### On/off-policy inner term (`draw_inner` in `dpo.py`)
- **on-policy**: draw `n_mc` fresh `a' ~ π(·|s')` and average `Φ` (Eq 15). `n_mc` matters.
- **off-policy (extreme)**: use the single *logged* next action `a'_data` (drawn from the
  uniform behaviour policy), no resampling. The estimate is biased by
  `E_uniform[Φ] − E_{π*}[Φ]`; `n_mc` has no leverage. Sharpest non-permissibility demo.
- **KL**: no short-circuit — `Φ_KL≡1` makes every draw equal to the constant (variance 0) and
  it cancels in `d = S_w − S_l` (equal #inner-terms per fixed-depth trajectory), so KL is
  exactly noise-free at every `n_mc` and in both A/B modes, *by the math*.

---

## Faithfulness notes vs the paper (so you can reconcile intent)

The Python port reproduces the **JS experiment** exactly (same math, same defaults). Where
the JS already diverges from the paper's §4.1 setup, you'll see it here too — flagged so you
can choose:

| item | paper §4.1 / Table | JS & this port | how to align |
|---|---|---|---|
| environment | deterministic complete tree, W=3 H=3, \|S\|=40 | layered MDP, SN=3/layer, **ε-stochastic** | set `EPS=0` for deterministic; the tree topology itself is a larger change |
| rewards | leaf-only `~N(0,1)`, internal = 0 | `Uniform(−0.8,0.8)` on **every** (l,s,a) | edit `new_rewards` in `mdp.py` |
| α-divergence | `α_div = 0.1` | **`a = 0.5`** (between RKL & KL); any `a` via `make_adiv(a)` | set `regularizers.DEFAULT_ADIV_A` or pass `a` |
| `C_Ω` estimator | Eq 15 as written (`f'−Ω`) | `Φ`-form (integrand) | **keep `Φ`-form**; reword Eq 15 in the paper as `(1/n)ΣΦ(u)` (only then is KL exactly 0-variance) |
| Δπ metric | occupancy-weighted (Eq 20) | **uniform** weight over states | weight `gap()` by `d_{π*}` in `dpo.py` |
| BT oracle temp | `β_or = 1` | implicit `1` (return diff) | — matches |
| seeds | 5 | configurable (`n_seeds`) | — |

None of these change the *qualitative* permissibility result (KL flat across `n_mc`; non-KL
improve with `n_mc`), but they change absolute numbers, so pick deliberately before quoting
figures in the paper.
