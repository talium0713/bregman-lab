# LLM-scale mechanism experiment — design

Goal: show at LLM scale the §4.2 mechanism — the inner term `C_Ω` is noise-free for admissible Ω
and noisy for every other Ω — and that the pathology **amplifies** with vocabulary size and
off-policyness rather than washing out.

Status: design only. No code written yet. Open decisions marked **[OPEN]**.

## 0. Why LLM scale is not just "bigger"

`Φ(u) = f'(u) − f(u)/u` (regularizers.py:269), `u = π_θ/π_ref`. Expanding each generator:

| Ω | key | Φ(u) | u→0 |
|---|---|---|---|
| reverse KL | `kl` | **1** | constant |
| forward KL | `rkl` | (ln u − 1)/u | −1/u |
| Pearson χ² | `chi2` | u − 1/u | −1/u |
| sq. Hellinger | `hel` | (√u − 1)/u | −1/u |
| JS | `js` | −ln(2/(1+u))/u | −ln2/u |

Every non-admissible Φ diverges as 1/u, and `u→0` **is** the off-policy condition: the logged
token is one the current policy finds unlikely. So estimator variance grows exactly in proportion
to how off-policy the data is.

Tabular (|A|=3) cannot exhibit this — u stays O(1). The memory note already records the symptom:
at |A|=100 the FKL/α-div Φ goes heavy-tailed and hides the 1/√n slope. At |V| = 32k–128k,
u ~ 1e-6 is routine. **The claim is therefore stronger at LLM scale, not weaker.**

## 0b. Relation to Pipano et al. (ICLR 2026) — the nearest prior result

**Displacement-Resistant Extensions of DPO with Nonconvex f-Divergences**, Pipano, Sabach, Asadi,
Ghavamzadeh, ICLR 2026, arXiv:2602.06788. Corollary 1: `f` is *DPO-inducing* (the partition
constant cancels) **iff** `lim_{t→0+} f'(t) = −∞`. They note this makes tractability and
divergence-at-zero the same condition.

**This is a different condition from admissibility, and strictly weaker.** Solving `Φ(u) ≡ c`:

    f'(u) − f(u)/u = c,  g := f/u  ⟹  g' = c/u  ⟹  f(u) = c·u ln u + k·u

With the standard normalization f(1) = 0 this forces k = 0, so

> **admissible ⟺ f = c·u ln u — reverse KL, up to scale, and nothing else.**

Verified numerically: `u ln u` and `3u ln u` give std[Φ] ~ 1e-16; `u ln u − u + 1` (the standard
f-divergence normalization) does not. This is exactly the affine-variance issue recorded in
`kl_normalization.md` — same phenomenon, now with a closed-form characterization.

Measured over the 6 f-divergences in `regularizers.py`:

| Ω | std[Φ] | admissible | f'(t→0+) | DPO-inducing (Pipano) |
|---|---|---|---|---|
| RKL | 3.3e-16 | **yes** | −∞ (log, 2.3/decade) | yes |
| α-div | 3.0e+05 | no | −∞ | yes |
| FKL | 2.1e+06 | no | −∞ | yes |
| JS | 1.0e+05 | no | −∞ (log) | yes |
| Hel | 1.5e+05 | no | −∞ | yes |
| χ² | 1.5e+05 | no | **→ −2 (finite)** | **no** |

So Pipano's condition admits five of six; admissibility admits exactly one. **The gap between the
two — FKL, JS, Hel, α-div: DPO-inducing but not admissible — is precisely the regime this paper is
about.** Position the contribution as the per-state/sequential analogue of their bandit-level
result, not as the first tractability characterization.

**χ² is a confound.** It is not DPO-inducing, so per Pipano its optimization is intractable for a
reason unrelated to inner-term variance. If χ² fails in Stage A/B, that failure is not evidence for
the admissibility thesis. Either drop it or report it separately with this caveat stated.

## 1. Correspondence

| tabular | LLM |
|---|---|
| state (l, s) | prompt x + generated prefix y_<t |
| action a ∈ {0,1,2} | next token y_t ∈ V |
| π_θ(·\|s) | LM next-token softmax |
| π_ref | SFT reference model (a real per-state distribution, not uniform) |
| depth = 4 (fixed) | response length T (**variable** — see §3) |
| trajectory | full response y |
| BT oracle on discounted returns | existing preference labels |

**Free correctness anchor.** For `kl` (f = t ln t), f'(u) = ln u + 1, so the chosen-action term is

    α[∇Ω]_{y_t} = α(log π_θ(y_t|s_t) − log π_ref(y_t|s_t)) + α

Summing over t gives `α·log(π_θ(y)/π_ref(y)) + αT` — the standard DPO implicit reward with β = α.
So the framework strictly generalizes DPO, and the `kl` arm must reproduce a stock DPO
implementation up to the αT term. Any divergence from that is a porting bug. Use this as the
first test.

## 2. The estimator question changes at LLM scale

`C_Ω(π) = Σ_a π_a Φ(u_a)` is a closed-form sum over the vocabulary. Logits `[B,T,V]` and reference
logprobs are already materialized by any DPO trainer, so the **exact** value costs one extra
reduction over a tensor already in memory. No MC is needed.

Consequence: `n_mc` is not the natural LLM axis. The natural axis is the `off` regime — use the
single logged next token y_{t+1}, no resampling — which is precisely what a real preference
dataset gives you. Category 1 is the *default* at LLM scale, not an artificial extreme.

So the comparison is **exact vs. single-logged-sample**, per divergence:
- admissible: single sample = exact, variance 0 by arithmetic (Φ ≡ 1)
- others: single sample is a genuinely noisy estimate of the exact sum, with 1/u tails

`n_mc > 1` (resampling from π_θ = the `off_on` / Dyna regime) stays as a secondary sweep.

**The literature is silently inconsistent here, which is an opening.** TDPO (Zeng et al. 2024)
computes its per-state term *exactly* as a vocabulary sum from the logits. TBPO / TokenRatio
(arXiv:2605.12288) estimates the same class of term by **single-sample Monte Carlo** (Schulman's K3
estimator, §4.4.1) — and states the non-cancellation directly, "state-only baseline terms that
would cancel in a same-state comparison no longer cancel," because the prefixes differ. Neither
compares against the other. So estimator variance is a *choice* nobody has audited, not an
inherent necessity — a more defensible framing than claiming the term is intrinsically
estimation-heavy. Cite TBPO's body, not its abstract: the abstract claims TBPO-A "removes the
baseline," which §4.4.1 contradicts.

**Counterargument to engage.** Xiao et al., "Preference Collapse and Matching Regularization"
(arXiv:2405.16455, accepted to JASA), Appendix A.5 considers f-divergence alternatives and
concludes they do not eliminate the algorithmic bias. This is the strongest published objection to
arguing for non-reverse-KL divergences and should be addressed head-on.

## 3. Length: no constant bias, but length-dependent variance

**Corrected 2026-07-19.** An earlier draft of this section claimed a residual α(T_w − T_l) survives
for admissible Ω at variable length. That is wrong — it counted the inner term's constant and
missed the chosen-action term's.

For f = t ln t: f'(u) = ln u + 1, so each of the T tokens contributes +α beyond the log-ratio,
while the inner term Φ ≡ 1 contributes −α at each of the T−1 successor states. Net:

    score(τ) = α·log(π_θ(y)/π_ref(y)) + αT − α(T−1) = α·log(π_θ(y)/π_ref(y)) + α

The constant is **α regardless of T**, so it cancels in d = S_w − S_l even when T_w ≠ T_l.
Verified numerically at T = 3, 7, 20, 64 (exact to 1e-12). The admissible arm is exactly
length-neutral, and reduces pathwise to standard DPO with β = α — strengthening the §1 anchor.

Both halves of the cancellation come from the same normalization choice (f'(1) = 1 ⟺ Φ → 1), the
affine-variance issue already documented in `kl_normalization.md`. Under the standard f-divergence
normalization f(1) = f'(1) = 0 they would not cancel.

**What is actually length-dependent.** For admissible Ω the identity above is *pathwise exact* —
the inner term needs no sampling, so the score carries no estimator noise at any T. For every other
Ω the inner term is a sampled quantity, so each token injects noise and the score's variance
**accumulates with response length**. Since T_w ≠ T_l generically, that noise enters d
asymmetrically.

So the prediction is not a length *bias* but a length-dependent *variance*, unique to
non-admissible Ω. This is directly measurable in Stage A: plot Var[score] against T per divergence
— flat at zero for RKL, growing in T for the rest. **[OPEN]** whether the growth is O(T) (iid
tokens) or faster (correlated prefixes) is an empirical question worth reporting.

### 3b. How Φ makes T matter — the formula

From dpo.py `score_traj`, a trajectory score splits into a deterministic chosen-action term and a
sampled inner term:

    S(τ) = Σ_{t=0}^{T-1} α·f'(u_{a_t})  −  α Σ_{t=1}^{T-1} Φ(u_{a'_t})

Only the inner term is sampled. Writing the off-policy single-sample moments per state,
μ_Ω(s) = E_{a'~β}[Φ], v_Ω(s) = Var_{a'~β}[Φ] (β = behaviour ≈ π_ref):

    Var[S]  ≈ α²(T−1)·v̄_Ω
    Var[d]  = Var[S_w − S_l] = α²(T_w + T_l − 2)·v̄_Ω
    E[d]    = signal − α(Σ_w μ_Ω − Σ_l μ_Ω)      ← extra length-dependent bias, non-admissible only

Because Φ is not constant, the T−1 inner terms each carry noise and *add up along the trajectory*,
so both variance and bias scale with T−1. RKL: Φ ≡ 1 ⇒ v = 0 and μ ≡ 1 ⇒ the whole inner
contribution is exact — zero variance, no bias, at every T.

**Experiment (`python/fig_length_variance.py` → `python/figs/length_variance.png`).** Toy off-policy
setup: |A|=100, π_ref uniform, π = softmax(N(0,1)), single logged next token a'~π_ref, α=1. Left:
std[d] vs T (T_w=T_l) — Monte-Carlo markers land exactly on the theory line σ_pool·√(2(T−1)),
straight on log-log (√T law); RKL pinned at the floor (exact 0). Right: empirical heatmap
std[d](T_w,T_l) for FKL — iso-contours are anti-diagonals, i.e. std[d] depends only on T_w+T_l,
not on the split, exactly as Var[d]=α²(T_w+T_l−2)v̄ predicts. Measured std[d] (α=1), matching
theory to ~1%:

    (T_w,T_l)   (8,8)   (8,4)   (32,4)
    RKL          0       0       0        ← v̄=0, exact at every length
    FKL         55.9    48.0    87.4
    χ²          15.8    13.3    24.5
    JS           8.95    7.54   14.0

σ_pool[Φ] (off-policy, a'~π_ref, |A|=100, scale=1): FKL 15, χ² 4.2, Hel 2.9, α-div 5.8, JS 2.4,
RKL 0. Note: an earlier throwaway used on-policy sampling (a~π) and reported smaller numbers
(FKL 13.4, …); the off-policy values above are the headline-regime ones and match the §3 narrative.

### 3c. Length normalization — use the standard sum, do NOT divide by T

Standard DPO reward is a **sum** over tokens, r = β·Σ_t log(π_θ/π_ref) = β·log(π_θ(y)/π_ref(y));
TDPO and TRL default to the sum. This is not incidental: the β(s) potentials telescope cleanly
only for the sum, so dividing by T breaks the implicit-reward derivation. SimPO (÷|y|) and R-DPO
(−α|y|) are deliberate deviations to fight length bias, and they move the objective's fixed points.

For this experiment the sum is also the *right* choice for visibility: normalizing by 1/T turns
Var[S/T] ≈ α²·v̄/T, which would *shrink* the non-admissible variance with length and mask the
pathology (RKL stays 0 either way). So: **main comparison = standard sum (no normalization)**,
matching TDPO/TRL and prior work; treat SimPO-style 1/T as a separate ablation axis, not the
default.

### 3d. Toy pre-check (before Stage A)

`python/fig_toy_ablation.py` → `python/figs/toy_Asize_ablation.png`. Toy family (π_ref uniform,
Gaussian logits) sweeping |A| = 3 … 152k. Left panel is exact and not toy-dependent: Φ_RKL ≡ 1
while every other Φ diverges as u→0. Right panel: off-policy single-sample noise
std_{a~π_ref}[Φ(u_a)] vs |A| (median ± IQR) — RKL pinned at the floor (~2e-16) at every |A|, others
rise from |A|=3 to 152k (FKL ~7e5→9e6). Confirms the mechanism qualitatively; the *magnitude* is
toy-dependent, so Stage A measures it on the real π_ref. (Notation: use |A| throughout — the
action-set size = vocabulary — not V.)

## 4. **[OPEN]** Heavy-tail handling — the main technical risk

With Φ ~ 1/u and u ~ 1e-6 over a 128k vocab, FKL/χ²/Hel single-sample estimates will produce
values ~1e6 and numerically swamp any plot or training run. This must be handled *principledly and
documented*, or a reviewer will read it as cherry-picking. Options:

- top-k / top-p truncation of the support, with the same mask applied to every Ω
- clamp u to [u_min, u_max], reporting sensitivity to the clamp
- report robust statistics (median, IQR, quantiles) alongside mean/std
- restrict to a token subset where π_ref is well-supported

Whatever is chosen must be identical across divergences — the tabular code's fairness discipline
(inner_term.py header: nothing hardcoded for KL, Φ computed from f and f') has to carry over. The
constancy of Φ_KL must remain *arithmetic*, never a special case.

## 5. Staged plan

**Stage A — measurement only, no training.** Load an SFT model + its reference, run forward passes
over an existing preference dataset, and measure the distribution of `Φ(u_{y_t})` per token and the
gap between the single-sample estimate and the exact vocab sum, for all 7 divergences.

- RKL: exactly 0 variance (up to float error) — the §4.2 signature at LLM scale
- others: positive variance, growing with off-policyness
- costs ~1 GPU-hour, needs no training, and is already a paper figure
- validates the whole premise before any training budget is spent

**Stage B — training.** Actual DPO runs per divergence, off-policy, multi-seed. Only after Stage A
confirms the measurement is clean and the tail handling from §4 works.

Stage A is the right first Killarney job: cheap, decisive, and it de-risks Stage B entirely.

## 6. Scale and resources

Mechanism claims are near-independent of model size, so seeds and off-policyness matter far more
than parameters. On Killarney, 1× L40S (48 GB) comfortably holds a 0.5B–1.5B model plus its frozen
reference. H100 is scarce and unnecessary here. Stage A is forward-pass only, so it is cheaper
still — and batchable across divergences in a single pass, since all 7 Φ are functions of the same
`u` tensor.

## 7. Repo notes

- No torch anywhere in this repo today (`requirements.txt`: streamlit / numpy / matplotlib). The
  LLM track is new code, not a modification of `dpo.py`.
- `dpo.py` / `regularizers.py` stay the tabular reference implementation and the source of truth for
  the math; the torch port should be validated against them on a small case.
- The `kl` vs RKL naming caveat applies (code key `kl` = reverse KL, displayed `RKL` via `SHORT`).
