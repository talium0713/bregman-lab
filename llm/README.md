# LLM DPO track — Killarney

LLM-scale side of the permissibility project (parallel to `../killarney/`, which is the CPU tabular
reproduction). Goal is unchanged: reproduce the §1–3 mechanism — the permissible divergence (RKL)
has a **noise-free off-policy inner term**, every other divergence's inner term is noisy, and the
gap widens with vocabulary size / off-policyness.

**Decisions (from the Notion "🚀 LLM-scale DPO" toggle):**
- Model: **Qwen3-1.7B** primary; **Qwen3-4B** scale check (full-param via **2× L40S + FSDP**, at Stage B).
- π_ref = **base** (`Qwen/Qwen3-1.7B-Base`); Stage A uses π_θ = instruct (`Qwen/Qwen3-1.7B`).
- Divergences: all **7** (RKL, FKL, JS, Hel, χ², α-div, Euc) — same set as tabular `REGKEYS`.
- Benchmarks: UltraFeedback (primary) + Math (robustness). Training = token-level, standard sum, off-policy.

## Stage A — measurement (this folder, ready now)

No training. Score real preference responses with two frozen models and, at every response token,
compute `u = π_θ/π_ref` and the single-logged-token inner term `Φ(u)` for all 7 divergences.
Prediction: RKL std ≈ 0; others spread, growing as `u→0`. Forward-only ⇒ **1× L40S is enough even
for 4B**. It's cheap, it's already a paper figure, and it de-risks Stage B.

```
llm/
  setup_env_gpu.sh     one-time GPU venv (torch wheelhouse + transformers), login node
  prefetch.sh          download Qwen3 models into HF cache (/scratch), login node
  make_uf_jsonl.py     dump UltraFeedback -> JSONL — run on your MAC (needs datasets), then scp
  divergences.py       torch Φ(u) for the 7 divergences (port of ../python/regularizers.py)
  stage_a_measure.py   the measurement, reads data/*.jsonl → results/<name>.{json,png}
  jobs/stage_a.slrm    1× L40S, HF_HUB_OFFLINE=1
  data/  logs/  results/
```

**Work from `/scratch` on Killarney** (`/project` is a full 5 TB group share; `/scratch/$USER` is your
own 2 TB). **No `datasets`/pyarrow on the cluster** — Alliance ships a build-failing dummy pyarrow, so
the dataset is prepared off-cluster as JSONL and read with plain `json`.

```bash
# on your MAC (normal pip, no Alliance dummy):
pip install datasets
python make_uf_jsonl.py --split test_prefs --max 1000 --out data/uf_test_prefs.jsonl
scp data/uf_test_prefs.jsonl talium@killarney.alliancecan.ca:/scratch/talium/bregman-lab/llm/data/

# on Killarney (from /scratch/$USER/bregman-lab/llm):
bash setup_env_gpu.sh     # 1) venv (login node) — torch + transformers only
bash prefetch.sh          # 2) download Qwen3 models -> /scratch HF cache (login node)
sbatch jobs/stage_a.slrm  # 3) measure (1× L40S)
tail -f logs/stageA-*.out
```

Output `results/stageA_qwen3_1p7b.{json,png}`: per-divergence std of the single-token Φ, the
single-vs-exact gap (`--exact`), and the `log_u` distribution (how off-policy the data is). This is
the LLM-scale analogue of `../python/figs/toy_Asize_ablation.png` on a *real* π_ref instead of a toy.

## Stage B — training (scaffolded)

Token-level f-DPO, own trainer (no TRL — the full-vocab inner term is a few lines, and TRL fights
the Alliance no-pyarrow constraint). Score `S(τ) = β Σ_t[f'(u_t) − C_Ω(s_t)]`, standard sum (§3c),
off-policy. **Design = exact vs single-sample inner term × 7 divergences** (`jobs/stage_b.slrm`,
13-way array). RKL is the correctness anchor: both arms collapse to standard DPO, `--selftest`
verifies to 2e-16.

```
  make_pairs_jsonl.py   dump UltraFeedback PAIRS (chosen+rejected) -> JSONL — run on your MAC, then scp
  stage_b_train.py      the trainer; `--selftest` validates the math with no GPU/models
  jobs/stage_b.slrm     1× L40S array (13 runs: 6 f-divs × {sample,exact} + euc exact)
```

```bash
# on your MAC:
python make_pairs_jsonl.py --split train_prefs --max 20000 --out data/uf_pairs_train.jsonl
python make_pairs_jsonl.py --split test_prefs  --max 1000  --out data/uf_pairs_test.jsonl
scp data/uf_pairs_*.jsonl talium@killarney.alliancecan.ca:/scratch/talium/bregman-lab/llm/data/

# on Killarney (validate math first — instant, CPU):
python stage_b_train.py --selftest
# smoke one arm on GPU (30 steps), then the full sweep:
python stage_b_train.py --div kl --inner sample --steps 30 --out results/smoke
sbatch jobs/stage_b.slrm
```

`--inner exact` = full-vocab C_Ω (fp32, π_θ-weighted → tail-tamed); `--inner sample` = single logged
token (realistic off-policy, heavy-tailed for non-permissible). 1.7B pilot on 1× L40S first; 4B via
**2× L40S + FSDP** is the next step (not yet wired — `stage_b_train.py` is currently single-GPU).
Prediction: RKL invariant across arms; non-permissible degrade under `sample`, recover under `exact`.
Length-bias measurement needs generation (separate eval, TODO). χ²/euc reported separately.

## Notes / open items

- Qwen3 needs `transformers>=4.51`; `enable_thinking=False` in the chat template suppresses `<think>`.
- HF cache is pinned to `~/projects/aip-rudner/$USER/hf_cache` (persistent; avoids home quota).
- α-div uses `a=0.5` (tabular default `DEFAULT_ADIV_A`); at a=0.5 its Φ is ∝ Hellinger's — expected.
- Euc/χ² are the two non-standard members (Euc not an f-div; χ² not DPO-inducing) — report with the
  same caveats as tabular. Kept only for parity with the tabular 7-set.
