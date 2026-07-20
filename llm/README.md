# LLM DPO track — Killarney

LLM-scale side of the admissibility project (parallel to `../killarney/`, which is the CPU tabular
reproduction). Goal is unchanged: reproduce the §1–3 mechanism — the admissible divergence (RKL)
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
  setup_env_gpu.sh     one-time GPU venv (torch wheelhouse + HF stack), login node
  prefetch.sh          download Qwen3 models + UltraFeedback into HF cache, login node
  divergences.py       torch Φ(u) for the 7 divergences (port of ../python/regularizers.py)
  stage_a_measure.py   the measurement → results/<name>.{json,png}
  jobs/stage_a.slrm    1× L40S, HF_HUB_OFFLINE=1
  logs/  results/
```

Run:
```bash
cd ~/projects/aip-rudner/$USER/bregman-lab/llm
bash setup_env_gpu.sh     # 1) venv (login node)
bash prefetch.sh          # 2) download models+data (login node; compute nodes have no internet)
sbatch jobs/stage_a.slrm  # 3) measure (1× L40S)
tail -f logs/stageA-*.out
```

Output `results/stageA_qwen3_1p7b.{json,png}`: per-divergence std of the single-token Φ, the
single-vs-exact gap (`--exact`), and the `log_u` distribution (how off-policy the data is). This is
the LLM-scale analogue of `../python/figs/toy_Asize_ablation.png` on a *real* π_ref instead of a toy.

## Stage B — training (later)

Token-level DPO per §5 (TRL fork + TDPO math), standard sum, off-policy, seed 3–5, all 7 divergences.
1.7B = full DPO on 1× L40S; 4B = full DPO on **2× L40S + FSDP/ZeRO-3** (accelerate/torchrun). Needs
the `selective_log_softmax`/liger bypass + `precompute_ref_log_probs=False` for the full-vocab inner
term. Not scaffolded yet — do Stage A first and read its numbers (esp. the tail handling) before
committing training compute.

## Notes / open items

- Qwen3 needs `transformers>=4.51`; `enable_thinking=False` in the chat template suppresses `<think>`.
- HF cache is pinned to `~/projects/aip-rudner/$USER/hf_cache` (persistent; avoids home quota).
- α-div uses `a=0.5` (tabular default `DEFAULT_ADIV_A`); at a=0.5 its Φ is ∝ Hellinger's — expected.
- Euc/χ² are the two non-standard members (Euc not an f-div; χ² not DPO-inducing) — report with the
  same caveats as tabular. Kept only for parity with the tabular 7-set.
