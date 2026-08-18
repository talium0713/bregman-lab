# Off-Policy *Permissibility* of KL in DPO — LLM track (minimal distribution)

This branch (`onboarding`) is a **minimal, distribution-focused slice** of the project: only the
**LLM-scale track** (`llm/`) — the part actively under study — plus the paper. The full project
(interactive theory viz `src/`, tabular NumPy proof `python/`, and all exploratory scripts) lives on
the **`main`** branch.

**Start here → [`ONBOARDING.md`](ONBOARDING.md)** (thesis, key results, pipeline, gotchas, fast-start).

> Terminology: the keyword is **permissibility** (formerly "admissibility"). The paper file
> `Off_policy_admissibility.pdf` and some code identifiers keep the legacy name.

## Thesis (one line)
Among all *f*-divergences used as the DPO regularizer, **reverse-KL (= standard DPO) is uniquely
permissible off-policy**: only its update improves generation quality; other divergences look fine on
preference *accuracy* but degrade generation, because their inner term is noisy / heavy-tailed.

## What's here
```
ONBOARDING.md                 project map + key results + how to reproduce
Off_policy_admissibility.pdf  the paper (theory; §1–4 = the mechanism)
llm/                          the LLM track:
  stage_b_train.py              f-DPO trainer (all divergences)
  gen_bench.py                  generation (RePO-match: rep_penalty + stops + system + --think)
  pairwise_h2h.py               direct head-to-head judge (OpenAI or any --base-url local judge)
  run_arena_v01_judge.sh        Arena-Hard v0.1 win rate vs gpt-4-0314
  fig_permissibility_{wr,h2h}.py  the two paper figures
  jobs/                         SLURM (train / gen / judge) for Killarney
  results/                      the two figures + head-to-head JSONs
  README.md                     Stage A/B design notes
```

## Headline result
With a **fixed generation harness** (an earlier bug pinned everything at a false floor), base→*f*-DPO
on Qwen3-1.7B: **only RKL beats the untrained base** (Arena-Hard v0.1 WR 14.8 vs 11.3); FKL/χ² fall
*below* it. RKL and FKL have **identical** held-out accuracy (0.706) yet RKL wins **68.3%**
head-to-head. See `llm/results/stageB_divergence_{wr,h2h}.png` and `ONBOARDING.md`.

## Reproduce (Killarney)
First-time setup — clone into **your own** `/scratch/$USER/` (it's private per-user, not a shared
path) and pull the shared SFT models + UltraFeedback data from the group backup. Full steps in
[`ONBOARDING.md`](ONBOARDING.md) → *Environments & storage*. Then, from `llm/`:
```bash
sbatch --exclude=kn010,kn001,kn035 jobs/stage_b_base_fdpo.slrm   # train base->f-DPO (6 divergences)
sbatch --exclude=kn010,kn001,kn035 jobs/gen_base_fdpo_arena.slrm # generate (RePO-match)
JUDGE=gpt-4.1-2025-04-14 bash run_arena_v01_judge.sh             # judge (login node; needs openai_key.txt)
python pairwise_h2h.py --a kl_bdpo --b rkl_bdpo --out logs/h2h.json
python fig_permissibility_wr.py && python fig_permissibility_h2h.py --dir results/bench/arena_v01/h2h
```
Data prep runs off-cluster (the cluster has no working `datasets`); see `llm/README.md`.
