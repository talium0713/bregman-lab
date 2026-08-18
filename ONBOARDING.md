# Onboarding — Unique Off-Policy *Permissibility* of KL in DPO

> **Terminology note.** The project keyword was renamed **admissibility → permissibility** (same
> concept). The paper draft `Off_policy_admissibility.pdf` and the older `README.md` / `llm/README.md`
> still say "admissibility" — read it as "permissibility". New docs/figures use permissibility.

> **This `onboarding` branch is a minimal distribution** — it ships only the **`llm/`** track (the
> active work) + the paper. The "Theory viz" (`src/`) and "Tabular proof" (`python/`) tracks described
> below live in the **full repo on `main`**; clone `main` if you need them.

## The one-sentence thesis
Among all *f*-divergences used as the DPO regularizer, **reverse-KL (RKL = standard DPO) is uniquely
"permissible" off-policy**: its per-token inner term is noise-free, so its gradient is a valid
off-policy ascent direction. Every other divergence's inner term is noisy / heavy-tailed, so
off-policy it mis-updates the policy — visible not in preference *accuracy* but in generation
*quality*.

## Three tracks (each has its own README)
| Track | Where | What it is | Read |
|---|---|---|---|
| **Theory viz** | `src/`, `index.html` | Interactive JS suite: dual manifold, theorem-live MDP, tabular DPO under 7 regularizers | `README.md` |
| **Tabular proof** | `python/` | Clean NumPy reimplementation of the tabular experiment (§4). Verifiable against the paper; the controlled proof of the mechanism | `python/README.md` |
| **LLM scale** | `llm/` | Stage A (measurement) + Stage B (real *f*-DPO on Qwen3-1.7B) + generation benchmarks | `llm/README.md` (Stage A/B) **+ this doc** (recent benchmark work) |

The 7 divergences are consistent across tracks: **RKL (`kl`) · FKL (`rkl`) · α-div (`adiv`) · JS
(`js`) · Hellinger (`hel`) · χ² (`chi2`) · Euclidean (`euc`)**. (Trainer keys in parentheses — note
`kl`=RKL and `rkl`=FKL, a historical naming quirk.)

---

## Key results so far

**Tabular (`python/`)** — RKL's inner term `C_Ω(s')` is constant (needs no estimator); all others
require a Monte-Carlo estimate whose variance grows as `u→0` / off-policy. Final policy-gap bars
show KL's permissibility signature. `python run_part3.py` regenerates every figure + `verify.py`
(7 numerical checks, all must PASS).

**Stage B held-out preference accuracy (Qwen3-1.7B, UltraFeedback test, n=3 seeds)** — RKL top &
tightest at both granularities (token 0.704±0.005, newline 0.709±0.005); the only divergence flat
token→newline. This is the robust ranking signal.

**Stage B generation benchmark (Arena-Hard v0.1, the headline recent work) —**

1. **⚠️ A generation-harness bug was found and fixed (2026-08).** Our HF `model.generate()` lacked
   controls that vLLM applies by default (`repetition_penalty`, `<|im_end|>`/string stop tokens,
   system prompt), so raw base answers ran on to 2048 tokens / degenerate loops and scored **2.6**
   WR. With the fix, raw Qwen3-1.7B-Base scores **11.2**, reproducing RePO's reported **10.4**.
   **Every benchmark number produced before this fix is confounded** — regenerate with the
   RePO-match recipe (below). Training is unaffected (teacher-forced); only *generation* was wrong.

2. **base→*f*-DPO permissibility sweep** (all 6 divergences trained directly on the base,
   `jobs/stage_b_base_fdpo.slrm`, `--init-noise 0.01`): **only RKL beats the untrained base**
   (14.8 vs 11.3); JS/Hel/α ≈ base; **χ² (7.4) and FKL (6.8) fall *below* base** (f-DPO actively
   hurts). Figure: `llm/results/stageB_divergence_wr.{png,pdf}`.

3. **The clincher — accuracy hides it, generation reveals it.** RKL and FKL have **identical**
   held-out accuracy (0.706 = 0.706), yet in a **direct head-to-head** (`pairwise_h2h.py`, 2 games /
   prompt, position-swapped, bootstrap CI) the judge prefers **RKL over FKL 68.3%** of the time —
   and RKL beats *every* opponent (χ² 65.9 · α 62.2 · JS 60.9 · Hel 60.1 · base 59.7; all 95%-CI
   lower bounds >56%). Figure: `llm/results/stageB_divergence_h2h.{png,pdf}`.

Full write-ups live in Notion: **"Recipe for Experiments"** (the recipe + the ⭐ RESULT toggle),
**"Summary of Experimental Results"**, **"API experiment"** (OpenAI budget).

---

## Current LLM benchmark pipeline (how to reproduce / extend)

Everything runs on **Killarney** from **your own** working copy at `/scratch/$USER/bregman-lab/llm`
(`$USER` = *your* login — `/scratch` is **private per-user**, so this is a directory you create, not a
shared one; see *Environments & storage* below for the one-time setup + where the shared models/data
are). Three stages:

**1. Train a policy** (GPU job). Settled recipe = π_init=π_ref (+init-noise) · β 0.1 · lr 5e-6 linear ·
2 ep · batch 64 · clamp 15 · single-sample · token-level.
```bash
sbatch --exclude=kn010,kn001,kn035 jobs/stage_b_base_fdpo.slrm   # base->f-DPO, 6 divergences (array 0-5)
# saves results/bench/stageB_{div}_bdpo_token_bench1p7b_policy
```

**2. Generate answers — MUST use the RePO-match recipe** (this is the bug fix; `gen_bench.py` gained
`--rep-penalty` / `--system` / `--think {off,default,on}` + built-in stop_strings):
```bash
sbatch --exclude=kn010,kn001,kn035 jobs/gen_base_fdpo_arena.slrm
# = gen_bench.py --think default --system "You are a helpful assistant." --rep-penalty 1.05 --max-new 1024
#   -> results/bench/arena_v01/{div}_bdpo.jsonl
```

**3. Judge** (login node — needs internet + `openai_key.txt`; NOT a compute node):
```bash
JUDGE=gpt-4.1-2025-04-14 bash run_arena_v01_judge.sh          # WR vs gpt-4-0314 (Bradley-Terry)
python pairwise_h2h.py --a kl_bdpo --b rkl_bdpo --out logs/h2h.json   # direct head-to-head + CI
```
`pairwise_h2h.py` also takes `--base-url` / `--judge-model` to use **any OpenAI-compatible judge**
(e.g. a local model — cost $0). Figures: `python fig_permissibility_wr.py`, `python fig_permissibility_h2h.py`.

Data prep (UltraFeedback pairs, Arena prompts) is done **off-cluster** — see `llm/README.md` (the
cluster has no working `datasets`/pyarrow).

---

## Environments & storage
**Storage model — this is what trips people up.** `$USER` always means *your own* login.
- `/scratch/$USER/` — **your private** per-user space (~2 TB, auto-purged ~60 d). It is **not shared**:
  you cannot see anyone else's `/scratch` (so `/scratch/talium/…` is invisible to you, and yours to
  them). **Your working copy of the repo lives here.**
- `~/projects/aip-rudner/` — the **shared 5 TB group** project space; every `aip-rudner` member can
  read it. The shared **models + data** live here (`~/projects/aip-rudner/talium/bregman-backup`).
- `/home` — small, persistent, private.

**First-time setup (new researcher, in the `aip-rudner` group):**
```bash
# 1) your OWN working copy — on YOUR scratch ($USER = your login, NOT talium):
git clone -b onboarding https://github.com/talium0713/bregman-lab.git /scratch/$USER/bregman-lab
cd /scratch/$USER/bregman-lab
bash check_access.sh                               # verify you can read the shared backup (read-only; see below)
cd llm && bash setup_env_gpu.sh && bash prefetch.sh   # 2) GPU venv + HF base models
# 3) shared SFT models + UltraFeedback data from the group backup (Taehyun's, group-readable):
SRC=~/projects/aip-rudner/talium/bregman-backup
rsync -a "$SRC"/models/ results/                   # SFT bases (results/sft_uc_1p7b_model, …) for the sft_base baseline
rsync -a "$SRC"/data/   data/                       # uf_pairs_*.jsonl, ultrachat, …
# 4) run the pipeline above (train → gen → judge → figures). The trained policies + Arena answers are
#    NOT in the backup (too big); the pipeline regenerates them.
```
`check_access.sh` (repo root, read-only) checks your group membership + the traversal chain + a real
read of each key file, and prints exactly which item is blocked. If it fails and you *are* in
`aip-rudner`, send Taehyun its output (it dumps `namei -l` for the blocked paths) — the fix is usually
`chgrp -R aip-rudner` + `chmod -R g+rX` on the backup.

- **venvs**: GPU `venv_gpu`; judge `~/arena_judge_env` (torch + the arena repo cloned at `~/arena-hard-auto`).
- **Secrets**: OpenAI key in `llm/openai_key.txt` (gitignored, never committed). A local judge needs no key.
- **Local (Mac)**: figures + data prep (`pip install datasets openai`). No GPU needed.

## Critical gotchas (will bite you)
- **The generation-harness bug** (above) — always generate with the RePO-match recipe; a raw
  `model.generate()` with only `pad_token_id=eos` gives degenerate answers and a false "1.7B floor".
- **Killarney SSH needs MFA** — a fresh connection prompts 2FA. Reuse a ControlMaster; run cluster
  commands via a **login shell** (`ssh killarney 'bash -lc "squeue --me"'`) or `sbatch`/`squeue`
  aren't on PATH.
- **Bad ECC nodes** `kn010, kn001, kn035` — always `--exclude` them; model-load ECC crashes recur.
- **No `datasets`/pyarrow on the cluster** — prepare all JSONL on your Mac and `scp`.
- **Don't** use `tr -d '[:space:]'` inside nested `ssh 'bash -lc "..."'` to read a key — the quoting
  mangles it (false 401). Read key files with Python (`pairwise_h2h.py` does).
- **The YGPT web UI is not an API** — it uses cookie/CSRF auth. For a local judge, point at the
  underlying model server's OpenAI-compatible `/v1` endpoint, not the UI URL.

## Open items / next steps
- Cross-check the divergence head-to-head with a **local judge** (handoff bundle prepared:
  `pairwise_h2h.py --base-url ... --judge-model ...`).
- An **accuracy-vs-WR dissociation** figure (RKL≡FKL accuracy, opposite WR) for the paper.
- Extend the RePO-match benchmark to **AlpacaEval 2 / MT-Bench**, and to **newline** granularity on
  the base track (currently token only).
- Scale check at **Qwen3-4B** (RePO's own numbers are 4B) to compare absolute DPO deltas.
- Uncommitted work lives on branch **`permissibility-gen-fix`** (not yet pushed).

## Fast start for a new researcher
1. Read the paper `Off_policy_admissibility.pdf` (§1–4) — the mechanism.
2. Look at the two result figures: `llm/results/stageB_divergence_{wr,h2h}.{png,pdf}`.
3. Skim `llm/README.md` (Stage A/B design) then this doc's pipeline section for the current state.
4. Reproduce: follow the pipeline section (train → gen → judge → figures).
5. To touch the cluster: get an `aip-rudner` allocation, read the gotchas above.
6. (Optional, on `main`) the tabular proof: `cd python && QUICK=1 python run_part3.py && python verify.py`.
