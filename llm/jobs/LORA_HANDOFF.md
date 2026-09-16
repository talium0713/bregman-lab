# 8B LoRA DPO on one L40S — run instructions

**What you are running.** Two DPO runs of Qwen3-8B-Base on UltraFeedback, identical except for one
flag: `--norm amari` vs `--norm canon`. They test whether the canonical form of an f-divergence beats
the Amari form at 8B scale. You need **one 48 GB GPU**, not four.

**Why it fits on one GPU.** Standard DPO holds a second frozen copy of the model as the reference,
and under FSDP that copy is replicated on every rank rather than sharded — that is what forces 8B
full fine-tuning onto 4× H100. With LoRA the base weights are never written, so *disabling the
adapter is the reference*, and the second copy disappears. Measured peak: **27.8 GB**.

**Time.** ~90 s/step on L40S, 1,910 steps (2 epochs) ≈ **48 hours** per arm. The two arms are an
array job and run independently, so wall-clock is ~48 h if both get scheduled together.

---

## 1. Get the code

```bash
git clone https://github.com/talium0713/bregman-lab.git
cd bregman-lab
git checkout lora-l40s
```

Everything below is run from the `llm/` directory unless stated otherwise.

## 2. Environment

On a **login node** (compute nodes have no internet):

```bash
module load StdEnv/2023 python/3.11 cuda/12.2
python -m venv venv_gpu            # at the REPO ROOT, i.e. bregman-lab/venv_gpu
source venv_gpu/bin/activate
pip install --no-index torch numpy
pip install transformers peft      # peft is NOT in the Alliance wheelhouse, needs PyPI
```

`flash-attn` is optional — the code falls back automatically if it is not importable, just slower.

Verify:

```bash
python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"
```

Known-good: torch 2.x, transformers 5.14.1, **peft 0.20.0**.

## 3. Data

The training data is **not in git** (78 MB). Regenerate it on a machine with internet and `datasets`
(your laptop is fine — this deliberately keeps `datasets`/pyarrow off the cluster):

```bash
pip install datasets
python make_pairs_jsonl.py --split train_prefs --max 100000 --out data/uf_pairs_train_full.jsonl
python make_pairs_jsonl.py --split test_prefs  --max 1000   --out data/uf_pairs_test.jsonl
```

Expected: **61,135** lines in `uf_pairs_train_full.jsonl`, 1,000 in `uf_pairs_test.jsonl`.
The step budget assumes exactly this — if your line count differs, the 2-epoch step count changes.

Copy them to the cluster, into `llm/data/` (or symlink `llm/data` at a scratch directory — that is
what the original setup does, since these files should not live on a quota-limited filesystem).

## 4. Model cache

Compute nodes run with `HF_HUB_OFFLINE=1`, so the model must be downloaded **first, on a login node**:

```bash
export HF_HOME=/scratch/$USER/hf_cache          # ~16 GB for this model
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B-Base')"
```

If that fails, see *Gotchas* below — every failure mode there is one we actually hit.

## 5. Submit

Edit the two SLURM headers in `jobs/stage_b_lora_8b_l40s.slrm` to match your allocation:

```bash
#SBATCH --account=aip-rudner       # <- your account
#SBATCH --gres=gpu:l40s:1          # <- your 48G+ GPU type
```

Then, **from `llm/`**:

```bash
sbatch jobs/stage_b_lora_8b_l40s.slrm
```

That submits an array of 2: index 0 = `amari`, index 1 = `canon`.

Outputs go to `OUT_DIR` (default `/scratch/$USER/bregman-lab/llm/results/bench`), overridable:

```bash
OUT_DIR=/my/scratch/path sbatch jobs/stage_b_lora_8b_l40s.slrm
```

## 6. Checking on it

```bash
squeue -u $USER
tail -f logs/lora8bL-<jobid>_<0|1>.out
```

A healthy log line looks like:

```
  step  100  loss 0.6612  train_acc 0.578  eval_acc 0.601 margin +0.043  |g| 2.11  9000s  mem 27.8G
```

What to watch:

| Signal | Healthy | Trouble |
|---|---|---|
| `mem` | ~28 G, flat | climbing → OOM coming |
| `loss` | drifting down from ~0.69 | **rising** — see below |
| `|g|` | order 1–10 | order 100+ → the run is collapsing |
| `sec` | ~90 × step | much higher → will not finish in 68 h |

A **rising loss on the `amari` arm is a real result, not a bug.** At 8B the Amari arm degraded while
canonical did not, and that instability is precisely what these runs are measuring. Do not "fix" it.
Report it.

## 7. Results to send back

Per arm, the two things that matter:

- `.../stageB_lora8b_kl_{amari,canon}_bdpo_token.json` — the full loss/eval curve. **Small, send this first.**
- `.../stageB_lora8b_kl_{amari,canon}_bdpo_token_policy/` — the merged model, 16 GB each.

The `_policy` directory is an ordinary HuggingFace causal-LM directory (the adapter is merged in at
save time), so it loads with plain `AutoModelForCausalLM.from_pretrained`. Generation + GPT-4.1 judging
happen on our side — you do **not** need an OpenAI key. If moving 16 GB is awkward, send the
`_adapter/` directory instead (**1.4 GB**) and we will merge it with `merge_adapter.py`.

## 8. If a job dies

`--save-adapter-every 100` writes `..._adapter/` (1.4 GB) every ~2.4 h, so a crash costs one interval, not the
run. Recover with:

```bash
python merge_adapter.py \
  --adapter results/bench/stageB_lora8b_kl_canon_bdpo_token_adapter \
  --base    Qwen/Qwen3-8B-Base \
  --out     results/bench/stageB_lora8b_kl_canon_bdpo_token_policy
```

Runs on CPU, needs ~16 GB RAM, no GPU. The result is indistinguishable from a normally-finished run.

---

## Gotchas

Every one of these cost us real time.

**Storage.** A saved 8B policy is 16 GB and there are two of them, plus adapters. Write to scratch,
never to a quota-limited project filesystem. Ours sits at 99% of 5 TiB and a failed write at the
*save* step kills a 48-hour job at the very end.

**Walltime tiers.** On Killarney the requested walltime selects a node pool. Asking for more time than
you need shrinks the pool you can land on; asking for too little kills the run at the save step, since
the merged policy is written only after the final training step. 68 h is deliberate: it covers up to
~128 s/step and still sits in the 3-day tier.

**`HF_HOME` vs the token.** Setting `HF_HOME` moves where `huggingface_hub` looks for its token, so a
token saved by `huggingface-cli login` at `~/.cache/huggingface/token` becomes invisible. Use
`export HF_TOKEN=hf_...` instead of relying on the token file.

**Xet backend.** If downloads fail with `Unable to parse string as hex hash value`, set
`export HF_HUB_DISABLE_XET=1`.

**Saving a token over SSH.** `read -rs` in a non-TTY SSH session silently writes an empty file. Check
with `wc -c` before trusting it.

**Chat templates on `-Base` models.** Base checkpoints often ship without a chat template. The trainer
installs a fallback, and that fallback **must** end the assistant turn with EOS — an earlier version
did not, and 41% of generations ran to the token cap emitting turn markers, silently invalidating a
whole result. If you point this at a different base model, check that generations terminate.

## Questions

Anything unexpected in the log — especially a memory climb, or `sec` per step far off 90 — is worth a
message before letting it burn 48 hours.
