#!/bin/bash
# Download models + dataset ON A LOGIN NODE (compute nodes have NO internet). Populates the HF
# cache so the batch job can run with HF_HUB_OFFLINE=1. Cache lives in the project dir (persistent,
# not home-quota-bound, not /scratch-purged).
#     cd ~/projects/aip-rudner/$USER/bregman-lab/llm && bash prefetch.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="/scratch/$USER/hf_cache"        # /project is full; scratch has TBs (purged when idle)
mkdir -p "$HF_HOME"
module load StdEnv/2023 python/3.11 cuda/12.2
source "$REPO/venv_gpu/bin/activate"

# Models only (no dataset here — that's a JSONL made off-cluster by make_uf_jsonl.py, see README).
# Stage A pair (1.7B). Add the 4B pair too so the scale check is ready.
for m in Qwen/Qwen3-1.7B-Base Qwen/Qwen3-1.7B Qwen/Qwen3-4B-Base Qwen/Qwen3-4B; do
  echo "== $m"; hf download "$m"          # `hf` replaced `huggingface-cli`; full repo (Qwen3 = clean safetensors)
done
echo "prefetch done -> HF_HOME=$HF_HOME"
