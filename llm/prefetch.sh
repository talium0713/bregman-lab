#!/bin/bash
# Download models + dataset ON A LOGIN NODE (compute nodes have NO internet). Populates the HF
# cache so the batch job can run with HF_HUB_OFFLINE=1. Cache lives in the project dir (persistent,
# not home-quota-bound, not /scratch-purged).
#     cd ~/projects/aip-rudner/$USER/bregman-lab/llm && bash prefetch.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="$HOME/projects/aip-rudner/$USER/hf_cache"
mkdir -p "$HF_HOME"
module load python/3.12 cuda/12.2
source "$REPO/venv_gpu/bin/activate"

# Stage A pair (1.7B). Add the 4B pair too so the scale check is ready.
for m in Qwen/Qwen3-1.7B-Base Qwen/Qwen3-1.7B Qwen/Qwen3-4B-Base Qwen/Qwen3-4B; do
  echo "== $m"; huggingface-cli download "$m" --exclude "*.pth" "original/*"
done

python - <<'PY'
from datasets import load_dataset
for sp in ("test_prefs", "train_prefs"):
    d = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=sp)
    print(sp, len(d), "examples cached")
PY
echo "prefetch done -> HF_HOME=$HF_HOME"
