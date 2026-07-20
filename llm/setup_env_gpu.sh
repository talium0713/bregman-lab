#!/bin/bash
# One-time GPU venv for the LLM DPO track on Killarney. Run ON A LOGIN NODE (needs internet
# for the HF stack). torch+numpy come from the hardware-tuned wheelhouse; transformers/datasets/
# accelerate/trl/peft from PyPI (need recent versions — Qwen3 requires transformers>=4.51).
#     cd ~/projects/aip-rudner/$USER/bregman-lab/llm && bash setup_env_gpu.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module load StdEnv/2023 python/3.11 cuda/12.2

virtualenv --no-download "$REPO/venv_gpu"
source "$REPO/venv_gpu/bin/activate"
pip install --no-index --upgrade pip

# hardware-tuned wheels (numpy is NOT pulled in by torch — install explicitly)
pip install --no-index torch numpy

# Stage A needs ONLY torch + transformers (+huggingface_hub for downloads). We deliberately do NOT
# install `datasets`/`trl` here — they pull pyarrow, which Alliance ships as a build-failing dummy
# wheel. The dataset is prepared off-cluster as JSONL (make_uf_jsonl.py) and read with plain json.
# Stage B (training) will need trl/datasets; solve the arrow-module/pyarrow bridge then.
pip install "transformers>=4.51" accelerate huggingface_hub

pip freeze > "$REPO/llm/requirements.lock.txt"
echo "gpu venv ready at $REPO/venv_gpu"
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| transformers", transformers.__version__)
PY
