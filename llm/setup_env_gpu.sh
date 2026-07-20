#!/bin/bash
# One-time GPU venv for the LLM DPO track on Killarney. Run ON A LOGIN NODE (needs internet
# for the HF stack). torch+numpy come from the hardware-tuned wheelhouse; transformers/datasets/
# accelerate/trl/peft from PyPI (need recent versions — Qwen3 requires transformers>=4.51).
#     cd ~/projects/aip-rudner/$USER/bregman-lab/llm && bash setup_env_gpu.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# arrow module provides pyarrow — Alliance's wheelhouse ships a dummy pyarrow that refuses to build,
# so `datasets` (needs pyarrow) fails without this. MUST be loaded before activating the venv.
module load python/3.12 cuda/12.2 gcc arrow

virtualenv --no-download "$REPO/venv_gpu"
source "$REPO/venv_gpu/bin/activate"
pip install --no-index --upgrade pip

# hardware-tuned wheels (numpy is NOT pulled in by torch — install explicitly)
pip install --no-index torch numpy

python -c "import pyarrow; print('pyarrow', pyarrow.__version__, '(from arrow module)')"
# HF stack from PyPI (login node has internet); transformers new enough for Qwen3.
# NOTE: if pip still tries to build pyarrow, the arrow module's pyarrow is older than latest
# datasets wants — pin datasets to match (e.g. pyarrow<21 -> `datasets<4`).
pip install "transformers>=4.51" datasets accelerate "trl>=0.12" peft huggingface_hub

pip freeze > "$REPO/llm/requirements.lock.txt"
echo "gpu venv ready at $REPO/venv_gpu"
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| transformers", transformers.__version__)
PY
