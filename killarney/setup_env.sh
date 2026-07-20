#!/bin/bash
# One-time CPU venv for the tabular reproduction on Killarney.
# Run ON A LOGIN NODE (may fall back to PyPI, which needs internet):
#     cd ~/projects/aip-rudner/$USER/bregman-lab/killarney && bash setup_env.sh
# The tabular code is pure NumPy — NO torch, NO GPU. venv lives at repo-root/venv.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (parent of killarney/)
cd "$REPO"

module load python/3.12

virtualenv --no-download "$REPO/venv"
source "$REPO/venv/bin/activate"
pip install --no-index --upgrade pip

# Only these two are needed (streamlit in requirements.txt is for the local Streamlit app; skip it).
pip install --no-index numpy matplotlib

pip freeze > "$REPO/killarney/requirements.lock.txt"
echo "venv ready at $REPO/venv"
echo "sanity: $(python -c 'import numpy, matplotlib; print("numpy", numpy.__version__, "| matplotlib", matplotlib.__version__)')"
