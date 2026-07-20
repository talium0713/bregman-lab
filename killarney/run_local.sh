#!/bin/bash
# Local dry-run of the whole reproduction WITHOUT SLURM — verify the pipeline before
# submitting to Killarney. Runs each experiments.list command sequentially, then figs.
#     bash killarney/run_local.sh            # full run (slow: ~13 data runs)
#     bash killarney/run_local.sh --figs-only  # skip data, just regenerate figures
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO/python"
export MPLBACKEND=Agg
PY="${PYTHON:-python3}"
JOBS="${JOBS:-$(( $( (nproc 2>/dev/null || sysctl -n hw.ncpu) ) - 2 ))}"

if [[ "${1:-}" != "--figs-only" ]]; then
  while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
    echo ">>> $line --jobs $JOBS"
    eval "${line/python/$PY} --jobs $JOBS"
  done < "$REPO/killarney/experiments.list"
fi

echo ">>> figures"
$PY fig_tabular.py
$PY fig_horizon.py
for p in 60 70 80; do
  $PY fig_adiv.py "data/tabular/run_adiv_p${p}/results.json"
  $PY fig_adiv.py "data/tabular/run_adiv_p${p}_kln/results.json"
done
$PY fig_adiv_compare.py
$PY fig_adiv_invariant.py
$PY fig_mechanism.py
echo "done — figures in python/figs/"
