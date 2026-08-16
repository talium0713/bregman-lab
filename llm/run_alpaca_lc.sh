#!/bin/bash
# AlpacaEval 2.0 — standard Length-Controlled (LC) + raw win rate (WR). Login node (internet +
# OPENAI_API_KEY). Judges each --model_outputs vs the BUNDLED gpt-4-turbo reference with the
# weighted_alpaca_eval_gpt4_turbo annotator (this is the leaderboard setup that gives LC + WR).
# Unlike run_alpaca_judge.sh (which compares vs OUR SFT baseline), this uses the standard reference so
# the numbers are comparable to published AlpacaEval2 tables.
#   bash run_alpaca_lc.sh results/bench/base_alpaca.json [more .json ...]
set -uo pipefail
cd "$(dirname "$0")"
[ -z "${OPENAI_API_KEY:-}" ] && [ -f openai_key.txt ] && export OPENAI_API_KEY="$(tr -d '[:space:]' < openai_key.txt)"
[ -n "${OPENAI_API_KEY:-}" ] || { echo "no OPENAI_API_KEY — export it or put it in $(pwd)/openai_key.txt"; exit 1; }
module load StdEnv/2023 python/3.11 2>/dev/null || true
ENV="${ALPACA_ENV:-$HOME/alpaca_judge_env}"
[ -d "$ENV" ] || python -m venv "$ENV"
source "$ENV/bin/activate"
python -c "import alpaca_eval" 2>/dev/null || pip install -q alpaca-eval --index-url https://pypi.org/simple/

for f in "$@"; do
  [ -f "$f" ] || { echo "skip (missing): $f"; continue; }
  name=$(basename "$f" .json)
  echo "=== AlpacaEval2 (LC + WR) vs gpt-4-turbo reference: $name ==="
  alpaca_eval --model_outputs "$f" \
              --annotators_config weighted_alpaca_eval_gpt4_turbo \
              --output_path "results/bench/ae_lc_${name}" 2>&1 | tail -20 || echo "  (alpaca_eval failed for $name)"
  # echo the LC + WR from the leaderboard csv
  python3 - "results/bench/ae_lc_${name}" <<'PY'
import csv, glob, sys
d = sys.argv[1]
fs = glob.glob(f"{d}/**/leaderboard.csv", recursive=True) + glob.glob(f"{d}/*leaderboard*.csv")
if fs:
    r = list(csv.DictReader(open(fs[-1])))[-1]
    lc = r.get("length_controlled_winrate") or r.get("lc_win_rate") or "-"
    wr = r.get("win_rate") or r.get("winrate") or "-"
    print(f"  -> LC = {lc}   WR = {wr}")
PY
done
