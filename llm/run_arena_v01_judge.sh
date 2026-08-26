#!/bin/bash
# Arena-Hard v0.1 judging (login node, internet + OPENAI_API_KEY). The REPORT-comparable version —
# baseline/anchor = gpt-4-0314 (weaker than v2's o3-mini), so a 1.7B model is not pinned at the floor.
# Reuses the already-cloned ~/arena-hard-auto + ~/arena_judge_env (run run_arena_judge.sh once first).
# Judge defaults to gpt-4.1 (cheap: ~$11/model); set JUDGE=gpt-4-1106-preview to match the v0.1
# leaderboard exactly (~5× cost). Judges every results/bench/arena_v01/*.jsonl.
#   bash run_arena_v01_judge.sh            # judge=gpt-4.1
#   JUDGE=gpt-4-1106-preview bash run_arena_v01_judge.sh
set -uo pipefail
cd "$(dirname "$0")"; LLM_DIR="$(pwd)"
ARENA_DIR="${ARENA_DIR:-$HOME/arena-hard-auto}"
JUDGE="${JUDGE:-gpt-4.1-2025-04-14}"   # RePO's exact Arena-Hard v0.1 judge snapshot
[ -z "${OPENAI_API_KEY:-}" ] && [ -f openai_key.txt ] && export OPENAI_API_KEY="$(tr -d '[:space:]' < openai_key.txt)"
[ -n "${OPENAI_API_KEY:-}" ] || { echo "no OPENAI_API_KEY — export it or put it in $(pwd)/openai_key.txt"; exit 1; }
ls results/bench/arena_v01/*.jsonl >/dev/null 2>&1 || { echo "no results/bench/arena_v01/*.jsonl — run gen_arena_v01.slrm first"; exit 1; }
[ -d "$ARENA_DIR" ] || { echo "$ARENA_DIR missing — run run_arena_judge.sh once first to clone the repo + data"; exit 1; }

module load StdEnv/2023 python/3.11 2>/dev/null || true
source "$HOME/arena_judge_env/bin/activate"

DST="$ARENA_DIR/data/arena-hard-v0.1/model_answer"; mkdir -p "$DST"
cp "$LLM_DIR"/results/bench/arena_v01/*.jsonl "$DST/"
MODELS=(); for f in "$LLM_DIR"/results/bench/arena_v01/*.jsonl; do MODELS+=("$(basename "$f" .jsonl)"); done
echo "=== Arena-Hard v0.1 judging: ${MODELS[*]}  (judge=$JUDGE, baseline=gpt-4-0314) ==="

cd "$ARENA_DIR"
python - "$JUDGE" "${MODELS[@]}" <<'PY'
import sys, yaml, os
judge, models = sys.argv[1], sys.argv[2:]
cfgp = "config/arena-hard-v0.1.yaml"
c = yaml.safe_load(open(cfgp)) or {}
c["judge_model"] = judge; c["model_list"] = models
yaml.safe_dump(c, open(cfgp, "w"), sort_keys=False)
ap = "config/api_config.yaml"; a = yaml.safe_load(open(ap)) if os.path.exists(ap) else {}
a = a or {}
a.setdefault(judge, {"model": judge, "endpoints": None, "api_type": "openai",
                     "parallel": 8, "max_tokens": 4096, "temperature": 0.0})
yaml.safe_dump(a, open(ap, "w"), sort_keys=False)
print("patched v0.1 config: judge=", judge, "| models=", models)
PY

python gen_judgment.py --setting-file config/arena-hard-v0.1.yaml
echo "=== win rate (arena-hard-v0.1, judge=$JUDGE, baseline gpt-4-0314) ==="
python show_result.py --benchmark arena-hard-v0.1 --judge-names "$JUDGE" --category arena-hard-v0.1 || true
echo "judgments -> $ARENA_DIR/data/arena-hard-v0.1/model_judgment/$JUDGE/"
