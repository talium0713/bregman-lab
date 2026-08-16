#!/bin/bash
# MT-Bench single-answer grading (login node, internet + OPENAI_API_KEY). Uses lm-sys/FastChat llm_judge
# with judge = gpt-4.1 and gpt-5.1 (the paper's two MT-Bench judges) → 1-10 scores (turn1/turn2/avg).
# Judges every results/bench/mtbench/*.jsonl (fastchat format from gen_bench.py --format fastchat).
#   bash run_mtbench_judge.sh          # judges gpt-4.1 then gpt-5.1
#   JUDGES="gpt-4.1" bash run_mtbench_judge.sh   # one judge only
set -uo pipefail
cd "$(dirname "$0")"; LLM_DIR="$(pwd)"
FC="${FC_DIR:-$HOME/FastChat}"
JUDGES="${JUDGES:-gpt-4.1 gpt-5.1}"
[ -z "${OPENAI_API_KEY:-}" ] && [ -f openai_key.txt ] && export OPENAI_API_KEY="$(tr -d '[:space:]' < openai_key.txt)"
[ -n "${OPENAI_API_KEY:-}" ] || { echo "no OPENAI_API_KEY"; exit 1; }
ls results/bench/mtbench/*.jsonl >/dev/null 2>&1 || { echo "no results/bench/mtbench/*.jsonl — run gen first"; exit 1; }

module load StdEnv/2023 python/3.11 2>/dev/null || true
ENV="${MT_ENV:-$HOME/mtbench_env}"
[ -d "$ENV" ] || python -m venv "$ENV"
source "$ENV/bin/activate"
python -m pip install -q --upgrade pip
if [ ! -d "$FC" ]; then
  git clone https://github.com/lm-sys/FastChat.git "$FC"
  ( cd "$FC" && pip install -q -e ".[model_worker,llm_judge]" --index-url https://pypi.org/simple/ )
fi
pip install -q openai anthropic 2>/dev/null || true
JUDGE_DIR="$FC/fastchat/llm_judge"; mkdir -p "$JUDGE_DIR/data/mt_bench/model_answer"

# copy our answers into FastChat's model_answer dir, adding answer_id/tstamp if missing (schema safety)
for f in "$LLM_DIR"/results/bench/mtbench/*.jsonl; do
  python - "$f" "$JUDGE_DIR/data/mt_bench/model_answer/$(basename "$f")" <<'PY'
import json, sys, time, hashlib
src, dst = sys.argv[1], sys.argv[2]
with open(src) as fi, open(dst, "w") as fo:
    for l in fi:
        if not l.strip(): continue
        r = json.loads(l)
        r.setdefault("answer_id", hashlib.md5((str(r["question_id"]) + r["model_id"]).encode()).hexdigest()[:22])
        r.setdefault("tstamp", time.time())
        fo.write(json.dumps(r) + "\n")
PY
done
MODELS=(); for f in "$LLM_DIR"/results/bench/mtbench/*.jsonl; do MODELS+=("$(basename "$f" .jsonl)"); done
echo "=== MT-Bench models: ${MODELS[*]} ==="

cd "$JUDGE_DIR"
for J in $JUDGES; do
  echo "=== gen_judgment (single) judge=$J ==="
  yes | python gen_judgment.py --model-list "${MODELS[@]}" --judge-model "$J" --mode single --parallel 8 2>&1 | tail -4
  echo "=== scores judge=$J (turn1 / turn2 / avg, 1-10) ==="
  python show_result.py --mode single --judge-model "$J" --model-list "${MODELS[@]}" 2>&1 | tail -25
done
