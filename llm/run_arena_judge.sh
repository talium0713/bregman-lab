#!/bin/bash
# Arena-Hard v2.0 JUDGING (login node — needs internet + OPENAI_API_KEY with gpt-4.1 access). NOT sbatch
# (compute nodes are offline). Uses the official lmarena/arena-hard-auto pipeline with judge = gpt-4.1
# (matches the Qwen3 tech report / Arena-Hard v2). Baseline is per-category (hard_prompt→o3-mini,
# creative_writing→gemini-2.0-flash) and ships with the HF data — we only generate + judge OUR models.
#
# Prereq: gen_bench_arena.slrm produced results/bench/arena/{tag}.jsonl (arenav2 schema).
# Provide the key: export OPENAI_API_KEY=sk-...  OR  echo sk-... > openai_key.txt (gitignored).
# One-time clone (~5 min) happens automatically. Long judge run (750 prompts × models) → use tmux.
#   bash run_arena_judge.sh
set -uo pipefail
cd "$(dirname "$0")"; LLM_DIR="$(pwd)"
ARENA_DIR="${ARENA_DIR:-$HOME/arena-hard-auto}"
JUDGE="${JUDGE:-gpt-4.1}"

[ -z "${OPENAI_API_KEY:-}" ] && [ -f openai_key.txt ] && export OPENAI_API_KEY="$(tr -d '[:space:]' < openai_key.txt)"
[ -n "${OPENAI_API_KEY:-}" ] || { echo "no OPENAI_API_KEY — export it or put it in $(pwd)/openai_key.txt"; exit 1; }
ls results/bench/arena/*.jsonl >/dev/null 2>&1 || { echo "no results/bench/arena/*.jsonl — run gen_bench_arena.slrm first"; exit 1; }

# ── one-time: clone repo + HF data (questions + per-category baseline answers) + install ──
if [ ! -d "$ARENA_DIR" ]; then
  echo "=== cloning arena-hard-auto + data (one-time) ==="
  git clone https://github.com/lmarena/arena-hard-auto.git "$ARENA_DIR"
  git lfs install
  git clone https://huggingface.co/datasets/lmarena-ai/arena-hard-auto "$ARENA_DIR/_hfdata"
  cp -r "$ARENA_DIR/_hfdata/data" "$ARENA_DIR/"
  ( cd "$ARENA_DIR" && pip install -r requirements.txt && pip install -r requirements-optional.txt pyyaml )
fi

# ── drop our generated answers into the repo's model_answer dir ──
DST="$ARENA_DIR/data/arena-hard-v2.0/model_answer"; mkdir -p "$DST"
cp "$LLM_DIR"/results/bench/arena/*.jsonl "$DST/"
MODELS=(); for f in "$LLM_DIR"/results/bench/arena/*.jsonl; do MODELS+=("$(basename "$f" .jsonl)"); done
echo "=== judging models: ${MODELS[*]}  (judge=$JUDGE) ==="

# ── patch configs: judge_model + model_list, and ensure the judge has an openai endpoint ──
cd "$ARENA_DIR"
python - "$JUDGE" "${MODELS[@]}" <<'PY'
import sys, yaml, os
judge, models = sys.argv[1], sys.argv[2:]
cfgp = "config/arena-hard-v2.0.yaml"
c = yaml.safe_load(open(cfgp)) or {}
c["judge_model"] = judge; c["model_list"] = models
yaml.safe_dump(c, open(cfgp, "w"), sort_keys=False)
ap = "config/api_config.yaml"
a = yaml.safe_load(open(ap)) if os.path.exists(ap) else {}
a = a or {}
a.setdefault(judge, {"model": judge, "endpoints": None, "api_type": "openai",
                     "parallel": 8, "max_tokens": 4096, "temperature": 0.0})
yaml.safe_dump(a, open(ap, "w"), sort_keys=False)
print("patched:", cfgp, "judge=", judge, "| models=", models)
PY

# ── optional: markdown metadata for style control (won't fail the run if absent) ──
[ -f add_markdown_info.py ] && python add_markdown_info.py --input-dir "$DST" --output-dir "$DST" 2>/dev/null || true

# ── judge (gpt-4.1) then win-rate ──
python gen_judgment.py
echo "=== win rate (hard_prompt, judge=$JUDGE) ==="
python show_result.py --judge-names "$JUDGE" --category hard_prompt || true
echo "=== win rate + STYLE CONTROL (markdown+length) ==="
python show_result.py --judge-names "$JUDGE" --category hard_prompt --control-features markdown length || true
echo "=== creative_writing ==="
python show_result.py --judge-names "$JUDGE" --category creative_writing || true
echo "judgments -> $ARENA_DIR/data/arena-hard-v2.0/model_judgment/$JUDGE/"
