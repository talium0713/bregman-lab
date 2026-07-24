#!/bin/bash
# AlpacaEval stage 2 (JUDGING) — run on your Mac (needs internet + OPENAI_API_KEY), after scp-ing the
# cluster-generated results/alpaca_*.json back. Judges each divergence's outputs vs the SFT baseline
# with the paper's default GPT-4 annotator (weighted_alpaca_eval_gpt4_turbo), then prints win rates.
#
#   pip install alpaca-eval
#   export OPENAI_API_KEY=sk-...
#   bash run_alpaca_judge.sh
#
# Win rate = fraction of the 805 prompts where the judge prefers the divergence model's response over
# the SFT baseline's (arXiv:2512.00778 B.3). RKL (kl) should land near their reported DPO win rate.
set -uo pipefail
cd "$(dirname "$0")"
REF="results/alpaca_sft_base.json"
ANN="weighted_alpaca_eval_gpt4_turbo"

[ -f "$REF" ] || { echo "missing $REF — generate it first (jobs/gen_alpaca.slrm task 0)"; exit 1; }
[ -n "${OPENAI_API_KEY:-}" ] || { echo "set OPENAI_API_KEY first"; exit 1; }

for d in kl adiv rkl js hel chi2; do
  [ -f "results/alpaca_${d}.json" ] || { echo "skip $d (no results/alpaca_${d}.json)"; continue; }
  echo "=== judging $d vs SFT baseline ==="
  alpaca_eval --model_outputs "results/alpaca_${d}.json" \
              --reference_outputs "$REF" \
              --annotators_config "$ANN" \
              --output_path "results/ae_${d}" || echo "  (alpaca_eval failed for $d)"
done

echo
echo "=== win-rate summary (candidate vs SFT baseline, GPT-4 judge) ==="
python3 - <<'PY'
import csv, glob, os
label = {"kl":"RKL","adiv":"alpha-div","rkl":"FKL","js":"JS","hel":"Hel","chi2":"chi2"}
print(f"{'Omega':10s} {'win_rate':>9s} {'LC_win':>9s}")
for d in ["kl","adiv","rkl","js","hel","chi2"]:
    best = None
    for f in glob.glob(f"results/ae_{d}/**/leaderboard.csv", recursive=True) + glob.glob(f"results/ae_{d}/*leaderboard*.csv"):
        best = f
    if not best:
        print(f"{label[d]:10s} {'-':>9s} {'-':>9s}   (no leaderboard.csv)"); continue
    try:
        rows = list(csv.DictReader(open(best)))
        r = rows[0]
        wr = r.get("win_rate") or r.get("winrate") or "-"
        lc = r.get("length_controlled_winrate") or r.get("lc_win_rate") or "-"
        print(f"{label[d]:10s} {str(wr):>9s} {str(lc):>9s}")
    except Exception as e:
        print(f"{label[d]:10s} parse-error: {e}")
PY
