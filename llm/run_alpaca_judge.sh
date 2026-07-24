#!/bin/bash
# AlpacaEval stage 2 (JUDGING) — run on your Mac (needs internet + OPENAI_API_KEY), after scp-ing the
# cluster-generated results/alpaca_*.json back. Judges each candidate's outputs vs the SFT baseline
# with the paper's default GPT-4 annotator (weighted_alpaca_eval_gpt4_turbo), then prints win rates.
# Covers BOTH granularities: token-level (alpaca_{div}.json) and newline (alpaca_{div}_newline.json).
#
#   pip install alpaca-eval
#   export OPENAI_API_KEY=sk-...
#   bash run_alpaca_judge.sh
#
# Win rate = fraction of the 805 prompts where the judge prefers the candidate's response over the SFT
# baseline's (arXiv:2512.00778 B.3). RKL(kl) token should land near their reported DPO win rate; the
# token vs newline gap shows the granularity effect per divergence.
set -uo pipefail
cd "$(dirname "$0")"
REF="results/alpaca_sft_base.json"
ANN="weighted_alpaca_eval_gpt4_turbo"

[ -f "$REF" ] || { echo "missing $REF — generate it first (jobs/gen_alpaca.slrm task 0)"; exit 1; }
[ -n "${OPENAI_API_KEY:-}" ] || { echo "set OPENAI_API_KEY first"; exit 1; }

CANDS=()
for d in kl adiv rkl js hel chi2; do CANDS+=("$d" "${d}_newline"); done   # token + newline per divergence

for c in "${CANDS[@]}"; do
  [ -f "results/alpaca_${c}.json" ] || { echo "skip $c (no results/alpaca_${c}.json)"; continue; }
  echo "=== judging $c vs SFT baseline ==="
  alpaca_eval --model_outputs "results/alpaca_${c}.json" \
              --reference_outputs "$REF" \
              --annotators_config "$ANN" \
              --output_path "results/ae_${c}" || echo "  (alpaca_eval failed for $c)"
done

echo
echo "=== win-rate summary (candidate vs SFT baseline, GPT-4 judge) ==="
python3 - <<'PY'
import csv, glob
base = {"kl":"RKL","adiv":"alpha-div","rkl":"FKL","js":"JS","hel":"Hel","chi2":"chi2"}
def wr(tag):
    fs = glob.glob(f"results/ae_{tag}/**/leaderboard.csv", recursive=True) + glob.glob(f"results/ae_{tag}/*leaderboard*.csv")
    if not fs:
        return None, None
    try:
        r = list(csv.DictReader(open(fs[-1])))[0]
        return (r.get("win_rate") or r.get("winrate") or "-",
                r.get("length_controlled_winrate") or r.get("lc_win_rate") or "-")
    except Exception as e:
        return f"err:{e}", "-"
print(f"{'Omega':10s} {'token_win':>10s} {'token_LC':>9s} {'newline_win':>12s} {'newline_LC':>11s}")
for d in ["kl","adiv","rkl","js","hel","chi2"]:
    tw, tl = wr(d); nw, nl = wr(f"{d}_newline")
    f = lambda x: "-" if x is None else str(x)
    print(f"{base[d]:10s} {f(tw):>10s} {f(tl):>9s} {f(nw):>12s} {f(nl):>11s}")
PY
