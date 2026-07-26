#!/bin/bash
# AlpacaEval stage 2 (JUDGING) — the ONLY internet-needing step. Run it where there IS internet:
#   • a Killarney LOGIN node (klogin*, has internet) — reads results/ on /scratch directly, no scp; OR
#   • your Mac (after scp-ing results/alpaca_*.json back).
# NOT via sbatch: compute nodes have no outbound internet, so they cannot reach api.openai.com.
# Judges each candidate vs the SFT baseline with the paper's default GPT-4 annotator
# (weighted_alpaca_eval_gpt4_turbo); covers token (alpaca_{div}.json) and newline (alpaca_{div}_newline.json).
#
# One-time setup (login node / Mac, needs internet — a separate venv keeps it off venv_gpu):
#   module load python/3.11 && virtualenv ~/judge_env && source ~/judge_env/bin/activate
#   pip install alpaca-eval
# Provide the key ONE of two ways (then run `bash run_alpaca_judge.sh`):
#   export OPENAI_API_KEY=sk-...          # env var, OR
#   echo 'sk-...' > openai_key.txt        # a file next to this script (gitignored) — "fill in and go"
# Long run (rate limits): use a persistent session first ->  tmux new -s judge
#
# Win rate = fraction of the 805 prompts where the judge prefers the candidate over the SFT baseline
# (arXiv:2512.00778 B.3). RKL(kl) token should land near their reported DPO win rate; the token vs
# newline gap is the granularity effect per divergence.
set -uo pipefail
cd "$(dirname "$0")"
REF="results/matched/alpaca_sft_base.json"
ANN="weighted_alpaca_eval_gpt4_turbo"

# API key: prefer $OPENAI_API_KEY, else read openai_key.txt (gitignored). "기입만 하면" = drop the key there.
if [ -z "${OPENAI_API_KEY:-}" ] && [ -f openai_key.txt ]; then
  export OPENAI_API_KEY="$(tr -d '[:space:]' < openai_key.txt)"
fi
[ -n "${OPENAI_API_KEY:-}" ] || { echo "no OPENAI_API_KEY — export it, or put the key in $(pwd)/openai_key.txt"; exit 1; }
command -v alpaca_eval >/dev/null 2>&1 || { echo "alpaca_eval not installed — on a login node/Mac (internet): pip install alpaca-eval"; exit 1; }
[ -f "$REF" ] || { echo "missing $REF — generate it first (jobs/gen_alpaca.slrm task 0)"; exit 1; }

CANDS=()   # token, newline, newline+kln per divergence (key comparison = newline vs newline+kln)
for d in kl adiv rkl js hel chi2; do CANDS+=("$d" "${d}_newline" "${d}_newline_kln"); done

for c in "${CANDS[@]}"; do
  [ -f "results/matched/alpaca_${c}.json" ] || { echo "skip $c (no results/matched/alpaca_${c}.json)"; continue; }
  echo "=== judging $c vs SFT baseline ==="
  alpaca_eval --model_outputs "results/matched/alpaca_${c}.json" \
              --reference_outputs "$REF" \
              --annotators_config "$ANN" \
              --output_path "results/matched/ae_${c}" || echo "  (alpaca_eval failed for $c)"
done

echo
echo "=== win-rate summary (candidate vs SFT baseline, GPT-4 judge) ==="
python3 - <<'PY'
import csv, glob
base = {"kl":"RKL","adiv":"alpha-div","rkl":"FKL","js":"JS","hel":"Hel","chi2":"chi2"}
def wr(tag):
    fs = glob.glob(f"results/matched/ae_{tag}/**/leaderboard.csv", recursive=True) + glob.glob(f"results/matched/ae_{tag}/*leaderboard*.csv")
    if not fs:
        return None, None
    try:
        r = list(csv.DictReader(open(fs[-1])))[0]
        return (r.get("win_rate") or r.get("winrate") or "-",
                r.get("length_controlled_winrate") or r.get("lc_win_rate") or "-")
    except Exception as e:
        return f"err:{e}", "-"
f = lambda x: "-" if x is None else str(x)
print("win_rate vs SFT baseline  (kln rescue shows up as newline -> newline+kln)")
print(f"{'Omega':10s} {'token':>8s} {'newline':>9s} {'nl+kln':>8s}")
for d in ["kl","adiv","rkl","js","hel","chi2"]:
    tw, _ = wr(d); nw, _ = wr(f"{d}_newline"); kw, _ = wr(f"{d}_newline_kln")
    print(f"{base[d]:10s} {f(tw):>8s} {f(nw):>9s} {f(kw):>8s}")
PY
