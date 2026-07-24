"""
make_alpaca_jsonl.py — dump the AlpacaEval eval-set instructions (805 prompts) to JSONL, so the offline
cluster can generate on them (gen_alpaca.py). Run on a machine with internet + `datasets`:
    pip install datasets
    python make_alpaca_jsonl.py --out data/alpaca_eval.jsonl
    scp data/alpaca_eval.jsonl talium@killarney.alliancecan.ca:/scratch/talium/bregman-lab/llm/data/
Each line: {"instruction": "..."}.
"""
import argparse, json, os
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/alpaca_eval.jsonl")
a = ap.parse_args()

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
try:
    d = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval", trust_remote_code=True)
except Exception:
    d = load_dataset("tatsu-lab/alpaca_eval", split="eval", trust_remote_code=True)
n = 0
with open(a.out, "w") as f:
    for ex in d:
        instr = ex.get("instruction")
        if instr:
            f.write(json.dumps({"instruction": instr}) + "\n"); n += 1
print(f"wrote {a.out}  ({n} prompts)")
