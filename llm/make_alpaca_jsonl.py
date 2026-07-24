"""
make_alpaca_jsonl.py — dump the AlpacaEval eval-set instructions (805 prompts) to JSONL, so the offline
cluster can generate on them (gen_alpaca.py). Run on a machine with internet:
    pip install huggingface_hub
    python make_alpaca_jsonl.py --out data/alpaca_eval.jsonl
    scp data/alpaca_eval.jsonl talium@killarney.alliancecan.ca:/scratch/talium/bregman-lab/llm/data/
Each line: {"instruction": "..."}.

tatsu-lab/alpaca_eval is a script-based dataset (`load_dataset` no longer runs those), so we pull the
raw eval JSON (a list of {"instruction","output",...}) straight from the repo instead.
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/alpaca_eval.jsonl")
a = ap.parse_args()

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
try:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id="tatsu-lab/alpaca_eval", filename="alpaca_eval.json", repo_type="dataset")
    data = json.load(open(path))
except Exception as e:
    print(f"hf_hub_download failed ({e}); falling back to direct URL")
    import urllib.request
    url = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"
    data = json.load(urllib.request.urlopen(url))

n = 0
with open(a.out, "w") as f:
    for ex in data:
        instr = ex.get("instruction")
        if instr:
            f.write(json.dumps({"instruction": instr}) + "\n"); n += 1
print(f"wrote {a.out}  ({n} prompts)")
