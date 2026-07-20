"""
make_uf_jsonl.py — dump UltraFeedback 'chosen' conversations to JSONL.

Run this on a machine with normal internet + `datasets` installed (e.g. your Mac — plain PyPI, no
Alliance dummy pyarrow):
    pip install datasets
    python make_uf_jsonl.py --split test_prefs --max 1000 --out data/uf_test_prefs.jsonl
Then copy it to the cluster:
    scp data/uf_test_prefs.jsonl talium@killarney.alliancecan.ca:/scratch/talium/bregman-lab/llm/data/

This keeps `datasets`/pyarrow OFF the cluster — the Stage A job reads the JSONL with plain json.
"""
import argparse, json, os
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
ap.add_argument("--split", default="test_prefs")
ap.add_argument("--max", type=int, default=1000)
ap.add_argument("--out", default="data/uf_test_prefs.jsonl")
a = ap.parse_args()

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
d = load_dataset(a.dataset, split=a.split)
n = 0
with open(a.out, "w") as f:
    for ex in d:
        if n >= a.max:
            break
        chosen = ex.get("chosen")
        if isinstance(chosen, list) and chosen:
            f.write(json.dumps({"chosen": chosen}) + "\n"); n += 1
print(f"wrote {a.out}  ({n} examples from {a.dataset}:{a.split})")
