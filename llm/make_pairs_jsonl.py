"""
make_pairs_jsonl.py — dump UltraFeedback preference PAIRS (chosen + rejected) to JSONL for Stage B.

Stage A only needed `chosen`; Stage B (f-DPO) needs both sides of each preference. Run on a machine
with normal internet + `datasets` (e.g. your Mac — plain PyPI, no Alliance dummy pyarrow):
    pip install datasets
    python make_pairs_jsonl.py --split train_prefs --max 20000 --out data/uf_pairs_train.jsonl
    python make_pairs_jsonl.py --split test_prefs  --max 1000  --out data/uf_pairs_test.jsonl
Then copy to the cluster:
    scp data/uf_pairs_*.jsonl talium@killarney.alliancecan.ca:/scratch/talium/bregman-lab/llm/data/

Each line: {"chosen": [msgs...], "rejected": [msgs...]} — the two conversations share the same
prompt and differ only in the final assistant turn. Keeps `datasets`/pyarrow OFF the cluster.
"""
import argparse, json, os
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
ap.add_argument("--split", default="train_prefs")
ap.add_argument("--max", type=int, default=20000)
ap.add_argument("--out", default="data/uf_pairs_train.jsonl")
a = ap.parse_args()

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
d = load_dataset(a.dataset, split=a.split)
n = skip = 0
with open(a.out, "w") as f:
    for ex in d:
        if n >= a.max:
            break
        c, r = ex.get("chosen"), ex.get("rejected")
        if not (isinstance(c, list) and c and isinstance(r, list) and r):
            skip += 1; continue
        # sanity: same prompt, both end on an assistant turn
        if c[:-1] != r[:-1] or c[-1].get("role") != "assistant" or r[-1].get("role") != "assistant":
            skip += 1; continue
        f.write(json.dumps({"chosen": c, "rejected": r}) + "\n"); n += 1
print(f"wrote {a.out}  ({n} pairs from {a.dataset}:{a.split}; skipped {skip})")
