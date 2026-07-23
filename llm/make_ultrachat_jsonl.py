"""
make_ultrachat_jsonl.py — dump UltraChat-200k SFT conversations to JSONL for the Zephyr-style SFT init.

arXiv:2512.00778 (and Zephyr, Tunstall et al. [42]) first SFT the base model on UltraChat-200k, THEN
run DPO on UltraFeedback. To reproduce their Qwen3-1.7B setup we need the same SFT init. Run on a
machine with internet + `datasets` (e.g. your Mac), then scp to the cluster:
    pip install datasets
    python make_ultrachat_jsonl.py --split train_sft --max 100000 --out data/ultrachat_sft.jsonl
    scp data/ultrachat_sft.jsonl talium@killarney.alliancecan.ca:/scratch/talium/bregman-lab/llm/data/

Each line: {"messages": [ {role, content}, ... ]} — a full multi-turn chat. sft_base.py reads it with
--field messages. --max caps the number of conversations (full train_sft ≈ 208k; Zephyr uses 1 epoch).
"""
import argparse, json, os
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
ap.add_argument("--split", default="train_sft")
ap.add_argument("--max", type=int, default=100000)
ap.add_argument("--out", default="data/ultrachat_sft.jsonl")
a = ap.parse_args()

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
d = load_dataset(a.dataset, split=a.split)
n = skip = 0
with open(a.out, "w") as f:
    for ex in d:
        if n >= a.max:
            break
        msgs = ex.get("messages")
        if not (isinstance(msgs, list) and msgs and all(m.get("role") and m.get("content") for m in msgs)):
            skip += 1; continue
        f.write(json.dumps({"messages": msgs}) + "\n"); n += 1
print(f"wrote {a.out}  ({n} conversations from {a.dataset}:{a.split}; skipped {skip})")
