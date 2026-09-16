"""
merge_adapter.py — fold a LoRA adapter into its base model and write an ordinary causal-LM directory.

stage_b_train.py merges automatically at the end of a successful run. This script is the rescue path
for the case that motivates --save-adapter-every: a multi-day run that died (walltime, node failure)
after writing {out}_adapter but before the final merge. The adapter alone is useless to gen_bench.py;
merged, it is indistinguishable from a normally-finished policy.

    python merge_adapter.py --adapter results/bench/stageB_lora8b_kl_canon_bdpo_token_adapter \
                            --base Qwen/Qwen3-8B-Base \
                            --out  results/bench/stageB_lora8b_kl_canon_bdpo_token_policy

Runs on CPU by default (no GPU needed, ~16G of RAM for 8B). The tokenizer is copied from the adapter
directory when it is there, else from --base, so the saved chat_template matches what training used.
"""
import argparse, os, torch

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True, help="directory written by --save-adapter-every")
ap.add_argument("--base", default="Qwen/Qwen3-8B-Base", help="the model the adapter was trained on")
ap.add_argument("--out", required=True, help="destination for the merged causal-LM directory")
args = ap.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print(f"base    : {args.base}")
print(f"adapter : {args.adapter}")
try:
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
except TypeError:
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()

os.makedirs(args.out, exist_ok=True)
merged.save_pretrained(args.out)
tok_src = args.adapter if os.path.exists(os.path.join(args.adapter, "tokenizer_config.json")) else args.base
AutoTokenizer.from_pretrained(tok_src).save_pretrained(args.out)
print(f"merged -> {args.out}  (tokenizer from {tok_src})")
