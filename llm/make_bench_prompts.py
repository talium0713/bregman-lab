"""
make_bench_prompts.py — fetch benchmark prompt sets (LOGIN NODE, needs internet). Compute nodes are
offline, so run this on a Killarney login node (klogin*) or your Mac.

  Arena-Hard v2.0 : HF dataset lmarena-ai/arena-hard-auto → data/arena-hard-v2.0/question.jsonl
                    (750 = 500 hard_prompt + 250 creative_writing; fields uid/category/subcategory/
                     language/prompt). Saved to data/arena_hard_v2.jsonl — gen_bench.py reads it as-is
                     (uid→question_id, prompt→single turn).
  MT-Bench        : FastChat raw question.jsonl (80 questions, 2-turn) → data/mt_bench.jsonl. [deferred]

    python make_bench_prompts.py --which arena          # Arena-Hard v2 only (current focus)
    python make_bench_prompts.py --which arena mtbench
"""
import argparse, os, shutil, urllib.request


def fetch_arena(out):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id="lmarena-ai/arena-hard-auto", repo_type="dataset",
                        filename="data/arena-hard-v2.0/question.jsonl")
    shutil.copy(p, out)
    n = sum(1 for _ in open(out))
    print(f"arena-hard-v2: {n} questions -> {out}")


def fetch_mtbench(out):
    url = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
    urllib.request.urlretrieve(url, out)
    n = sum(1 for _ in open(out))
    print(f"mt-bench: {n} questions -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["arena"], choices=["arena", "mtbench"])
    ap.add_argument("--outdir", default="data")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if "arena" in a.which:
        fetch_arena(os.path.join(a.outdir, "arena_hard_v2.jsonl"))
    if "mtbench" in a.which:
        fetch_mtbench(os.path.join(a.outdir, "mt_bench.jsonl"))
