"""
gen_alpaca.py — generate AlpacaEval responses from a trained policy (cluster GPU, offline). Greedy,
max 512 new tokens (the arXiv:2512.00778 protocol). Output is AlpacaEval JSON: a list of
{"instruction", "output", "generator"}.

This is stage 1 of a two-stage eval. Stage 2 (judging) runs `alpaca_eval` on a machine WITH an
LLM-judge API (e.g. your Mac with OPENAI_API_KEY), comparing each model's outputs against the SFT
baseline's outputs to get the win rate — see jobs/gen_alpaca.slrm footer for the exact commands.

Run (cluster):
  python gen_alpaca.py --model results/sft_uc_1p7b_model     --generator sft_base --out results/alpaca_sft_base.json
  python gen_alpaca.py --model results/stageB_kl_match1p7b_policy --generator RKL --out results/alpaca_kl.json
"""
import argparse, json, os, time
import torch


def load_prompts(path):
    P = []
    for l in open(path):
        l = l.strip()
        if not l:
            continue
        j = json.loads(l)
        instr = j.get("instruction") or j.get("prompt")
        if not instr and isinstance(j.get("messages"), list) and j["messages"]:
            instr = j["messages"][0].get("content")
        if instr:
            P.append(instr)
    return P


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint dir to generate from (policy or SFT baseline)")
    ap.add_argument("--prompts", default="data/alpaca_eval.jsonl")
    ap.add_argument("--generator", required=True, help="name tag for this model in the AlpacaEval output")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--max-prompts", type=int, default=0, help="0 = all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {}
    try:                                            # Qwen3: suppress the <think> block
        tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, enable_thinking=False)
        kw = {"enable_thinking": False}
    except TypeError:
        pass
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.to("cuda").eval()

    prompts = load_prompts(args.prompts)
    if args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]
    out, t0 = [], time.time()
    for i, instr in enumerate(prompts):
        ids = tok.apply_chat_template([{"role": "user", "content": instr}], tokenize=True,
                                      add_generation_prompt=True, return_tensors="pt", **kw).to("cuda")
        g = model.generate(ids, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()
        out.append({"instruction": instr, "output": text, "generator": args.generator})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(prompts)}  {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}  ({len(out)} responses, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
