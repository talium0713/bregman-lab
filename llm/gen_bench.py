"""
gen_bench.py — generate benchmark responses from a trained policy (cluster GPU, offline). Generalizes
gen_alpaca.py to the 3 eval sets: AlpacaEval 2.0 (single-turn), Arena-Hard v2 (single-turn), MT-Bench
(2-turn). Handles multi-turn: turn k is generated with the model's own turns <k in context.

Output format (--format):
  · alpaca   : [{"instruction", "output", "generator"}]           (single-turn; AlpacaEval / alpaca_eval)
  · fastchat : JSONL {"question_id", "model_id", "choices":[{"index":0,"turns":[resp0(,resp1)]}]}
               (the shared Arena-Hard-auto / FastChat MT-Bench model_answer schema)

Input JSONL (one record/line) — fields tried in order:
  · question_id ← question_id | uid | id | (running index)
  · turns       ← turns (list of user-prompt strings) | [prompt] | [instruction] | [messages[0].content]

Run (cluster):
  # AlpacaEval (greedy 512, alpaca format)
  python gen_bench.py --model results/bench/stageB_kl_newline_bench1p7b_policy --generator RKL-newline \
      --prompts data/alpaca_eval.jsonl --format alpaca --max-new 512 --out results/bench/alpaca_kl_newline.json
  # Arena-Hard v2 (single-turn, fastchat format)
  python gen_bench.py --model <policy> --generator <name> --prompts data/arena_hard_v2.jsonl \
      --format fastchat --max-new 2048 --out results/bench/arena/<name>.jsonl
  # MT-Bench (2-turn, fastchat format)
  python gen_bench.py --model <policy> --generator <name> --prompts data/mt_bench.jsonl \
      --format fastchat --max-new 1024 --out results/bench/mtbench/<name>.jsonl
"""
import argparse, json, os, time
import torch


def load_questions(path):
    """→ list of (question_id, [user_turn_str, ...])."""
    Q = []
    for k, l in enumerate(open(path)):
        l = l.strip()
        if not l:
            continue
        j = json.loads(l)
        qid = j.get("question_id", j.get("uid", j.get("id", k)))
        turns = j.get("turns")
        if turns is None:
            one = j.get("prompt") or j.get("instruction")
            if one is None and isinstance(j.get("messages"), list) and j["messages"]:
                one = j["messages"][0].get("content")
            turns = [one] if one else []
        # normalize turn entries to strings (some sets store {"content": ...})
        turns = [t.get("content") if isinstance(t, dict) else t for t in turns]
        turns = [t for t in turns if t]
        if turns:
            Q.append((qid, turns))
    return Q


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="policy or SFT baseline checkpoint dir")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--generator", required=True, help="model name tag in the output")
    ap.add_argument("--format", choices=["alpaca", "fastchat"], default="fastchat")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default); >0 = sampling")
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

    Q = load_questions(args.prompts)
    if args.max_prompts > 0:
        Q = Q[:args.max_prompts]
    gen_kw = dict(max_new_tokens=args.max_new, pad_token_id=tok.eos_token_id)
    if args.temperature and args.temperature > 0:
        gen_kw.update(do_sample=True, temperature=args.temperature, top_p=0.9)
    else:
        gen_kw.update(do_sample=False)

    alpaca_out, fastchat_out, t0 = [], [], time.time()
    for i, (qid, turns) in enumerate(Q):
        msgs, responses = [], []
        for turn in turns:                          # multi-turn: model's own prior turns stay in context
            msgs.append({"role": "user", "content": turn})
            ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                          return_tensors="pt", **kw).to("cuda")
            g = model.generate(ids, **gen_kw)
            resp = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()
            responses.append(resp)
            msgs.append({"role": "assistant", "content": resp})
        if args.format == "alpaca":
            alpaca_out.append({"instruction": turns[0], "output": responses[0], "generator": args.generator})
        else:
            fastchat_out.append({"question_id": qid, "model_id": args.generator,
                                 "choices": [{"index": 0, "turns": responses}]})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(Q)}  {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.format == "alpaca":
        json.dump(alpaca_out, open(args.out, "w"), indent=2)
    else:
        with open(args.out, "w") as f:
            for r in fastchat_out:
                f.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}  ({len(Q)} questions, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
