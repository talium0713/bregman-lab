"""
gen_bench.py — generate benchmark responses from a trained policy (cluster GPU, offline). Generalizes
gen_alpaca.py to the 3 eval sets: AlpacaEval 2.0 (single-turn), Arena-Hard v2 (single-turn), MT-Bench
(2-turn). Handles multi-turn: turn k is generated with the model's own turns <k in context.

Output format (--format):
  · alpaca   : [{"instruction", "output", "generator"}]           (single-turn; AlpacaEval / alpaca_eval)
  · fastchat : JSONL {"question_id", "model_id", "choices":[{"index":0,"turns":[resp0(,resp1)]}]}
               (FastChat MT-Bench model_answer schema — data/mt_bench/model_answer/{model}.jsonl)
  · arenav2  : JSONL {"uid","ans_id","model","messages":[{user},{assistant:{content:{answer}}}],
               "tstamp","metadata":{token_len}}  (Arena-Hard v2.0 schema — judge reads
               messages[-1]["content"]["answer"]; drop into data/arena-hard-v2.0/model_answer/{model}.jsonl)

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
    ap.add_argument("--format", choices=["alpaca", "fastchat", "arenav2"], default="fastchat")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16, help="batched generation for single-turn sets (arena/alpaca); MT-Bench multi-turn runs sequentially")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default); >0 = sampling")
    ap.add_argument("--rep-penalty", type=float, default=1.0,
                    help="repetition_penalty (RePO uses 1.05 to kill degenerate loops like 'umably umably…'; 1.0 = off)")
    ap.add_argument("--system", default="",
                    help="optional system prompt prepended to every turn (RePO uses 'You are a helpful assistant.')")
    ap.add_argument("--max-prompts", type=int, default=0, help="0 = all")
    ap.add_argument("--think", choices=["off", "default", "on"], default="off",
                    help="Qwen3 chat mode: off=enable_thinking=False (empty <think> prefill, direct answer — "
                         "matches our training); default=no kwarg (open assistant turn, model may reason — RePO's setup); "
                         "on=enable_thinking=True (open turn, reasoning encouraged)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if getattr(tok, "chat_template", None) is None:
        # Untrained base checkpoints (Llama-3.2-*, gemma-*-pt) ship no chat template, so
        # apply_chat_template below raises. Install the SAME fallback stage_b_train.py installs, so
        # the base reference is generated in exactly the format its trained policies were trained in
        # — otherwise the "before" column is not comparable to the "after" ones.
        tok.chat_template = (
            "{% for m in messages %}"
            "{% if m['role'] == 'user' %}{{ '\n\nHuman: ' + m['content'] }}"
            "{% elif m['role'] == 'assistant' %}{{ '\n\nAssistant: ' + m['content'] + eos_token }}"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ '\n\nAssistant:' }}{% endif %}")
        print(f"[tok] {args.model}: no chat_template -> installed the DPO-style fallback "
              f"(Human/Assistant turns, EOS={tok.eos_token!r})", flush=True)
    kw = {}
    if args.think == "off":
        try:                                        # Qwen3: suppress the <think> block (empty prefill)
            tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, enable_thinking=False)
            kw = {"enable_thinking": False}
        except TypeError:
            pass
    elif args.think == "on":
        try:
            tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, enable_thinking=True)
            kw = {"enable_thinking": True}
        except TypeError:
            pass
    # args.think == "default" -> kw stays {} (no enable_thinking kwarg = RePO's open assistant turn)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.to("cuda").eval()

    Q = load_questions(args.prompts)
    if args.max_prompts > 0:
        Q = Q[:args.max_prompts]
    # STOP tokens: a ChatML assistant turn ends with <|im_end|>, but Qwen3-Base's eos is <|endoftext|>,
    # so generate() would run past the answer to max_new_tokens (→ degenerate repetition) unless we add
    # <|im_end|> as a stop id. (vLLM does this automatically from the chat template; HF .generate does not.)
    stop_ids = [tok.eos_token_id]
    # A chat template normally ends the assistant turn with a special token that is NOT the model's
    # eos, and generate() stops on eos_token_id only — so without these the model writes a perfectly
    # good answer, emits its turn terminator, and then keeps going to max_new_tokens.
    #   Qwen3 <|im_end|>   Gemma <end_of_turn>   Llama-3 <|eot_id|>   Phi <|end|>
    # convert_tokens_to_ids returns the UNK id for a token outside the vocabulary, so that has to be
    # excluded explicitly: for gemma-3-1b-it, "<|im_end|>" resolved to <unk>=3 and was being added as
    # a stop id while the real terminator <end_of_turn> was not, leaving 68.6% of answers at the cap.
    _unk = tok.unk_token_id
    for _t in ("<|im_end|>", "<end_of_turn>", "<|eot_id|>", "<|end|>"):
        _i = tok.convert_tokens_to_ids(_t)
        if isinstance(_i, int) and _i >= 0 and _i != _unk and _i not in stop_ids:
            stop_ids.append(_i)
    gen_kw = dict(max_new_tokens=args.max_new, pad_token_id=tok.eos_token_id, eos_token_id=stop_ids)
    if args.rep_penalty and args.rep_penalty != 1.0:   # RePO=1.05 → suppresses degenerate repetition loops
        gen_kw["repetition_penalty"] = args.rep_penalty
    # STRING stops (RePO's qwen list): catch run-on that never emits a stop TOKEN — e.g. a base model
    # that starts a fake new turn "\n\nQuestion:" instead of <|im_end|>. Needs the tokenizer passed too.
    STOP_STRINGS = ["<|im_end|>", "<|endoftext|>", "</s>", "\n\nQuestion:", "<|end|>"]
    # The fallback template that stage_b_train.py installs for base checkpoints with no chat template
    # delimits turns with "\n\nHuman:" / "\n\nAssistant:" — plain text, not a special token — so a
    # policy that opens a fake next turn never hits a stop. That is what happened to Llama-3.2-1B:
    # 314/500 answers contained "\n\nHuman:" and 94.8% ran to the cap.
    # Derive these from the template rather than adding them globally. For a ChatML model the turn
    # delimiter is <|im_end|>, already above; a Qwen3 answer that merely *contains* "\n\nAssistant:"
    # is the policy failing to emit <|im_end|>, which is a result we are measuring (8B Amari: 45/500)
    # and must not be silently truncated away.
    _ct = getattr(tok, "chat_template", None) or ""
    if "Human: " in _ct and "Assistant:" in _ct:
        STOP_STRINGS = STOP_STRINGS + ["\n\nHuman:", "\n\nAssistant:"]
    gen_kw["stop_strings"] = STOP_STRINGS
    gen_kw["tokenizer"] = tok
    if args.temperature and args.temperature > 0:
        gen_kw.update(do_sample=True, temperature=args.temperature, top_p=0.9)
    else:
        gen_kw.update(do_sample=False)
    print(f"stop_ids={stop_ids} rep_penalty={args.rep_penalty} stop_strings={STOP_STRINGS} "
          f"system={bool(args.system)} (eos={tok.eos_token_id}, turn-terminator stop ids="
          f"{[i for i in stop_ids if i != tok.eos_token_id]})", flush=True)

    import hashlib
    tok.padding_side = "left"                        # decoder-only batched generation needs left padding
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    rows = []

    def emit(qid, turns, responses):
        if args.format == "alpaca":
            rows.append({"instruction": turns[0], "output": responses[0], "generator": args.generator})
        elif args.format == "fastchat":             # MT-Bench model_answer schema
            rows.append({"question_id": qid, "model_id": args.generator,
                         "choices": [{"index": 0, "turns": responses}]})
        else:                                       # arenav2: judge reads messages[-1].content.answer
            ans_id = hashlib.md5(f"{qid}-{args.generator}".encode()).hexdigest()[:22]
            tlen = len(tok(responses[0], add_special_tokens=False).input_ids)
            rows.append({"uid": qid, "ans_id": ans_id, "model": args.generator,
                         "messages": [{"role": "user", "content": turns[0]},
                                      {"role": "assistant", "content": {"answer": responses[0]}}],
                         "tstamp": time.time(), "metadata": {"token_len": tlen}})

    t0 = time.time()
    single_turn = all(len(t) == 1 for _, t in Q)
    if single_turn and args.batch_size > 1:          # BATCHED (AlpacaEval / Arena-Hard) — ~10× faster
        for b0 in range(0, len(Q), args.batch_size):
            batch = Q[b0:b0 + args.batch_size]
            texts = [tok.apply_chat_template(
                         (([{"role": "system", "content": args.system}] if args.system else [])
                          + [{"role": "user", "content": t[0]}]),
                         tokenize=False, add_generation_prompt=True, **kw) for _, t in batch]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            g = model.generate(**enc, **gen_kw)
            plen = enc["input_ids"].shape[1]
            for j, (qid, t) in enumerate(batch):
                resp = tok.decode(g[j, plen:], skip_special_tokens=True).strip()
                emit(qid, [t[0]], [resp])
            print(f"  {min(b0 + args.batch_size, len(Q))}/{len(Q)}  {time.time() - t0:.0f}s", flush=True)
    else:                                            # sequential (MT-Bench multi-turn, or --batch-size 1)
        for i, (qid, turns) in enumerate(Q):
            msgs = [{"role": "system", "content": args.system}] if args.system else []
            responses = []
            for turn in turns:                       # model's own prior turns stay in context
                msgs.append({"role": "user", "content": turn})
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
                enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
                g = model.generate(**enc, **gen_kw)
                resp = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                responses.append(resp); msgs.append({"role": "assistant", "content": resp})
            emit(qid, turns, responses)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(Q)}  {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.format == "alpaca":
        json.dump(rows, open(args.out, "w"), indent=2)
    else:                                           # fastchat / arenav2 → JSONL
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}  ({len(Q)} questions, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
