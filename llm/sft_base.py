"""
sft_base.py — light SFT of the base model on the CHOSEN responses, to make π_θ_init ≠ π_ref.

Stage B's single-sample arm freezes at init when π_θ = π_ref (u≡1), because the score collapses to
S = β Σ_t f(u_t)/u_t whose init gradient ∝ f'(1) = 0 for the standard-normalized f-divergences. A
LIGHT SFT of the base model on the preference-dataset chosen responses moves π_θ off π_ref (u≠1), so
the single-sample score has a nonzero init gradient. π_ref stays the frozen base; π_θ is initialized
from this SFT checkpoint (pass it to stage_b_train via --policy).

Standard next-token SFT loss on the response (completion) tokens only. No TRL (Alliance has no pyarrow).

Run (cluster, after prefetch + data):
  python sft_base.py --model Qwen/Qwen3-1.7B-Base --data data/uf_pairs_train.jsonl \
      --steps 500 --lr 1e-5 --out results/sft_1p7b
Then verify the off-policyness moved (u should spread, min_u ≪ 1):
  python stage_a_measure.py --ref Qwen/Qwen3-1.7B-Base --policy results/sft_1p7b_model \
      --data data/uf_pairs_test.jsonl --max-samples 200 --out results/sft_offpolicy_check
Then Stage B single-sample:
  python stage_b_train.py --ref Qwen/Qwen3-1.7B-Base --policy results/sft_1p7b_model --inner sample ...
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from stage_b_train import _encode_side, load_model      # reuse tokenization + bf16 loader (train mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--data", default="data/uf_pairs_train.jsonl", help="uses each example's 'chosen' conversation")
    ap.add_argument("--steps", type=int, default=500, help="LIGHT: enough to move u off 1, not full convergence")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--out", default="results/sft_1p7b")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {}
    try:                                                # Qwen3: suppress <think>
        tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, enable_thinking=False)
        kw = {"enable_thinking": False}
    except TypeError:
        pass

    model = load_model(args.model, train=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    ds = [json.loads(l) for l in open(args.data) if l.strip()]
    rng = np.random.default_rng(args.seed); rng.shuffle(ds)

    def gen():
        while True:
            for j in rng.permutation(len(ds)):
                yield ds[j]
    it = gen()

    hist = []
    t0 = time.time()
    print(f"=== light SFT {args.model} on chosen | steps={args.steps} lr={args.lr} accum={args.grad_accum} | n={len(ds)} ===")
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        losses, got = [], 0
        while got < args.grad_accum:
            ex = next(it)
            msgs = ex.get("chosen")
            if not (isinstance(msgs, list) and msgs):
                continue
            ids, comp = _encode_side(tok, msgs, args.max_len, kw)
            m = comp[1:]                                 # score the response (completion) tokens only
            if m.sum() == 0:
                continue
            logits = model(ids.unsqueeze(0).to("cuda")).logits[0].float()          # [T, V]
            logp = torch.log_softmax(logits[:-1], -1)                              # predict tokens 1..T-1
            tgt, mm = ids[1:].to("cuda"), m.to("cuda")
            nll = -(logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mm).sum() / mm.sum()
            (nll / args.grad_accum).backward()
            losses.append(nll.item()); got += 1
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        opt.step()
        if step % args.log_every == 0 or step == 1:
            rec = {"step": step, "loss": float(np.mean(losses)), "grad_norm": gnorm, "sec": round(time.time() - t0, 1)}
            hist.append(rec)
            print(f"  step {step:4d}  nll {rec['loss']:.4f}  |g| {gnorm:.2f}  {rec['sec']:.0f}s")
            json.dump({"args": vars(args), "history": hist}, open(args.out + ".json", "w"), indent=2)

    out_model = args.out + "_model"
    model.save_pretrained(out_model); tok.save_pretrained(out_model)
    print("saved SFT checkpoint ->", out_model, "  (use as stage_b_train --policy)")


if __name__ == "__main__":
    main()
