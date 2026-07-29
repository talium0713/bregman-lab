"""
eval_length_bias.py — per-pair length-bias probe for the TRL estimator (Stage B, Goal 1).

Forward-only eval (NO training) of a SAVED single-sample newline-f-DPO policy over a held-out
preference set. For every pair it dumps the single-sample score margin AND the newline step
counts K, so the downstream local analysis (fig_length_bias.py) can inject TRL's per-step length
term via the EXACT identity

    S_TRL(τ) = S_single(τ) + β · f'(1) · K          (per response; f'(1)=1−KLN_SHIFT)

  · RKL  (--div kl ): f'(1)=+1  → TRL margin = single margin + β·(K_w−K_l)   (K_w>K_l ⇒ helps)
  · FKL  (--div rkl): f'(1)=−1  → TRL margin = single margin − β·(K_w−K_l)   (⇒ hurts)
  · non-admissible  : f'(1)= 0  → NO length term (TRL≠single is genuine Φ-noise removal, not length)

The identity holds bit-for-bit even with --clamp (the clamp acts on log u_k identically in both f'
and Φ; for RKL f'−Φ=clamp(log u_k), f'=clamp(log u_k)+1, so TRL−single=Σ1=K regardless). So the
counterfactual +β·f'(1)·(K_w−K_l) reconstructs TRL from the single model exactly — proving TRL's
accuracy change on RKL/FKL is manufactured by (K_w−K_l), not by the estimator learning more.

Reuses stage_b_train's encoder/scorer so scores match the trainer exactly.

    python eval_length_bias.py \
        --policy results/matched/stageB_kl_newline_match1p7b_policy \
        --ref    results/sft_uc_1p7b_model \
        --div kl --eval-data data/uf_pairs_test.jsonl \
        --beta 0.1 --clamp 15 --max-len 1024 \
        --out results/lenbias/stageB_kl_newline_match1p7b_perpair
"""
from __future__ import annotations

import argparse, csv, json, os

import numpy as np
import torch

from divergences import KEYS, SHORT, KLN_SHIFT, DEFAULT_ADIV_A
from stage_b_train import load_model, encode_pair, pair_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="saved single-sample NEWLINE policy dir (…_policy)")
    ap.add_argument("--ref", required=True, help="frozen reference (matched recipe: results/sft_uc_1p7b_model)")
    ap.add_argument("--div", default="kl", choices=KEYS, help="kl=RKL (admissible), rkl=FKL, js/hel/adiv/chi2=non-adm")
    ap.add_argument("--eval-data", default="data/uf_pairs_test.jsonl")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--adiv-a", type=float, default=DEFAULT_ADIV_A)
    ap.add_argument("--clamp", type=float, default=15.0, help="match the trainer's --clamp (per-step |log u| guard); 0 disables")
    ap.add_argument("--max-len", type=int, default=1024, help="match the trainer's --max-len")
    ap.add_argument("--n", type=int, default=0, help="0 = every pair in --eval-data")
    ap.add_argument("--out", default="results/lenbias/perpair")
    args = ap.parse_args()

    clamp = args.clamp if args.clamp and args.clamp > 0 else None
    fp1 = 1.0 - KLN_SHIFT[args.div]                       # f'(1): kl=+1, rkl=−1, non-adm=0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.policy)
    kw = {}
    try:                                                 # Qwen3: suppress the <think> block (match trainer)
        tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, enable_thinking=False)
        kw = {"enable_thinking": False}
    except TypeError:
        pass

    ref = load_model(args.ref, train=False)
    policy = load_model(args.policy, train=False)

    ds = [json.loads(l) for l in open(args.eval_data) if l.strip()]
    if args.n > 0:
        ds = ds[:args.n]

    rows, skipped = [], 0
    print(f"=== length-bias eval Ω={SHORT[args.div]} (key {args.div}, f'(1)={fp1:+.0f})  "
          f"policy={args.policy}  n={len(ds)}  clamp={args.clamp} ===", flush=True)
    for i, ex in enumerate(ds):
        enc = encode_pair(tok, ex, args.max_len, kw, step_mode="newline")
        if enc is None:
            skipped += 1
            continue
        (iw, cw, sw), (il, cl, sl) = enc
        with torch.no_grad():
            Sw, Sl = pair_scores(policy, ref, enc, args.div, args.beta, "sample",
                                 args.adiv_a, clamp, kln=False, step_size=1)
        Kw = int(sw.max()) + 1 if sw is not None and len(sw) else 1     # newline step count (trainer's K)
        Kl = int(sl.max()) + 1 if sl is not None and len(sl) else 1
        rows.append({"idx": i, "Sw": Sw.item(), "Sl": Sl.item(),
                     "Kw": Kw, "Kl": Kl, "Tw": int(cw.sum()), "Tl": int(cl.sum())})
        if (i + 1) % 100 == 0:
            acc = np.mean([r["Sw"] > r["Sl"] for r in rows])
            print(f"  {i+1}/{len(ds)}  kept={len(rows)}  single-acc={acc:.3f}", flush=True)

    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "Sw", "Sl", "Kw", "Kl", "Tw", "Tl"])
        w.writeheader(); w.writerows(rows)

    m = np.array([r["Sw"] - r["Sl"] for r in rows])                    # single-sample margin
    dK = np.array([r["Kw"] - r["Kl"] for r in rows], dtype=float)      # length gap
    b = args.beta
    acc = lambda x: float(np.mean(np.asarray(x) > 0))
    summary = {
        "div": args.div, "Omega": SHORT[args.div], "fprime1": fp1, "beta": b,
        "n": len(rows), "skipped": skipped,
        "acc_single": acc(m),                                          # baseline: sign of single margin
        "acc_TRL_reconstructed": acc(m + b * fp1 * dK),                # single + β·f'(1)·ΔK  (== TRL for kl/rkl)
        "acc_plus_bK": acc(m + b * dK),                                # +ΔK direction (RKL length help)
        "acc_minus_bK": acc(m - b * dK),                               # −ΔK direction (FKL length hurt)
        "acc_dK0_subset": acc(m[dK == 0]) if np.any(dK == 0) else None,  # length-neutral pairs (Goal 2 preview)
        "n_dK0": int(np.sum(dK == 0)),
        "EdK": float(dK.mean()), "K_w_mean": float(np.mean([r["Kw"] for r in rows])),
        "K_l_mean": float(np.mean([r["Kl"] for r in rows])),
        "frac_dK_pos": float(np.mean(dK > 0)),
    }
    with open(args.out + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("wrote", args.out + ".csv  and  " + args.out + ".summary.json")


if __name__ == "__main__":
    main()
