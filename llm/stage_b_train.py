"""
stage_b_train.py — token-level f-DPO training at LLM scale (Stage B).

Score (design doc §3b), summed over completion tokens t, with u_t = π_θ(y_t|s_t)/π_ref(y_t|s_t):

    S(τ) = β · Σ_t [ f'(u_t) − C_Ω(s_t) ]

  --inner exact  : C_Ω(s_t) = Σ_a π_θ(a|s_t)·Φ(u_a)   (full-vocab sum; the π_θ weight tames the
                    1/u tail, so this arm is numerically stable — it is TDPO's per-state term)
  --inner sample : C_Ω(s_t) ≈ Φ(u_{y_t})              (single logged token — the off-policy
                    single-sample estimate a real preference dataset gives; heavy-tailed for
                    non-admissible Ω, exact (=1) for RKL)

Loss:  L = −log σ( S(τ_w) − S(τ_l) ).

RKL correctness anchor (§1): Φ_RKL ≡ 1 and f'_RKL = ln u + 1, so f'−Φ = ln u and BOTH arms collapse
to S = β·Σ log(π_θ/π_ref) = the standard DPO implicit reward. So `--div kl` must reproduce stock
DPO; `--selftest` checks this to 1e-5 on synthetic logits (no models / no GPU needed).

Caveats carried from the design doc:
  · χ²  — not DPO-inducing (Pipano): report separately; a χ² failure is NOT evidence for the thesis.
  · euc — Bregman, not an f-divergence: the inner integrand h_a = (π−π_ref) − (π−π_ref)²/(2π)
          depends on π,π_ref separately (not on u alone), so it has no Φ(u) — but it IS a valid
          per-action expectation, so single-sample works: S = β Σ d²/(2π_θ), d=π_θ−π_ref (1/π_θ tail
          ⇒ noise, like the f-divs; freezes at π_θ=π_ref, so it also needs the SFT init).

Run (cluster, after prefetch + pair JSONL):
  python stage_b_train.py --ref Qwen/Qwen3-1.7B-Base --data data/uf_pairs_train.jsonl \
      --div kl --inner sample --beta 0.1 --steps 1000 --out results/stageB_kl_sample
Validate the math first (seconds, CPU):
  python stage_b_train.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from divergences import (KEYS, SHORT, DEFAULT_ADIV_A,
                         fprime_from_logu, phi_from_logu, exact_C)
# transformers is imported lazily (in load_model / main) so --selftest runs without it.


# ─────────────────────────────────────────────────────────────────────────────────────────
# The score — a pure function of logits, so it is unit-testable on synthetic data (--selftest)
# with no model or GPU. Position t predicts token t+1; we score positions whose predicted token
# is a completion (response) token.
# ─────────────────────────────────────────────────────────────────────────────────────────
def score_from_logits(lg_pol, lg_ref, ids, comp, key, beta, inner, adiv_a, clamp, kln=False):
    """S(τ) = β Σ_t[f'(u_t) − C_Ω(s_t)] over completion tokens. Differentiable in lg_pol.
    lg_pol, lg_ref : [T, V] raw logits.  ids : [T].  comp : [T] bool (True on response tokens).
    kln=True applies the KL-normalization (f'(1)=1) to the f-divergence (not euc)."""
    lp, lr = lg_pol[:-1], lg_ref[:-1]                 # [T-1, V]: row t predicts token t+1
    tgt = ids[1:]                                     # realized next tokens
    m = comp[1:].to(lp.dtype)                         # score where the predicted token is response
    if m.sum() == 0:
        return lg_pol.new_zeros(())

    # realized-token log-ratio via logsumexp — no full softmax needed for the sample arm
    lse_pol = torch.logsumexp(lp, -1); lse_ref = torch.logsumexp(lr, -1)
    logp_y_pol = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - lse_pol
    logp_y_ref = lr.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - lse_ref
    log_u = logp_y_pol - logp_y_ref
    if clamp is not None:
        log_u = log_u.clamp(-clamp, clamp)            # heavy-tail guard (§4) — identical for every Ω

    if key == "euc":                                  # Bregman: chosen term = [∇Ω]_a = π_θ(a) − π_ref(a)
        p_y_pol = logp_y_pol.exp()
        chosen = p_y_pol - logp_y_ref.exp()           # d = π_θ(y) − π_ref(y)
        if inner == "sample":                         # single-sample inner h = d − d²/(2π_θ) ⇒ S = β Σ d²/(2π_θ)
            inner_c = chosen - chosen * chosen / (2.0 * p_y_pol.clamp_min(1e-12))   # 1/π_θ tail (guard floor)
        else:
            inner_c = exact_C("euc", torch.log_softmax(lp, -1), torch.log_softmax(lr, -1),
                              adiv_a, dtype=lp.dtype)
    else:
        chosen = fprime_from_logu(key, log_u, adiv_a, kln=kln)
        if inner == "sample":
            inner_c = phi_from_logu(key, log_u, adiv_a, kln=kln)
        else:                                         # exact vocab sum (fp32 for training)
            inner_c = exact_C(key, torch.log_softmax(lp, -1), torch.log_softmax(lr, -1),
                              adiv_a, dtype=lp.dtype, kln=kln)
    return beta * ((chosen - inner_c) * m).sum()


# ─────────────────────────────────────────────────────────────────────────────────────────
# Self-test: on random logits, RKL's sample and exact arms must both equal the standard DPO
# reward β·Σ log(π_θ/π_ref); the torch f'/Φ must match the numpy reference in python/regularizers.py.
# ─────────────────────────────────────────────────────────────────────────────────────────
def selftest():
    torch.manual_seed(0)
    T, V, beta = 24, 64, 0.13
    lg_pol = torch.randn(T, V, dtype=torch.float64)
    lg_ref = torch.randn(T, V, dtype=torch.float64)
    ids = torch.randint(0, V, (T,))
    comp = torch.zeros(T, dtype=torch.bool); comp[6:] = True     # first 6 = "prompt"

    # standard DPO reward = β·Σ_completion log(π_θ/π_ref) at the realized tokens
    lp = torch.log_softmax(lg_pol[:-1], -1); lr = torch.log_softmax(lg_ref[:-1], -1)
    tgt, m = ids[1:], comp[1:]
    logu = (lp.gather(-1, tgt[:, None]) - lr.gather(-1, tgt[:, None])).squeeze(-1)
    dpo_ref = beta * (logu * m).sum().item()

    ok = True
    for arm in ("sample", "exact"):
        S = score_from_logits(lg_pol, lg_ref, ids, comp, "kl", beta, arm, DEFAULT_ADIV_A, None).item()
        d = abs(S - dpo_ref)
        print(f"  RKL {arm:6s}: S={S:+.6f}  vs standard-DPO={dpo_ref:+.6f}  |Δ|={d:.2e}  "
              + ("OK" if d < 1e-9 else "FAIL"))
        ok &= d < 1e-9

    # exact_C(kl) ≡ 1 by arithmetic
    C = exact_C("kl", torch.log_softmax(lg_pol, -1), torch.log_softmax(lg_ref, -1)).sub(1).abs().max().item()
    print(f"  exact_C(RKL) − 1 : max |Δ|={C:.2e}  " + ("OK" if C < 1e-9 else "FAIL")); ok &= C < 1e-9

    # port check: torch f'(u), Φ(u) vs numpy regularizers on random u∈[1e-4,1e2]
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
        from regularizers import REG
        u = np.exp(np.random.default_rng(1).uniform(-9, 4.6, size=500))
        lu = torch.tensor(np.log(u), dtype=torch.float64)
        for k in ("kl", "rkl", "js", "hel", "chi2", "adiv"):
            fp = fprime_from_logu(k, lu).numpy(); fp_np = REG[k].spec.fp(u)
            ph = phi_from_logu(k, lu).numpy();    ph_np = np.atleast_1d(REG[k].Phi(u)) * np.ones_like(u)
            e = max(np.max(np.abs(fp - fp_np) / (np.abs(fp_np) + 1e-9)),
                    np.max(np.abs(ph - ph_np) / (np.abs(ph_np) + 1e-9)))
            print(f"  port {SHORT[k]:6s}: max rel |Δ|(f', Φ) vs numpy = {e:.2e}  " + ("OK" if e < 1e-6 else "FAIL"))
            ok &= e < 1e-6
    except Exception as ex:                                       # numpy ref not importable → skip
        print(f"  (port check skipped: {ex})")

    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ─────────────────────────────────────────────────────────────────────────────────────────
# Models / data
# ─────────────────────────────────────────────────────────────────────────────────────────
def load_model(name, train):
    from transformers import AutoModelForCausalLM
    try:
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
    m.to("cuda")
    if train:
        m.train(); m.gradient_checkpointing_enable()
    else:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return m


def _ids(x):
    if hasattr(x, "ids"):
        return list(x.ids)
    if hasattr(x, "input_ids") or isinstance(x, dict):
        v = x["input_ids"]
        return list(v[0]) if v and isinstance(v[0], (list, tuple)) else list(v)
    return list(x)


def _encode_side(tok, msgs, max_len, kw):
    full = _ids(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, **kw))
    prompt = _ids(tok.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True, **kw))
    ids = torch.tensor(full[:max_len], dtype=torch.long)
    comp = torch.zeros(len(ids), dtype=torch.bool)
    comp[min(len(prompt), len(ids)):] = True
    return ids, comp


def encode_pair(tok, ex, max_len, kw):
    """One preference example {'chosen':[msgs], 'rejected':[msgs]} → ((ids_w,comp_w),(ids_l,comp_l))."""
    c, r = ex.get("chosen"), ex.get("rejected")
    if not (isinstance(c, list) and c and isinstance(r, list) and r):
        return None
    cw, cl = _encode_side(tok, c, max_len, kw), _encode_side(tok, r, max_len, kw)
    if cw[1].sum() == 0 or cl[1].sum() == 0:
        return None
    return cw, cl


def _logits(model, ids):
    return model(ids.unsqueeze(0).to("cuda")).logits[0].float()      # [T, V] fp32


def pair_scores(policy, ref, enc, key, beta, inner, adiv_a, clamp, kln=False):
    (iw, cw), (il, cl) = enc
    with torch.no_grad():
        rw, rl = _logits(ref, iw), _logits(ref, il)
    Sw = score_from_logits(_logits(policy, iw), rw, iw.to("cuda"), cw.to("cuda"), key, beta, inner, adiv_a, clamp, kln)
    Sl = score_from_logits(_logits(policy, il), rl, il.to("cuda"), cl.to("cuda"), key, beta, inner, adiv_a, clamp, kln)
    return Sw, Sl


@torch.no_grad()
def evaluate(policy, ref, tok, ds, key, beta, inner, adiv_a, clamp, max_len, kw, n, kln=False):
    policy.eval()
    acc = tot = 0
    margins = []
    for ex in ds[:n]:
        enc = encode_pair(tok, ex, max_len, kw)
        if enc is None:
            continue
        Sw, Sl = pair_scores(policy, ref, enc, key, beta, inner, adiv_a, clamp, kln)
        acc += int(Sw.item() > Sl.item()); tot += 1; margins.append(Sw.item() - Sl.item())
    policy.train()
    return {"eval_acc": acc / max(tot, 1), "eval_margin": float(np.mean(margins)) if margins else 0.0,
            "eval_n": tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="validate the score math on synthetic logits, then exit")
    ap.add_argument("--ref", default="Qwen/Qwen3-1.7B-Base", help="frozen reference; also the policy init")
    ap.add_argument("--policy", default=None, help="policy init (default = --ref)")
    ap.add_argument("--data", default="data/uf_pairs_train.jsonl")
    ap.add_argument("--div", default="kl", choices=KEYS)
    ap.add_argument("--inner", default="sample", choices=["sample", "exact"])
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--grad-accum", type=int, default=8, help="pairs per optimizer step")
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--adiv-a", type=float, default=DEFAULT_ADIV_A)
    ap.add_argument("--kln", action="store_true", help="KL-normalize the generator (f'(1)=1); no freeze from π_ref init. Not for euc.")
    ap.add_argument("--clamp", type=float, default=15.0, help="clamp |log u| (heavy-tail guard, §4); 0 disables")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm (raise to relax the aggressive default)")
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--eval-n", type=int, default=128)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/stageB")
    ap.add_argument("--save-policy", action="store_true", help="save the trained policy at the end")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(selftest())

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    clamp = args.clamp if args.clamp and args.clamp > 0 else None
    if args.kln and args.div == "euc":
        raise SystemExit("euc is excluded from kln (Bregman, no f(u) generator) — drop --kln or --div euc")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.policy or args.ref)
    kw = {}
    try:                                            # Qwen3: suppress the <think> block
        tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, enable_thinking=False)
        kw = {"enable_thinking": False}
    except TypeError:
        pass

    ref = load_model(args.ref, train=False)
    policy = load_model(args.policy or args.ref, train=True)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95))

    ds = [json.loads(l) for l in open(args.data) if l.strip()]
    rng = np.random.default_rng(args.seed); rng.shuffle(ds)
    n_eval = max(args.eval_n, int(len(ds) * args.eval_frac))
    ds_eval, ds_train = ds[:n_eval], ds[n_eval:]

    def train_iter():
        while True:
            order = rng.permutation(len(ds_train))
            for j in order:
                yield ds_train[j]
    it = train_iter()

    hist = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()          # measure the TRAINING peak (exclude load transients)
    t0 = time.time()
    print(f"=== Stage B: Ω={SHORT[args.div]} (key {args.div}) inner={args.inner} kln={args.kln} beta={args.beta} "
          f"lr={args.lr} steps={args.steps} accum={args.grad_accum} | train={len(ds_train)} eval={len(ds_eval)} ===")
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        losses, accs = [], []
        got = 0
        while got < args.grad_accum:
            enc = encode_pair(tok, next(it), args.max_len, kw)
            if enc is None:
                continue
            Sw, Sl = pair_scores(policy, ref, enc, args.div, args.beta, args.inner, args.adiv_a, clamp, args.kln)
            loss = -F.logsigmoid(Sw - Sl)
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at step {step} — skipping this pair"); continue
            (loss / args.grad_accum).backward()
            losses.append(loss.item()); accs.append(int(Sw.item() > Sl.item())); got += 1
        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip).item()
        opt.step()

        if step % args.log_every == 0 or step == 1:
            rec = {"step": step, "loss": float(np.mean(losses)), "train_acc": float(np.mean(accs)),
                   "grad_norm": gnorm, "sec": round(time.time() - t0, 1)}
            if torch.cuda.is_available():             # cost of the inner-term arm: peak GPU memory
                rec["gpu_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                rec["gpu_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
            rec.update(evaluate(policy, ref, tok, ds_eval, args.div, args.beta, args.inner,
                                args.adiv_a, clamp, args.max_len, kw, args.eval_n, args.kln))
            hist.append(rec)
            print(f"  step {step:4d}  loss {rec['loss']:.4f}  train_acc {rec['train_acc']:.3f}  "
                  f"eval_acc {rec['eval_acc']:.3f}  margin {rec['eval_margin']:+.3f}  |g| {gnorm:.2f}  "
                  f"{rec['sec']:.0f}s  mem {rec.get('gpu_reserved_gb', '?')}G")
            with open(args.out + ".json", "w") as f:
                json.dump({"args": vars(args), "history": hist}, f, indent=2)

    if args.save_policy:
        policy.save_pretrained(args.out + "_policy"); tok.save_pretrained(args.out + "_policy")
        print("saved policy ->", args.out + "_policy")
    print("done ->", args.out + ".json")


if __name__ == "__main__":
    main()
