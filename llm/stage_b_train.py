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
def score_from_logits(lg_pol, lg_ref, ids, comp, key, beta, inner, adiv_a, clamp, norm="natural",
                      step_size=1, step_ids=None):
    """S(τ) = β Σ_t[f'(u_t) − C_Ω(s_t)] over completion tokens. Differentiable in lg_pol.
    lg_pol, lg_ref : [T, V] raw logits.  ids : [T].  comp : [T] bool (True on response tokens).
    kln=True applies the KL-normalization (f'(1)=1) to the f-divergence (not euc).

    STEP-LEVEL (either arg set): group the completion tokens into steps and apply f'/Φ to the per-step
    ratio u_k = Π_{t∈step} u_t = exp(Σ log u_t) — i.e. S = β Σ_k f(u_k)/u_k (single-sample).
      · step_size>1  : FIXED-size steps of step_size tokens (mechanism ablation, no confound).
      · step_ids     : [n_completion] long, an explicit token→step map (e.g. newline segmentation,
                       Step-DPO-style variable-length reasoning steps). Overrides step_size.
    Exact over the sentence action space is intractable, so step-level is single-sample only (no euc).
    token-level (step_size=1, step_ids=None) is the default; step→∞ approaches sequence-level. RKL is
    invariant to the segmentation (f/u=ln u is additive: Σ_k ln u_k = Σ_t log u_t = standard DPO)."""
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

    if step_ids is not None or step_size > 1:         # STEP-LEVEL: aggregate completion tokens into steps
        if key == "euc" or inner not in ("sample", "trl"):
            raise ValueError("step-level supports --inner sample|trl, f-divergences only "
                             "(the exact inner term over the sentence action space is intractable)")
        lu = log_u[comp[1:].bool()]                   # completion log-ratios, in response order
        n = lu.shape[0]
        if n == 0:
            return lg_pol.new_zeros(())
        if step_ids is not None:                      # explicit token→step map (newline / variable steps)
            sid = step_ids.to(lu.device)
        else:                                         # fixed-size steps: tokens 0..n-1 → step index // step_size
            sid = torch.arange(n, device=lu.device) // step_size
        log_u_k = lu.new_zeros(int(sid.max()) + 1).index_add_(0, sid, lu)   # [K]: Σ_{t∈step} log u_t (differentiable)
        if clamp is not None:
            log_u_k = log_u_k.clamp(-clamp, clamp)
        chosen = fprime_from_logu(key, log_u_k, adiv_a, norm=norm)
        if inner == "trl":                            # TRL/sequence estimator applied per step: chosen f'(u_k), NO inner term
            return beta * chosen.sum()                # β Σ_k f'(u_k)  (drops Φ; sequence-TRL's inner term cancels only at K=1)
        inner_c = phi_from_logu(key, log_u_k, adiv_a, norm=norm)
        return beta * (chosen - inner_c).sum()        # β Σ_k f(u_k)/u_k

    if clamp is not None:
        log_u = log_u.clamp(-clamp, clamp)            # heavy-tail guard (§4) — identical for every Ω

    if key == "euc":                                  # Bregman: chosen term = [∇Ω]_a = π_θ(a) − π_ref(a)
        p_y_pol = logp_y_pol.exp()
        chosen = p_y_pol - logp_y_ref.exp()           # d = π_θ(y) − π_ref(y)
        if inner == "trl":                            # TRL-style: chosen only, no inner term
            inner_c = torch.zeros_like(chosen)
        elif inner == "sample":                       # single-sample inner h = d − d²/(2π_θ) ⇒ S = β Σ d²/(2π_θ)
            inner_c = chosen - chosen * chosen / (2.0 * p_y_pol.clamp_min(1e-12))   # 1/π_θ tail (guard floor)
        else:
            inner_c = exact_C("euc", torch.log_softmax(lp, -1), torch.log_softmax(lr, -1),
                              adiv_a, dtype=lp.dtype)
    else:
        chosen = fprime_from_logu(key, log_u, adiv_a, norm=norm)
        if inner == "trl":                            # TRL/sequence estimator applied per token: chosen f'(u), NO inner term
            inner_c = torch.zeros_like(chosen)
        elif inner == "sample":
            inner_c = phi_from_logu(key, log_u, adiv_a, norm=norm)
        else:                                         # exact vocab sum (fp32 for training)
            inner_c = exact_C(key, torch.log_softmax(lp, -1), torch.log_softmax(lr, -1),
                              adiv_a, dtype=lp.dtype, norm=norm)
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

    # RKL is invariant to step_size (f/u = ln u is additive: Σ_k ln u_k = Σ_t log u_t = standard DPO)
    for ssz in (3, 5, 100):
        S = score_from_logits(lg_pol, lg_ref, ids, comp, "kl", beta, "sample", DEFAULT_ADIV_A, None, step_size=ssz).item()
        d = abs(S - dpo_ref)
        print(f"  RKL step_size={ssz:<3d}: |Δ| vs standard-DPO = {d:.2e}  " + ("OK" if d < 1e-9 else "FAIL"))
        ok &= d < 1e-9

    # …and to an ARBITRARY (variable-length, e.g. newline) segmentation via explicit step_ids
    ncomp = int(comp[1:].sum())
    raw = torch.tensor(np.sort(np.random.default_rng(7).integers(0, max(2, ncomp // 3), size=ncomp)))
    _, sid = torch.unique(raw, return_inverse=True)                   # contiguous 0..K-1 step ids
    S = score_from_logits(lg_pol, lg_ref, ids, comp, "kl", beta, "sample", DEFAULT_ADIV_A, None, step_ids=sid).item()
    d = abs(S - dpo_ref)
    print(f"  RKL newline(var-steps, K={int(sid.max())+1}): |Δ| vs standard-DPO = {d:.2e}  " + ("OK" if d < 1e-9 else "FAIL"))
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
# device for tensors/models; overridden to the per-rank accelerator device when launched under
# Accelerate+FSDP (multi-GPU full-FT for 4B/8B/14B). Default "cuda" keeps the single-GPU path identical.
_DEVICE = "cuda"


def load_model(name, train, place=True, ckpt=True):
    from transformers import AutoModelForCausalLM
    kw = {}
    try:                                    # flash-attention-2 (wheelhouse) — big speedup on long seqs
        import flash_attn  # noqa: F401
        kw["attn_implementation"] = "flash_attention_2"
    except Exception:
        pass
    try:
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, **kw)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, **kw)
    if place:                               # under FSDP the policy is placed/sharded by accelerator.prepare
        m.to(_DEVICE)
    if train:
        m.train()
        if ckpt:                            # off trades memory for ~25-30% speed; the math is unchanged
            m.gradient_checkpointing_enable()
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


def _newline_step_ids(tok, ids, comp):
    """Step-DPO-style newline segmentation → a token→step map over the completion tokens (length =
    comp.sum(), aligned to `log_u[comp[1:]]` in score_from_logits). A step ends at a run of newline
    tokens; the next content token opens the next step. A completion with no newline → one step (that
    response = sequence-level). Byte-level BPE encodes '\\n' as 'Ċ'; a decoded '\\n' is the fallback."""
    comp_ids = ids[comp].tolist()
    if not comp_ids:
        return torch.zeros(0, dtype=torch.long)
    pieces = tok.convert_ids_to_tokens(comp_ids)
    is_nl = [bool(p) and ("Ċ" in p or "\n" in p) for p in pieces]
    sid, k = [], 0
    for j in range(len(comp_ids)):
        sid.append(k)
        if is_nl[j] and (j + 1 >= len(comp_ids) or not is_nl[j + 1]):   # end of a newline run → new step
            k += 1
    return torch.tensor(sid, dtype=torch.long)


def encode_pair(tok, ex, max_len, kw, step_mode="token"):
    """One preference example {'chosen':[msgs], 'rejected':[msgs]} → ((ids_w,comp_w,sid_w),(ids_l,comp_l,sid_l)).
    sid_* is the newline token→step map when step_mode='newline', else None (token / fixed-size steps)."""
    c, r = ex.get("chosen"), ex.get("rejected")
    if not (isinstance(c, list) and c and isinstance(r, list) and r):
        return None
    (iw, cw), (il, cl) = _encode_side(tok, c, max_len, kw), _encode_side(tok, r, max_len, kw)
    if cw.sum() == 0 or cl.sum() == 0:
        return None
    sw = sl = None
    if step_mode == "newline":
        sw, sl = _newline_step_ids(tok, iw, cw), _newline_step_ids(tok, il, cl)
    return (iw, cw, sw), (il, cl, sl)


def _logits(model, ids):
    return model(ids.unsqueeze(0).to(_DEVICE)).logits[0].float()      # [T, V] fp32


class AdapterRef:
    """The reference model under LoRA. Calling it runs the policy with the adapter switched off, so
    the base weights serve as pi_ref and no second copy of the model is held. This is the whole
    memory argument for LoRA here: the replicated frozen reference is what forced 8B full
    fine-tuning onto four GPUs."""

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, ids):
        with self.policy.disable_adapter():
            return self.policy(ids)


def pair_scores(policy, ref, enc, key, beta, inner, adiv_a, clamp, norm="natural", step_size=1):
    (iw, cw, sw), (il, cl, sl) = enc
    with torch.no_grad():
        rw, rl = _logits(ref, iw), _logits(ref, il)
    sw = sw.to(_DEVICE) if sw is not None else None
    sl = sl.to(_DEVICE) if sl is not None else None
    Sw = score_from_logits(_logits(policy, iw), rw, iw.to(_DEVICE), cw.to(_DEVICE), key, beta, inner, adiv_a, clamp, norm, step_size, sw)
    Sl = score_from_logits(_logits(policy, il), rl, il.to(_DEVICE), cl.to(_DEVICE), key, beta, inner, adiv_a, clamp, norm, step_size, sl)
    return Sw, Sl


@torch.no_grad()
def evaluate(policy, ref, tok, ds, key, beta, inner, adiv_a, clamp, max_len, kw, n, norm="natural", step_size=1, step_mode="token"):
    policy.eval()
    acc = tot = 0
    margins = []
    for ex in ds[:n]:
        enc = encode_pair(tok, ex, max_len, kw, step_mode)
        if enc is None:
            continue
        Sw, Sl = pair_scores(policy, ref, enc, key, beta, inner, adiv_a, clamp, norm, step_size)
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
    ap.add_argument("--eval-data", default=None, help="held-out preference file for eval (e.g. the UF test_prefs split); "
                                                      "if unset, a slice of --data is held out instead")
    ap.add_argument("--div", default="kl", choices=KEYS)
    ap.add_argument("--inner", default="sample", choices=["sample", "exact", "trl"],
                    help="inner term estimator: sample (logged token/step Φ), exact (full-vocab, token only), "
                         "trl (drop the inner term — TRL/sequence-level estimator applied per step)")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=0, help="if >0, train this many passes over ds_train (overrides --steps)")
    ap.add_argument("--lr-schedule", default="constant", choices=["constant", "linear", "cosine"])
    ap.add_argument("--warmup-ratio", type=float, default=0.0, help="fraction of steps for LR warmup")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--adam-beta2", type=float, default=0.95, help="AdamW β2 (0.999 = standard DPO/TRL default)")
    ap.add_argument("--grad-accum", type=int, default=8, help="pairs per optimizer step (effective batch)")
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--adiv-a", type=float, default=DEFAULT_ADIV_A)
    ap.add_argument("--kln", action="store_true", help="legacy alias for --norm kln (KL-normalize, f'(1)=1). Not for euc.")
    ap.add_argument("--norm", default="natural", choices=["natural", "amari", "kln", "canon"],
                    help="generator normalization: natural | amari (f'(1)=0) | kln (f'(1)=1) | canon (f'(1)=f''(1)). Not for euc.")
    ap.add_argument("--step-mode", default="token", choices=["token", "fixed", "newline"],
                    help="token=token-level; fixed=--step-size tokens/step (ablation); newline=Step-DPO-style sentence steps")
    ap.add_argument("--step-size", type=int, default=1, help="tokens per step for --step-mode fixed (>1)")
    # ---- LoRA. Settings follow the alignment-handbook Zephyr DPO QLoRA recipe (r=alpha=128,
    # dropout 0.05, adapters on all seven linear projections, lr 5e-6, max_length 1024), which is the
    # closest published DPO configuration to ours since it also trains on UltraFeedback in bf16 with
    # flash-attention-2. Targeting every linear layer rather than q/v only follows QLoRA (Dettmers
    # et al. 2023). With an adapter the reference model is not loaded at all: disabling the adapter
    # restores the base model, which is what TRL does ("the reference model is not needed since the
    # adapter can be disabled to revert to the initial model").
    ap.add_argument("--save-adapter-every", type=int, default=0,
                    help="LoRA only: every N steps, write the ADAPTER (~1.4G fp32, not the 16G merged model) to "
                         "{out}_adapter. Insurance for multi-day runs: the merged policy is written only "
                         "after the final step, so without this a walltime overrun or node failure loses "
                         "everything. Merge a rescued adapter with merge_adapter.py.")
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="disable gradient checkpointing (recompute). Numerically identical, ~25-30%% faster, "
                         "much more activation memory — worth it under LoRA on one 80G H100, where the frozen "
                         "base leaves most of the card unused.")
    ap.add_argument("--lora-r", type=int, default=0, help="LoRA rank, 0 disables LoRA (full fine-tuning)")
    ap.add_argument("--lora-alpha", type=int, default=0, help="LoRA alpha, defaults to --lora-r")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--clamp", type=float, default=15.0, help="clamp |log u| (heavy-tail guard, §4); 0 disables")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm (raise to relax the aggressive default)")
    ap.add_argument("--init-noise", type=float, default=0.0,
                    help="one-time ε perturbation of the policy init (fraction of each weight tensor's std) so u_init≠1 "
                         "when π_θ=π_ref (standard-DPO init); breaks the f'(1)=0 freeze of non-admissible single-sample")
    ap.add_argument("--grad-noise", type=float, default=0.0,
                    help="SGLD-style decaying gradient noise σ_t=grad_noise/(1+t)^0.55 (saddle-escape at u=1); tune small")
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--eval-n", type=int, default=128)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=0,
                    help="run held-out eval every N steps (0 = every --log-every); set larger to log loss/|g| often but eval rarely")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/stageB")
    ap.add_argument("--save-policy", action="store_true", help="save the trained policy at the end")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(selftest())

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    clamp = args.clamp if args.clamp and args.clamp > 0 else None
    norm = "kln" if args.kln else args.norm       # --kln is a legacy alias; --norm wins otherwise
    if norm != "natural" and args.div == "euc":
        raise SystemExit("euc is excluded from normalization (Bregman, no f(u) generator) — use --norm natural or --div ≠ euc")
    if args.step_mode == "token" and args.step_size > 1:   # --step-size>1 alone ⇒ fixed-size steps (back-compat)
        args.step_mode = "fixed"
    if args.step_mode == "fixed" and args.step_size <= 1:
        raise SystemExit("--step-mode fixed needs --step-size > 1")
    if args.step_mode != "token" and (args.div == "euc" or args.inner == "exact"):
        raise SystemExit("step-level (--step-mode fixed/newline) needs --inner sample and not euc "
                         "(the exact inner term over the sentence action space is intractable)")

    # ── optional multi-GPU full-FT via Accelerate + FSDP (4B/8B/14B). Activates only when launched
    #    distributed (WORLD_SIZE>1, e.g. `accelerate launch --config_file fsdp.yaml`). The single-GPU
    #    path (accel is None) is byte-identical to before. ─────────────────────────────────────────
    global _DEVICE
    accel = None
    world, rank, is_main = 1, 0, True
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from accelerate import Accelerator
        accel = Accelerator()                     # FSDP + bf16 come from the accelerate config at launch
        _DEVICE = accel.device
        world, rank, is_main = accel.num_processes, accel.process_index, accel.is_main_process
        accel.print(f"[FSDP] world={world}  device={_DEVICE}  (grad_accum {args.grad_accum} -> "
                    f"{max(1, args.grad_accum // world)}/rank)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.policy or args.ref)
    if getattr(tok, "chat_template", None) is None:
        # Qwen3-*-Base ships a chat template; Llama-3.2-* and gemma-*-pt do not, and encode_pair needs
        # one to find the prompt/response boundary it masks on. Follow the convention the DPO codebase
        # uses for base checkpoints (Rafailov et al. 2023, and the f-DPO fork of it): the prompt is
        # "\n\nHuman: ... \n\nAssistant:" and the response is terminated with the EOS token, which that
        # code appends explicitly (chosen_tokens['input_ids'].append(tokenizer.eos_token_id)).
        #
        # The EOS is the part that matters. An earlier version of this fallback emitted "role: content"
        # with no terminator, so nothing in training ever taught the policy to stop. gen_bench.py halts
        # on eos_token_id, the policy never produced one, and 41% of Llama answers ran to the 1024-token
        # cap while re-emitting the turn markers. That is an artificial failure to terminate, injected
        # into the exact axis this paper measures, so it invalidated those runs.
        tok.chat_template = (
            "{% for m in messages %}"
            "{% if m['role'] == 'user' %}{{ '\n\nHuman: ' + m['content'] }}"
            "{% elif m['role'] == 'assistant' %}{{ '\n\nAssistant: ' + m['content'] + eos_token }}"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ '\n\nAssistant:' }}{% endif %}")
        print(f"[tok] {args.policy or args.ref}: no chat_template -> installed the DPO-style fallback "
              f"(Human/Assistant turns, response terminated with {tok.eos_token!r})", flush=True)
    kw = {}
    try:                                            # Qwen3: suppress the <think> block
        tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, enable_thinking=False)
        kw = {"enable_thinking": False}
    except (TypeError, ValueError):                 # kwarg unknown, or template does not accept it
        pass

    ckpt = not args.no_grad_checkpoint
    policy = load_model(args.policy or args.ref, train=True, place=(accel is None), ckpt=ckpt)
    if args.lora_r > 0:
        from peft import LoraConfig, get_peft_model
        if ckpt:
            policy.enable_input_require_grads()             # gradient checkpointing + frozen base: without this no
                                                        # checkpointed block input requires grad and backward fails
        policy = get_peft_model(policy, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha or args.lora_r, lora_dropout=args.lora_dropout,
            target_modules=[m for m in args.lora_target.split(",") if m], bias="none", task_type="CAUSAL_LM"))
        ref = AdapterRef(policy)                        # π_ref = the base weights, reached by disabling the adapter
        if is_main:
            policy.print_trainable_parameters()
    else:
        ref = load_model(args.ref, train=False)                   # frozen reference: replicated on each rank

    if args.init_noise > 0:                            # break the u=1 degeneracy of the standard-DPO init (π_θ=π_ref):
        torch.manual_seed(args.seed + 1)               # one-time ε weight perturbation so u_init≠1 (else non-admissible
        with torch.no_grad():                          # single-sample freezes, g'(1)=f'(1)=0). RKL is insensitive to it (control).
            # Under LoRA the perturbation MUST land on lora_B (zero-initialised, hence u_init=1 exactly) and not on
            # the base weights, which are π_ref itself here — perturbing them would move the reference too.
            tgt = [(n, p) for n, p in policy.named_parameters() if "lora_B" in n] if args.lora_r > 0 \
                else [(n, p) for n, p in policy.named_parameters()]
            for n, p in tgt:                           # (before FSDP shard: full model, identical on every rank)
                if p.dim() >= 2:                       # perturb weight matrices only (not norms/biases)
                    sd = p.float().std()
                    if not torch.isfinite(sd) or sd == 0:   # lora_B starts at exactly 0 -> std 0; seed off lora_A's scale
                        sd = torch.tensor(1.0 / max(args.lora_r, 1))
                    p.add_(torch.randn_like(p) * (args.init_noise * sd))
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, args.adam_beta2), weight_decay=args.weight_decay)
    if accel is not None:
        policy, opt = accel.prepare(policy, opt)       # FSDP-shard the policy + its optimizer states

    ds = [json.loads(l) for l in open(args.data) if l.strip()]
    rng = np.random.default_rng(args.seed); rng.shuffle(ds)
    if args.eval_data:                                # dedicated held-out split (e.g. UF test_prefs) — no train/eval overlap
        ds_train = ds
        ds_eval = [json.loads(l) for l in open(args.eval_data) if l.strip()]
    else:                                             # fallback: hold out a slice of --data
        n_eval = max(args.eval_n, int(len(ds) * args.eval_frac))
        ds_eval, ds_train = ds[:n_eval], ds[n_eval:]

    if args.epochs > 0:                               # epochs override --steps (== a full pass count over ds_train)
        args.steps = max(1, len(ds_train) * args.epochs // args.grad_accum)
    sched = None
    if args.lr_schedule != "constant":                # linear/cosine decay with warmup (match the paper's DPO recipe)
        from transformers import get_scheduler
        sched = get_scheduler(args.lr_schedule, opt,
                              num_warmup_steps=int(args.warmup_ratio * args.steps),
                              num_training_steps=args.steps)

    def train_iter():
        while True:
            order = rng.permutation(len(ds_train))     # same permutation on every rank (shared rng seed)
            for j in order[rank::world]:               # each rank consumes a disjoint stride -> data-parallel
                yield ds_train[j]
    it = train_iter()
    accum_local = max(1, args.grad_accum // world)     # per-rank micro-steps; FSDP averages across ranks
                                                       # so global effective batch stays ~grad_accum

    hist = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()          # measure the TRAINING peak (exclude load transients)
    t0 = time.time()
    seg = f"newline" if args.step_mode == "newline" else (f"fixed×{args.step_size}" if args.step_mode == "fixed" else "token")
    is_main and print(f"=== Stage B: Ω={SHORT[args.div]} (key {args.div}) inner={args.inner} norm={norm} step={seg} "
          f"beta={args.beta} lr={args.lr}({args.lr_schedule},wu{args.warmup_ratio}) wd={args.weight_decay} "
          f"β2={args.adam_beta2} steps={args.steps}{f'(={args.epochs}ep)' if args.epochs else ''} accum={args.grad_accum} "
          f"clip={args.grad_clip} | train={len(ds_train)} eval={len(ds_eval)}"
          f"{' ('+args.eval_data+')' if args.eval_data else ' (train-slice)'} ===")
    if args.init_noise or args.grad_noise:
        print(f"    (u=1 freeze-escape: init_noise={args.init_noise} grad_noise={args.grad_noise})")
    if args.step_mode == "newline":                   # sanity: confirm the tokenizer's newline splitting fired
        e0 = encode_pair(tok, ds_train[0], args.max_len, kw, "newline")
        if e0 is not None:
            (_, cw, sw), (_, cl, sl) = e0
            nsw = int(sw.max()) + 1 if sw is not None and len(sw) else 0
            print(f"  [newline] example 0: chosen {nsw} steps / {int(cw.sum())} tok  "
                  f"(≈1 step ⇒ '\\n' not detected — check tokenizer)")
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        losses, accs = [], []
        got = 0
        while got < accum_local:
            enc = encode_pair(tok, next(it), args.max_len, kw, args.step_mode)
            if enc is None:
                continue
            Sw, Sl = pair_scores(policy, ref, enc, args.div, args.beta, args.inner, args.adiv_a, clamp, norm, args.step_size)
            loss = -F.logsigmoid(Sw - Sl)
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at step {step} — skipping this pair"); continue
            if accel is not None:                       # /accum_local per rank; FSDP mean-reduce over ranks
                accel.backward(loss / accum_local)      #   -> global grad = mean over the ~grad_accum batch
            else:
                (loss / args.grad_accum).backward()
            losses.append(loss.item()); accs.append(int(Sw.item() > Sl.item())); got += 1
        if args.grad_noise > 0:                       # SGLD-style decaying gradient noise: kicks θ off the u=1 critical point
            sigma = args.grad_noise / (1.0 + step) ** 0.55
            for p in policy.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p.grad) * sigma)
        if accel is not None:
            gnorm = accel.clip_grad_norm_(policy.parameters(), args.grad_clip).item()
        else:
            gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip).item()
        opt.step()
        if sched is not None:
            sched.step()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            rec = {"step": step, "loss": float(np.mean(losses)), "train_acc": float(np.mean(accs)),
                   "grad_norm": gnorm, "sec": round(time.time() - t0, 1)}
            if torch.cuda.is_available():             # cost of the inner-term arm: peak GPU memory
                rec["gpu_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                rec["gpu_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
            eval_every = args.eval_every if args.eval_every > 0 else args.log_every
            if step % eval_every == 0 or step == 1 or step == args.steps:   # eval is the expensive part — decoupled from logging
                rec.update(evaluate(policy, ref, tok, ds_eval, args.div, args.beta, args.inner,   # FSDP: ALL ranks (collective forward)
                                    args.adiv_a, clamp, args.max_len, kw, args.eval_n, norm, args.step_size, args.step_mode))
            if is_main:                                   # only main logs/saves; every rank still ran eval above
                hist.append(rec)
                ev = f"eval_acc {rec['eval_acc']:.3f} margin {rec['eval_margin']:+.3f}  " if "eval_acc" in rec else ""
                print(f"  step {step:4d}  loss {rec['loss']:.4f}  train_acc {rec['train_acc']:.3f}  "
                      f"{ev}|g| {gnorm:.2f}  {rec['sec']:.0f}s  mem {rec.get('gpu_reserved_gb', '?')}G")
                with open(args.out + ".json", "w") as f:
                    json.dump({"args": vars(args), "history": hist}, f, indent=2)

        if args.lora_r > 0 and args.save_adapter_every > 0 and step % args.save_adapter_every == 0 and is_main:
            # Adapter only: ~0.7G and, unlike merge_and_unload(), it does NOT consume the PeftModel,
            # so training continues untouched. Overwrites in place -> one directory, not one per step.
            policy.save_pretrained(args.out + "_adapter")
            tok.save_pretrained(args.out + "_adapter")   # carry the RUNTIME chat_template with it: a base
            # model may have none and we install a fallback, so merging against a fresh base tokenizer
            # would silently lose the template the policy was trained against.
            print(f"  [ckpt] adapter -> {args.out}_adapter (step {step})")

    if args.save_policy:
        if accel is not None:                              # FSDP: gather the full (unsharded) state dict, save on main
            accel.wait_for_everyone()
            state = accel.get_state_dict(policy)
            unwrapped = accel.unwrap_model(policy)
            if is_main:
                unwrapped.save_pretrained(args.out + "_policy", state_dict=state, safe_serialization=True)
                tok.save_pretrained(args.out + "_policy")
        else:
            to_save = policy.merge_and_unload() if args.lora_r > 0 else policy   # LoRA: fold BA into the base so
            to_save.save_pretrained(args.out + "_policy")                         # gen_*.slrm needs no change
            tok.save_pretrained(args.out + "_policy")
        if is_main:
            print("saved policy ->", args.out + "_policy")
    if is_main:
        print("done ->", args.out + ".json")


if __name__ == "__main__":
    main()
