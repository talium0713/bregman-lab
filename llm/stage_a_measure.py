"""
stage_a_measure.py — Stage A: measure the off-policy inner-term noise at LLM scale (NO training).

The LLM-scale version of `python/fig_toy_ablation.py`. For a realistic off-policy pair
(π_ref = base, π_θ = instruct) we score real preference responses and, at every response token y_t,
compute u = π_θ(y_t|s_t)/π_ref(y_t|s_t) and the single-logged-token inner-term estimate Φ(u) for all
7 divergences. Prediction (§1–3): RKL's Φ ≡ 1 (variance exactly 0), every other Φ spreads, and the
spread grows with how off-policy the token is (u→0). With --exact we also compute the full-vocab
closed form C_Ω and the single-sample-vs-exact gap.

Off-policy by construction: the response tokens come from the dataset (behaviour), and u compares
π_θ (instruct) to π_ref (base) — no resampling.

Data comes from a local JSONL (no `datasets`/pyarrow on the cluster — Alliance ships a dummy
pyarrow wheel). Prepare it once on a machine with normal internet via make_uf_jsonl.py, then scp.

Run (on a compute node, after prefetch + scp of the jsonl):
  python stage_a_measure.py --ref Qwen/Qwen3-1.7B-Base --policy Qwen/Qwen3-1.7B \
      --data data/uf_test_prefs.jsonl --max-samples 500 --max-len 1024 --exact \
      --out results/stageA_1p7b
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from divergences import KEYS, SHORT, COLORS, phi_from_logu, phi_euc, exact_C, DEFAULT_ADIV_A
# transformers is imported lazily (in load_model / main) so --replot works without it (e.g. on a laptop).


def load_model(name):
    from transformers import AutoModelForCausalLM
    try:                                    # transformers 5.x renamed torch_dtype -> dtype
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
    m.to("cuda").eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _ids(x):
    """apply_chat_template(tokenize=True) returns a list (old transformers) or a tokenizers.Encoding
    / BatchEncoding (transformers 5.x). Normalize to a flat list of ints."""
    if hasattr(x, "ids"):                           # tokenizers.Encoding
        return list(x.ids)
    if hasattr(x, "input_ids") or isinstance(x, dict):
        v = x["input_ids"]
        return list(v[0]) if v and isinstance(v[0], (list, tuple)) else list(v)
    return list(x)


def encode_chosen(tok, example, max_len):
    """Tokenize one preference example's `chosen` conversation; return (ids, completion_mask)."""
    msgs = example.get("chosen")
    if not isinstance(msgs, list) or not msgs:
        return None
    kw = {}
    try:                                            # Qwen3: suppress <think> block
        tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False); kw = {"enable_thinking": False}
    except TypeError:
        pass
    full = _ids(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, **kw))
    prompt = _ids(tok.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True, **kw))
    ids = torch.tensor(full[:max_len], dtype=torch.long)
    comp = torch.zeros(len(ids), dtype=torch.bool)
    comp[min(len(prompt), len(ids)):] = True        # mask = completion (response) tokens only
    return ids, comp


@torch.no_grad()
def token_logps(model, ids):
    """Per-position next-token log-prob at the realized token + full log-probs (fp32).
    Returns (logp_token[T-1], logits[T-1, V] in bf16). Uses the logsumexp trick for logp_token."""
    out = model(ids.unsqueeze(0).to("cuda")).logits[0].float()   # [T, V]
    logits = out[:-1]                                            # predict token t+1 from ≤t
    labels = ids[1:].to("cuda")
    logp_tok = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)
    return logp_tok, logits


def make_figure(summary, log_u, out):
    """(L) per-Ω inner-term noise: std (all tokens) + robust |Φ| 99th-pct.  (R) per-token log-ratio."""
    d = summary["divergences"]
    pol = summary["policy"].split("/")[-1]; rf = summary["ref"].split("/")[-1]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.8, 4.9))
    floor = 1e-4
    x = np.arange(len(KEYS)); w = 0.4
    stds = [max(d[k]["phi_std"], floor) for k in KEYS]
    q99 = [max(d[k].get("phi_abs_q99", floor), floor) for k in KEYS]
    axL.bar(x - w / 2, stds, w, color=[COLORS[k] for k in KEYS], label="std (all tokens)")
    axL.bar(x + w / 2, q99, w, color=[COLORS[k] for k in KEYS], alpha=0.4, hatch="//",
            edgecolor="white", label="|Φ| 99th pct (robust)")
    axL.set_xticks(x); axL.set_xticklabels([SHORT[k] for k in KEYS])
    axL.set_yscale("log"); axL.axhline(floor, color="0.6", ls=":", lw=1)
    axL.set_ylabel(r"single-token inner term  $|\Phi(u_{y_t})|$")
    axL.set_title(r"off-policy inner-term noise per Ω   (RKL: $\Phi\equiv1\Rightarrow$ std 0)")
    axL.legend(fontsize=8, framealpha=0.9)
    lg = (log_u / np.log(10)).astype(float)
    axR.hist(lg, bins=100, color="#4065E9", alpha=0.85)
    axR.set_yscale("log"); axR.axvline(0.0, color="0.5", ls="--", lw=1)
    axR.set_xlim(np.floor(lg.min()) - 0.5, np.ceil(lg.max()) + 0.5)
    axR.set_xlabel(r"$\log_{10}(\pi_\theta/\pi_{\mathrm{ref}})$ per logged token"
                   "\n(0 = agree · left = policy gives far less prob than ref)")
    axR.set_ylabel("token count (log scale)")
    axR.set_title(r"per-token $\pi_\theta$ vs $\pi_{\mathrm{ref}}$ gap  (min $u$ = "
                  + f"{summary['min_u']:.1e})")
    fig.suptitle(r"Stage A · $\pi_\theta$ (policy) = " + pol + r"   vs   $\pi_{\mathrm{ref}}$ (reference) = "
                 + rf + f"   ·   {summary['n_tokens']} tokens", fontsize=11, y=1.02)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
    print("wrote", out + ".png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--policy", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data", default="data/uf_test_prefs.jsonl",
                    help="JSONL, one obj/line with 'chosen': [messages]. Make it with make_uf_jsonl.py.")
    ap.add_argument("--max-samples", type=int, default=500)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--adiv-a", type=float, default=DEFAULT_ADIV_A)
    ap.add_argument("--exact", action="store_true", help="also compute full-vocab C_Ω + gap (heavier)")
    ap.add_argument("--out", default="results/stageA")
    ap.add_argument("--replot", default=None,
                    help="prefix of a prior run (<prefix>.json + <prefix>_raw.npz) — just redraw the figure, no models")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.replot:                          # redraw only — instant, no GPU/models/data
        with open(args.replot + ".json") as f:
            summary = json.load(f)
        log_u = np.load(args.replot + "_raw.npz")["log_u"]
        make_figure(summary, log_u, args.replot)
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.policy)
    ref, pol = load_model(args.ref), load_model(args.policy)
    with open(args.data) as f:
        ds = [json.loads(line) for line in f if line.strip()]

    log_u_all = []
    phi_all = {k: [] for k in KEYS}
    gap_all = {k: [] for k in KEYS} if args.exact else None
    n_tok = 0
    for i in range(min(args.max_samples, len(ds))):
        enc = encode_chosen(tok, ds[i], args.max_len)
        if enc is None:
            continue
        ids, comp = enc
        m = comp[1:]                                # align to next-token positions
        if m.sum() == 0:
            continue
        lp_pol, lg_pol = token_logps(pol, ids)
        lp_ref, lg_ref = token_logps(ref, ids)
        log_u = (lp_pol - lp_ref)[m]                # off-policy log-ratio at logged tokens
        log_u_all.append(log_u.cpu())
        for k in KEYS:
            if k == "euc":
                phi = phi_euc(torch.exp(lp_pol[m]), torch.exp(lp_ref[m]))
            else:
                phi = phi_from_logu(k, log_u, args.adiv_a)
            phi_all[k].append(phi.cpu())
        if args.exact:
            logpi = torch.log_softmax(lg_pol[m], -1)
            logrf = torch.log_softmax(lg_ref[m], -1)
            for k in KEYS:
                C = exact_C(k, logpi, logrf, args.adiv_a)          # [n_tok_i]
                phi_single = phi_all[k][-1].to(C.device)
                gap_all[k].append((phi_single - C).cpu())
        n_tok += int(m.sum())

    log_u = torch.cat(log_u_all).numpy()
    summary = {"ref": args.ref, "policy": args.policy, "data": args.data,
               "n_samples_used": len(log_u_all), "n_tokens": n_tok, "adiv_a": args.adiv_a,
               "log_u_quantiles": {q: float(np.quantile(log_u, q)) for q in (0.001, 0.01, 0.5, 0.99)},
               "min_u": float(np.exp(log_u.min())), "divergences": {}}
    for k in KEYS:
        phi = torch.cat(phi_all[k]).numpy()
        rec = {"phi_mean": float(np.mean(phi)), "phi_std": float(np.std(phi)),
               "phi_abs_q99": float(np.quantile(np.abs(phi), 0.99))}
        if args.exact:
            gap = torch.cat(gap_all[k]).numpy()
            rec["gap_std"] = float(np.std(gap)); rec["gap_abs_mean"] = float(np.mean(np.abs(gap)))
        summary["divergences"][k] = rec
    with open(args.out + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    np.savez_compressed(args.out + "_raw.npz", log_u=log_u)   # raw log-ratios -> re-plot via --replot
    make_figure(summary, log_u, args.out)
    print("wrote", args.out + ".json / .png / _raw.npz")


if __name__ == "__main__":
    main()
