"""pairwise_h2h.py — direct head-to-head between two of our Arena-Hard v0.1 answer sets, reusing arena-hard's
EXACT judge (OG_ARENA_HARD_PROMPT system prompt + prompt_template + [[A>>B]] regex + gpt-4.1-2025-04-14,
temp 0). Two games per prompt with position swap (A/B), then A's win rate vs B = (wins + 0.5*ties)/games,
with a prompt-level bootstrap CI. Does NOT touch the arena repo.

    python pairwise_h2h.py --a kl_bdpo --b rkl_bdpo --out logs/h2h_rkl_fkl.json
"""
import json, os, re, argparse, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

SYSTEM = (
"Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants "
"to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your "
"job is to evaluate which assistant's answer is better.\n\n"
"Begin your evaluation by generating your own answer to the prompt. You must provide your answers before "
"judging any answers.\n\n"
"When evaluating the assistants' answers, compare both assistants' answers with your answer. You must "
"identify and correct any mistakes or inaccurate information.\n\n"
"Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer "
"correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or "
"more than one interpretation, it is more helpful and appropriate to ask for clarifications or more "
"information from the user than providing an answer based on assumptions. Relevant means all parts of the "
"response closely connect or are appropriate to what is being asked. Concise means the response is clear "
"and not verbose or excessive.\n\n"
"Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any "
"missing important information in the assistants' answers that would be beneficial to include when "
"responding to the user prompt.\n\n"
"After providing your explanation, you must output only one of the following choices as your final verdict "
"with a label:\n\n"
"1. Assistant A is significantly better: [[A>>B]]\n"
"2. Assistant A is slightly better: [[A>B]]\n"
"3. Tie, relatively the same: [[A=B]]\n"
"4. Assistant B is slightly better: [[B>A]]\n"
"5. Assistant B is significantly better: [[B>>A]]\n\n"
'Example output: "My final verdict is tie: [[A=B]]".'
)
TEMPLATE = ("<|User Prompt|>\n{QUESTION}\n\n\n<|The Start of Assistant A's Answer|>\n\n{ANSWER_A}\n\n"
            "<|The End of Assistant A's Answer|>\n\n\n<|The Start of Assistant B's Answer|>\n\n{ANSWER_B}\n\n"
            "<|The End of Assistant B's Answer|>")
PATTERNS = [re.compile(r"\[\[([AB<>=]+)\]\]"), re.compile(r"\[([AB<>=]+)\]")]

def get_verdict(text):
    for p in PATTERNS:
        m = [x for x in p.findall(text.upper()) if x]
        if m:
            return m[-1].strip()
    return None

def winner(v):
    """'A'/'B'/'tie'/None from a verdict token like A>>B, A>B, A=B, B>A."""
    if v is None:
        return None
    v = v.replace(">>", ">").replace("<<", "<")
    if "=" in v:
        return "tie"
    if ">" in v:
        return v.split(">")[0].strip()[:1]      # left side is better
    if "<" in v:
        return v.split("<")[-1].strip()[:1]     # right side is better
    return None

def load(tag):
    m = {}
    for l in open(f"results/bench/arena_v01/{tag}.jsonl"):
        if l.strip():
            r = json.loads(l); a = r["messages"][-1]["content"]
            a = a.get("answer", "") if isinstance(a, dict) else a
            m[r["uid"]] = (r["messages"][0]["content"], a)
    return m

_key = os.environ.get("OPENAI_API_KEY") or ""
if len(_key) < 40 and os.path.exists("openai_key.txt"):   # robust: read the key file directly (no fragile tr)
    _key = open("openai_key.txt").read().strip()
client = OpenAI(api_key=_key)
_first_err = [True]           # surface the first API error (don't silently swallow 401/quota for 40 min)
def call(q, ansA, ansB, tries=4):
    user = TEMPLATE.format(QUESTION=q, ANSWER_A=ansA, ANSWER_B=ansB)
    for t in range(tries):
        try:
            r = client.chat.completions.create(
                model="gpt-4.1-2025-04-14",
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                temperature=0.0, max_tokens=4096)
            return get_verdict(r.choices[0].message.content or "")
        except Exception as e:
            if _first_err[0]:
                print(f"  [API ERROR] {type(e).__name__}: {str(e)[:200]}", flush=True)
                _first_err[0] = False
            if t == tries - 1:
                return None
            time.sleep(3 * (t + 1))

def game(uid, q, ans_a, ans_b, a_pos):
    """a_pos: 'A' means model-a is Assistant A this game. Returns a's outcome 1/0/0.5/None."""
    if a_pos == "A":
        w = winner(call(q, ans_a, ans_b))
    else:
        w = winner(call(q, ans_b, ans_a))
    if w is None:
        return None
    if w == "tie":
        return 0.5
    return 1.0 if w == a_pos else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="model-a tag (the one whose win rate we report)")
    ap.add_argument("--b", required=True, help="model-b tag (opponent)")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--out", default="logs/h2h.json")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    uids = [u for u in A if u in B]
    jobs = []  # (uid, a_pos)
    for u in uids:
        jobs += [(u, "A"), (u, "B")]

    # preflight: one real call to fail fast on a bad key / API issue instead of running ~40 min of nulls
    u0 = uids[0]
    if game(u0, A[u0][0], A[u0][1], B[u0][1], "A") is None:
        print("PREFLIGHT FAILED — aborting before the full run (see [API ERROR] above). Fix the key/API and re-run.", flush=True)
        return

    results = {}  # uid -> list of outcomes
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        fut = {ex.submit(game, u, A[u][0], A[u][1], B[u][1], pos): (u, pos) for u, pos in jobs}
        for f in as_completed(fut):
            u, pos = fut[f]
            results.setdefault(u, []).append(f.result())
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    # flatten valid game outcomes
    per_prompt = {u: [o for o in outs if o is not None] for u, outs in results.items()}
    flat = [o for outs in per_prompt.values() for o in outs]
    n_games = len(flat); nulls = len(jobs) - n_games
    wins = sum(1 for o in flat if o == 1.0); ties = sum(1 for o in flat if o == 0.5); losses = sum(1 for o in flat if o == 0.0)
    wr = (wins + 0.5 * ties) / n_games if n_games else float("nan")

    # prompt-level bootstrap (deterministic: LCG, no numpy/random-seed dependence issues)
    us = list(per_prompt.keys())
    seed = 12345; boot = []
    for _ in range(1000):
        s = 0.0; c = 0
        for _ in range(len(us)):
            seed = (1103515245 * seed + 12345) & 0x7fffffff
            outs = per_prompt[us[seed % len(us)]]
            s += sum(outs); c += len(outs)
        boot.append(s / c if c else 0.0)
    boot.sort()
    lo, hi = boot[24], boot[975]

    out = {"a": args.a, "b": args.b, "n_prompts": len(uids), "n_games": n_games, "nulls": nulls,
           "wins": wins, "ties": ties, "losses": losses,
           "win_rate": round(wr * 100, 1), "ci_lo": round(lo * 100, 1), "ci_hi": round(hi * 100, 1)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n=== {args.a} vs {args.b} (head-to-head, arena judge gpt-4.1-2025-04-14, 2 games/prompt) ===")
    print(f"{args.a} win rate vs {args.b}: {out['win_rate']}%  (95% CI {out['ci_lo']}-{out['ci_hi']})")
    print(f"  games={n_games} (nulls {nulls}) | {args.a} wins {wins} · ties {ties} · losses {losses}")


if __name__ == "__main__":
    main()
