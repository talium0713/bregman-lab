"""
arena_to_md.py — turn an Arena-Hard v2 model-answer JSONL into a readable Markdown of Q/A pairs,
grouped by category/subcategory. Optional --questions file (arena_hard_v2.jsonl) adds category tags.

    python3 arena_to_md.py --answers results/bench/arena/RKL-token.jsonl \
        --questions data/arena_hard_v2.jsonl --out results/bench/arena_readable/RKL-token.md
"""
import argparse, json, os


def load_cat(qf):
    m = {}
    if qf and os.path.exists(qf):
        for l in open(qf):
            if l.strip():
                r = json.loads(l)
                m[r["uid"]] = (r.get("category", ""), r.get("subcategory", ""), r.get("language", ""))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--questions", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.answers) if l.strip()]
    cat = load_cat(args.questions)
    rows.sort(key=lambda r: cat.get(r["uid"], ("", "", "")))
    if args.max > 0:
        rows = rows[:args.max]
    model = rows[0].get("model", os.path.basename(args.answers))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"# Arena-Hard v2 answers — {model}\n\n{len(rows)} questions"
                f"{' (grouped by category)' if cat else ''}\n\n")
        for i, r in enumerate(rows, 1):
            c, sc, lang = cat.get(r["uid"], ("", "", ""))
            q = r["messages"][0]["content"]
            a = r["messages"][-1]["content"]
            a = a.get("answer", "") if isinstance(a, dict) else a
            tl = r.get("metadata", {}).get("token_len", "")
            tag = f"[{c}/{sc}{'/' + lang if lang and lang != 'English' else ''}]" if c else ""
            f.write(f"---\n\n## {i}. {tag} `{r['uid']}`  ·  {tl} tok\n\n")
            f.write(f"**Q:**\n\n{q}\n\n**A — {model}:**\n\n{a}\n\n")
    print(f"wrote {args.out}  ({len(rows)} Q/A, {os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
