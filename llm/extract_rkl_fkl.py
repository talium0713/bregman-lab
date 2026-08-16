"""Align kl_bdpo (RKL) vs rkl_bdpo (FKL) Arena v0.1 answers by uid, write a readable side-by-side MD,
and print a couple of short examples to stdout."""
import json, os, sys

D = "results/bench/arena_v01"
def load(tag):
    m = {}
    for l in open(f"{D}/{tag}.jsonl"):
        if l.strip():
            r = json.loads(l)
            a = r["messages"][-1]["content"]; a = a.get("answer","") if isinstance(a,dict) else a
            m[r["uid"]] = (r["messages"][0]["content"], a, r.get("metadata",{}).get("token_len",0))
    return m

rkl = load("kl_bdpo")     # RKL
fkl = load("rkl_bdpo")    # FKL
uids = [u for u in rkl if u in fkl]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12

os.makedirs(f"{D}/arena_readable", exist_ok=True)
out = f"{D}/arena_readable/RKL_vs_FKL.md"
with open(out, "w") as f:
    f.write("# RKL (kl_bdpo, WR 14.8) vs FKL (rkl_bdpo, WR 6.8) — Arena-Hard v0.1, base->f-DPO\n\n")
    for i, u in enumerate(uids[:N], 1):
        q, ar, tr = rkl[u]; _, af, tf = fkl[u]
        f.write(f"---\n\n## {i}. `{u}`\n\n**Q:**\n\n{q}\n\n")
        f.write(f"**RKL ({tr} tok):**\n\n{ar}\n\n**FKL ({tf} tok):**\n\n{af}\n\n")
print(f"wrote {out}  ({min(N,len(uids))} pairs, {os.path.getsize(out)/1e3:.0f} KB)")

# print 2 short examples inline (pick the shortest-combined for readability)
uids.sort(key=lambda u: rkl[u][2] + fkl[u][2])
for u in uids[:2]:
    q, ar, tr = rkl[u]; _, af, tf = fkl[u]
    print("\n" + "="*90)
    print("Q:", q[:300].replace("\n", " "))
    print(f"\n--- RKL ({tr} tok) ---\n{ar[:700]}")
    print(f"\n--- FKL ({tf} tok) ---\n{af[:700]}")
