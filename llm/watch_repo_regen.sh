#!/bin/bash
# Detached watcher: wait until the RePO-match regen (genarenav01repo) leaves the queue, then dump answer
# stats to logs/repo_regen_stats.txt and touch logs/repo_regen_DONE. Launched with setsid so it holds no
# ssh channel (avoids the long-hold Exit-255). Poll from outside with: cat logs/repo_regen_stats.txt
cd ~/scratch/bregman-lab/llm
for i in $(seq 1 120); do
  squeue --me -h -n genarenav01repo 2>/dev/null | grep -q . || break
  sleep 60
done
python3 - > logs/repo_regen_stats.txt 2>&1 <<'PY'
import json, statistics as st, os
def ans(r):
    a=r["messages"][-1]["content"]; return a.get("answer","") if isinstance(a,dict) else a
def looped(s):
    s=s[-400:]; return len(s)>200 and len(set(s[i:i+8] for i in range(0,len(s)-8,8)))<=6
print("%-26s %5s %7s %7s %9s %6s"%("file","n","mean","med",">=1000","loop"))
for f in ["Base_repo","sft_base_repo","sft_base_fix","Qwen3-1.7B-Base","sft_base"]:
    p="results/bench/arena_v01/%s.jsonl"%f
    q="results/bench/arena_v01/_skip/%s.jsonl"%f
    p = p if os.path.exists(p) else q
    if not os.path.exists(p): print("%-26s MISSING"%f); continue
    rows=[json.loads(l) for l in open(p) if l.strip()]
    tls=[r.get("metadata",{}).get("token_len",0) for r in rows]
    print("%-26s %5d %7.0f %7.0f %6d/%d %6d"%(f,len(rows),st.mean(tls),st.median(tls),sum(t>=1000 for t in tls),len(rows),sum(looped(ans(r)) for r in rows)))
PY
touch logs/repo_regen_DONE
