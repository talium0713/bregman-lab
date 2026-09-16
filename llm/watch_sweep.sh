#!/bin/bash
# Detached driver: wait for the base->f-DPO RePO-match gen array (genbfdpo) -> stats -> judge the 6 new
# {div}_bdpo answers (Base_repo/basedpo_repo/etc. are cached) -> write leaderboard to logs/sweep_RESULT.txt
# and touch logs/sweep_DONE. setsid-launched so it survives ssh drops; poll logs/sweep_RESULT.txt.
cd ~/scratch/bregman-lab/llm
for i in $(seq 1 120); do
  squeue --me -h -n genbfdpo 2>/dev/null | grep -q . || break
  sleep 30
done
{
  echo "=== base->f-DPO gen stats (RePO-match) ==="
  python3 - <<'PY'
import json, statistics as st, os
def ans(r):
    a=r["messages"][-1]["content"]; return a.get("answer","") if isinstance(a,dict) else a
def looped(s):
    s=s[-400:]; return len(s)>200 and len(set(s[i:i+8] for i in range(0,len(s)-8,8)))<=6
for d in ["kl","rkl","adiv","js","hel","chi2"]:
    p="results/bench/arena_v01/%s_bdpo.jsonl"%d
    if not os.path.exists(p): print("  %s_bdpo MISSING"%d); continue
    rows=[json.loads(l) for l in open(p) if l.strip()]
    tls=[r.get("metadata",{}).get("token_len",0) for r in rows]
    print("  %-10s n=%d med_tok=%.0f loop=%d"%(d+"_bdpo",len(rows),st.median(tls),sum(looped(ans(r)) for r in rows)))
PY
} > logs/sweep_RESULT.txt 2>&1

miss=0
for d in kl rkl adiv js hel chi2; do [ -f results/bench/arena_v01/${d}_bdpo.jsonl ] || miss=1; done
if [ "$miss" = 1 ]; then echo "ABORT: some gen outputs missing" >> logs/sweep_RESULT.txt; touch logs/sweep_DONE; exit 1; fi

JUDGE=gpt-4.1-2025-04-14 bash run_arena_v01_judge.sh > logs/judge_sweep.log 2>&1
echo "" >> logs/sweep_RESULT.txt
echo "=== Arena v0.1 leaderboard (judge gpt-4.1-2025-04-14, baseline gpt-4-0314) ===" >> logs/sweep_RESULT.txt
grep -aA14 "Category: arena-hard-v0.1" logs/judge_sweep.log | tail -14 >> logs/sweep_RESULT.txt
grep -a "null judgments" logs/judge_sweep.log | tail -1 >> logs/sweep_RESULT.txt
touch logs/sweep_DONE
