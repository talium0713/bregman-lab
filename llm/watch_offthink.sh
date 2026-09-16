#!/bin/bash
# Detached driver: wait for the train-consistent DPO gen (genbasedpooff) -> eyeball stats -> judge
# basedpo_offthink (Base_repo/sft_base_repo/basedpo_repo are cached, so only this one is judged) ->
# write the leaderboard to logs/offthink_RESULT.txt and touch logs/offthink_DONE. Launch with setsid so
# it survives ssh drops; poll logs/offthink_RESULT.txt from outside.
cd ~/scratch/bregman-lab/llm
for i in $(seq 1 60); do
  squeue --me -h -n genbasedpooff 2>/dev/null | grep -q . || break
  sleep 30
done
{
  echo "=== basedpo_offthink gen stats ==="
  python3 - <<'PY'
import json, statistics as st, os
p="results/bench/arena_v01/basedpo_offthink.jsonl"
if not os.path.exists(p):
    print("NO GEN OUTPUT"); raise SystemExit
rows=[json.loads(l) for l in open(p) if l.strip()]
def ans(r):
    a=r["messages"][-1]["content"]; return a.get("answer","") if isinstance(a,dict) else a
def looped(s):
    s=s[-400:]; return len(s)>200 and len(set(s[i:i+8] for i in range(0,len(s)-8,8)))<=6
tls=[r.get("metadata",{}).get("token_len",0) for r in rows]
print("basedpo_offthink n=%d med_tok=%.0f mean=%.0f loop=%d"%(len(rows),st.median(tls),st.mean(tls),sum(looped(ans(r)) for r in rows)))
PY
} > logs/offthink_RESULT.txt 2>&1

if [ ! -f results/bench/arena_v01/basedpo_offthink.jsonl ]; then
  echo "ABORT: no gen output" >> logs/offthink_RESULT.txt; touch logs/offthink_DONE; exit 1
fi

JUDGE=gpt-4.1-2025-04-14 bash run_arena_v01_judge.sh > logs/judge_offthink.log 2>&1
echo "" >> logs/offthink_RESULT.txt
echo "=== Arena v0.1 leaderboard (judge gpt-4.1-2025-04-14, baseline gpt-4-0314) ===" >> logs/offthink_RESULT.txt
grep -aA8 "Category: arena-hard-v0.1" logs/judge_offthink.log | tail -8 >> logs/offthink_RESULT.txt
grep -a "null judgments" logs/judge_offthink.log | tail -1 >> logs/offthink_RESULT.txt
touch logs/offthink_DONE
