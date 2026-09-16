#!/bin/bash
# Detached driver: RKL (kl_bdpo) direct head-to-head vs each remaining opponent (FKL already done).
# Sequential to avoid gpt-4.1 TPM rate-limit storms. Each writes logs/h2h_kl_vs_{opp}.json.
# setsid-launched → survives ssh drops; poll logs/h2h_all_DONE.
cd ~/scratch/bregman-lab/llm
source ~/arena_judge_env/bin/activate
rm -f logs/h2h_all_DONE
for B in js_bdpo hel_bdpo adiv_bdpo chi2_bdpo Base_repo; do
  echo "=== RKL vs $B  $(date) ===" >> logs/h2h_all.log
  python pairwise_h2h.py --a kl_bdpo --b "$B" --out "logs/h2h_kl_vs_${B}.json" >> logs/h2h_all.log 2>&1
done
touch logs/h2h_all_DONE
