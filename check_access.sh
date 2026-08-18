#!/bin/bash
# check_access.sh — verify you can read the SHARED bregman-lab backup (models + data) on Killarney.
# Run on a Killarney LOGIN node after cloning the repo:   bash check_access.sh
# It checks group membership, the traversal chain, and a real read of each key file, then tells you
# exactly what (if anything) to send to Taehyun. Read-only — it changes nothing.

SRC="$HOME/projects/aip-rudner/talium/bregman-backup"
[ -d "$SRC" ] || SRC="/project/6105494/talium/bregman-backup"   # fallback if the ~/projects symlink is absent

pass=0; fail=0; failed_paths=()
ok()   { echo "  [OK]   $1"; pass=$((pass+1)); }
bad()  { echo "  [FAIL] $1"; fail=$((fail+1)); [ -n "$2" ] && failed_paths+=("$2"); }

# dir: needs read (list) + execute (traverse).  file: needs a real 1-byte read.
chk_dir()  { if [ -r "$1" ] && [ -x "$1" ]; then ok "$2"; else bad "$2" "$1"; fi; }
chk_file() { if head -c1 "$1" >/dev/null 2>&1; then ok "$2"; else bad "$2" "$1"; fi; }

echo "=== Killarney shared-backup access check ==="
echo "user=$(whoami)   SRC=$SRC"
echo

echo "-- group membership --"
if id -Gn | tr ' ' '\n' | grep -qx aip-rudner; then
  ok "you are in group 'aip-rudner'"
else
  bad "you are NOT in group 'aip-rudner'  (ask the PI to add you — nothing below will work without it)"
fi

echo "-- traversal chain (each dir must be readable+traversable) --"
chk_dir "/project/6105494"                 "/project/6105494 (aip-rudner project space)"
chk_dir "/project/6105494/talium"          "  .../talium"
chk_dir "$SRC"                             "  .../bregman-backup"

echo "-- models (SFT bases, for the sft_base baseline) --"
chk_dir  "$SRC/models"                                        "models/"
chk_dir  "$SRC/models/sft_uc_1p7b_model"                      "models/sft_uc_1p7b_model/"
chk_file "$SRC/models/sft_uc_1p7b_model/config.json"          "read models/sft_uc_1p7b_model/config.json"

echo "-- data (UltraFeedback pairs + UltraChat SFT) --"
chk_dir  "$SRC/data"                              "data/"
for f in uf_pairs_train_full.jsonl uf_pairs_test.jsonl ultrachat_sft.jsonl; do
  chk_file "$SRC/data/$f" "read data/$f"
done

echo
echo "=== $pass passed, $fail failed ==="
if [ "$fail" -eq 0 ]; then
  echo "ALL CHECKS PASSED. Pull the shared models + data into your working copy:"
  echo "    rsync -a \"$SRC\"/models/ results/     # -> results/sft_uc_1p7b_model, ..."
  echo "    rsync -a \"$SRC\"/data/   data/         # -> data/uf_pairs_*.jsonl, ..."
  echo "Then follow the pipeline in ONBOARDING.md (train -> gen -> judge)."
else
  echo "Some checks FAILED. First confirm you're in 'aip-rudner' (top line)."
  echo "If you ARE and paths still fail, send Taehyun this + the output below:"
  for p in "${failed_paths[@]}"; do
    echo "  ---- namei -l $p ----"
    namei -l "$p" 2>&1 | sed 's/^/    /'
  done
fi
