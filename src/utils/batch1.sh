#!/bin/bash
cd "$(git rev-parse --show-toplevel)" || exit 1
for f in T16_reluDE1 T12_reluTP55 T05_reluW_rat T15_indXLcap T09_reluV_ci T02_reluneg T24_reluTP45W T06_indW_rat T01_inddef; do
  R=$(python3 -m kaggle competitions submit -c switch-energy-x-india -f subs/X_$f.csv -m "probe $f" 2>&1 | grep -oE "[0-9]+ submissions remaining today|400 Client Error|error")
  echo "X_$f -> ${R:-UNKNOWN}"
  sleep 3
done
