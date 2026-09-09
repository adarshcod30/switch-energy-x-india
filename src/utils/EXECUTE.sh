#!/bin/bash
# Batch A: 16 orthogonal probes + Xv3 linear model
cd "$(git rev-parse --show-toplevel)" || exit 1
LIST="$(python3 -c "import json;print(' '.join(json.load(open('artifacts/probe_order.json'))))") V3_lin01"
for f in $LIST; do
  R=$(python3 -m kaggle competitions submit -c switch-energy-x-india -f subs/$f.csv -m "$f" 2>&1 | grep -oE "[0-9]+ submissions remaining today")
  echo "$f -> ${R:-FAILED}"
  sleep 2
done
