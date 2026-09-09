#!/bin/bash
# Round 1: 16 orthogonalized atom probes.
cd "$(git rev-parse --show-toplevel)" || exit 1
for f in $(python3 -c "import json;print(' '.join(json.load(open('artifacts/probe_order.json'))))"); do
  echo -n "$f: "
  python3 -m kaggle competitions submit -c switch-energy-x-india -f subs/$f.csv -m "orth probe $f" 2>&1 | grep -oE "[0-9]+ submissions remaining today" || echo FAILED
  sleep 2
done
