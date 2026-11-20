#!/bin/bash
# Train LMs for a list of tokenizers sequentially on one GPU (skips finished runs).
#   scripts/run_lm_queue.sh <config> <gpu> <seeds e.g. "0 1"> tok1 tok2 ...
set -u
CFG=$1; GPU=$2; SEEDS=$3; shift 3
cd "$(dirname "$0")/.." && . .venv/bin/activate
OUT=$(python -c "from motok.config import load_cfg; print(load_cfg('$CFG').out)")/lm
mkdir -p $OUT
for tok in "$@"; do
  for s in $SEEDS; do
    [ -f $OUT/${tok}_s$s.json ] && [ $(python -c "import json;print(json.load(open('$OUT/${tok}_s$s.json'))['args']['epochs'])" 2>/dev/null || echo 0) -ge 4 ] && continue
    python scripts/07_lm.py --config $CFG --tokenizer $tok --gpu $GPU --seed $s > $OUT/${tok}_s$s.log 2>&1
  done
done
echo "[queue gpu$GPU] done"
