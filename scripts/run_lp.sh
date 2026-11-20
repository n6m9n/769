#!/bin/bash
# Parallel LP pipeline for one problem: 3 anchors in parallel -> 5 sweeps in parallel -> merge.
#   [SWEEPS="parity morph"] scripts/run_lp.sh <config> <out-name> [extra 04_sweep args...]
set -u
CFG=$1; NAME=$2; shift 2
cd "$(dirname "$0")/.." && . .venv/bin/activate
LOG=results/$NAME/lp_logs; mkdir -p $LOG
for an in compression minmax min_morph; do
  /usr/bin/time -v python scripts/04_sweep.py --config $CFG --out-name $NAME --phase anchor --anchor $an "$@" > $LOG/anchor_$an.log 2>&1 &
done
wait
for sw in ${SWEEPS:-parity morph grid wpar wmor}; do
  /usr/bin/time -v python scripts/04_sweep.py --config $CFG --out-name $NAME --phase sweep --sweeps $sw "$@" > $LOG/sweep_$sw.log 2>&1 &
done
wait
python scripts/04_sweep.py --config $CFG --out-name $NAME --phase merge > $LOG/merge.log 2>&1
echo "[run_lp] $NAME done"
