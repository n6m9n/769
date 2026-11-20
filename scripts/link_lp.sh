#!/bin/bash
# Link LP tokenizers of results/<run> into results/full/tokenizers as <prefix>_<tag>.pkl
#   scripts/link_lp.sh full_10k lp10k
cd "$(dirname "$0")/.."
for f in results/$1/tokenizers/toast_*.pkl; do
  t=$(basename $f .pkl); t=${t#toast_}
  ln -sfn ../../$1/tokenizers/toast_$t.pkl results/full/tokenizers/$2_$t.pkl
done
ls results/full/tokenizers | grep -c "^$2_"
