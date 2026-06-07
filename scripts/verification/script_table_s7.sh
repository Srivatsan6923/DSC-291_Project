#!/usr/bin/env bash
set -Eeuo pipefail

echo "Table S7 - RS-Del comparison for AG-News, SST-2, IMDB, and Fake-News."
echo "Using main lr=100, nl=1 weights_best.pt checkpoints; outputs use suffix _lr100_best."

SAMPLES="${SAMPLES:-1000}"
SUFFIX="${SUFFIX:-_lr100_best}"
DATASETS=(${DATASETS:-ag_news sst2 imdb fake-news})
PS=(${PS:-inf 1.0 2.0})
SEEDS=(${SEEDS:-1 2 3})

find_model() {
  local dataset="$1"
  local p="$2"
  local seed="$3"
  local base=""

  if [ "$p" = "inf" ]; then
    base="results_ConvLips"
  else
    base="results_ConvLips_p$p"
  fi

  if [ ! -d "$base" ]; then
    return 0
  fi

  find "$base" -type f -name "weights_best.pt" \
    | grep "/results_${dataset}_valid1000_" \
    | grep "_nl1_" \
    | grep "_lr100" \
    | grep "_seed${seed}/weights_best.pt" \
    | sort \
    | head -n 1
}

for dataset in "${DATASETS[@]}"; do
  for p in "${PS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      model_path="$(find_model "$dataset" "$p" "$seed" || true)"

      if [ -z "$model_path" ]; then
        echo "Missing checkpoint: dataset=$dataset p=$p seed=$seed"
        continue
      fi

      echo "========================================"
      echo "dataset=$dataset p=$p seed=$seed n_samples=$SAMPLES"
      echo "$model_path"

      python -u verify.py \
        --dataset "$dataset" \
        --model_path "$model_path" \
        --n_samples "$SAMPLES" \
        --suffix "$SUFFIX" \
        --method rsdel
    done
  done
done
