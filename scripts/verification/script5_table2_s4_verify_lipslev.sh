#!/usr/bin/env bash
set -Eeuo pipefail

echo "Table 2 + Table S4 - LipsLev and Charmer verification for AG-News, SST-2, IMDB, and Fake-News."
echo "Using main lr=100, nl=1 weights_best.pt checkpoints; outputs use suffix _lr100_best."

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

charmer_samples_for_dataset() {
  local dataset="$1"

  if [ "$dataset" = "fake-news" ]; then
    echo 50
  else
    echo 1000
  fi
}

csv_rows() {
  local path="$1"

  if [ ! -f "$path" ]; then
    echo 0
    return
  fi

  python - "$path" <<'PY'
import sys
import pandas as pd

try:
    print(len(pd.read_csv(sys.argv[1])))
except Exception:
    print(0)
PY
}

run_one() {
  local dataset="$1"
  local p="$2"
  local seed="$3"
  local model_path="$4"
  local run_dir=""
  local lips_file=""
  local charmer_file=""
  local lips_rows=0
  local charmer_rows=0
  local lips_samples=1000
  local charmer_samples=""

  run_dir="$(dirname "$model_path")"
  lips_file="$run_dir/${SUFFIX}_results_lipslev.csv"
  charmer_file="$run_dir/${SUFFIX}_results_charmer.csv"
  charmer_samples="$(charmer_samples_for_dataset "$dataset")"

  lips_rows="$(csv_rows "$lips_file")"
  charmer_rows="$(csv_rows "$charmer_file")"

  echo "========================================"
  echo "dataset=$dataset p=$p seed=$seed"
  echo "$model_path"
  echo "LipsLev rows=$lips_rows / $lips_samples"
  echo "Charmer rows=$charmer_rows / $charmer_samples"

  if [ "$lips_rows" -lt "$lips_samples" ]; then
    echo "Running LipsLev..."
    python -u verify.py \
      --dataset "$dataset" \
      --model_path "$model_path" \
      --n_samples "$lips_samples" \
      --suffix "$SUFFIX" \
      --method lipslev
  else
    echo "Skipping LipsLev; already complete."
  fi

  if [ "$charmer_rows" -lt "$charmer_samples" ]; then
    echo "Running Charmer..."
    python -u verify.py \
      --dataset "$dataset" \
      --model_path "$model_path" \
      --n_samples "$charmer_samples" \
      --suffix "$SUFFIX" \
      --method charmer
  else
    echo "Skipping Charmer; already complete."
  fi
}

for dataset in "${DATASETS[@]}"; do
  for p in "${PS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      model_path="$(find_model "$dataset" "$p" "$seed" || true)"

      if [ -z "$model_path" ]; then
        echo "Missing checkpoint: dataset=$dataset p=$p seed=$seed"
        continue
      fi

      run_one "$dataset" "$p" "$seed" "$model_path"
    done
  done
done
