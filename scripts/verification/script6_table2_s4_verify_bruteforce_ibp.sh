#!/usr/bin/env bash
set -Eeuo pipefail

echo "Table 2 + Table S4 - Brute-Force and IBP verification for AG-News, SST-2, IMDB, and Fake-News."
echo "Using main lr=100, nl=1 weights_best.pt checkpoints; outputs use suffix _lr100_best."

SUFFIX="${SUFFIX:-_lr100_best}"
PS=(${PS:-inf 1.0 2.0})

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

samples_for_dataset() {
  local dataset="$1"

  if [ "$dataset" = "fake-news" ]; then
    echo 50
  else
    echo 1000
  fi
}

seeds_for_dataset() {
  local dataset="$1"

  case "$dataset" in
    ag_news|sst2)
      echo "1 2 3"
      ;;
    imdb|fake-news)
      echo "1"
      ;;
    *)
      echo "Unsupported dataset: $dataset" >&2
      return 1
      ;;
  esac
}

run_verify() {
  local dataset="$1"
  local p="$2"
  local seed="$3"
  local n_samples="$4"
  local model_path="$5"

  echo "========================================"
  echo "dataset=$dataset p=$p seed=$seed n_samples=$n_samples"
  echo "$model_path"

  python -u verify.py \
    --dataset "$dataset" \
    --model_path "$model_path" \
    --n_samples "$n_samples" \
    --suffix "$SUFFIX" \
    --method bf

  python -u verify.py \
    --dataset "$dataset" \
    --model_path "$model_path" \
    --n_samples "$n_samples" \
    --suffix "$SUFFIX" \
    --method ibp
}

for dataset in ag_news sst2 imdb fake-news; do
  n_samples="$(samples_for_dataset "$dataset")"
  read -r -a seeds <<< "$(seeds_for_dataset "$dataset")"

  for p in "${PS[@]}"; do
    for seed in "${seeds[@]}"; do
      model_path="$(find_model "$dataset" "$p" "$seed" || true)"

      if [ -z "$model_path" ]; then
        echo "Missing checkpoint: dataset=$dataset p=$p seed=$seed"
        continue
      fi

      run_verify "$dataset" "$p" "$seed" "$n_samples" "$model_path"
    done
  done
done
