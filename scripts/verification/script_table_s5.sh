#!/usr/bin/env bash
set -Eeuo pipefail

echo "Table S5 — Fake-News LipsLev verification up to k=10 (3 seeds, p in 1/2/inf, 1000 samples)."

for p in 1 2 inf; do
  for seed in 1 2 3; do
    if [ "$p" = "inf" ]; then
      BASE="results_ConvLips"
    else
      BASE="results_ConvLips_p$p.0"
    fi

    MODEL_PATH=$(find "$BASE" -type f -name "weights_best.pt" \
      | grep "fake-news" \
      | grep "lr100" \
      | grep "seed$seed" \
      | head -n 1)

    if [ -z "$MODEL_PATH" ]; then
      echo "Missing checkpoint: p=$p seed=$seed base=$BASE"
      continue
    fi

    echo "p=$p seed=$seed model=$MODEL_PATH"

    python verify.py \
      --dataset fake-news \
      --model_path "$MODEL_PATH" \
      --n_samples 1000 \
      --suffix "_s5_k10" \
      --method lipslev
  done
done
