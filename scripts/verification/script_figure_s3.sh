#!/usr/bin/env bash
set -Eeuo pipefail

echo "Figure S3 - Learning rate grid search on AG-News and SST-2 (seed 1, p in 1/2/inf)."
echo "Outputs are isolated under results_ConvLips*/figure_s3_* to avoid parallel-run clashes."

for p in 1 2 inf; do
  for lr in 0.1 0.5 1 5 10 50 100 500 1000; do
    python -u train.py --dataset ag_news --n_classes 4 --p "$p" --seed 1 \
      --lr "$lr" --kernel_size 10 --output_folder figure_s3
  done
done

for p in 1 2 inf; do
  for lr in 0.1 0.5 1 5 10 50 100 500 1000; do
    python -u train.py --dataset sst2 --n_classes 2 --p "$p" --seed 1 \
      --lr "$lr" --kernel_size 5 --output_folder figure_s3
  done
done
