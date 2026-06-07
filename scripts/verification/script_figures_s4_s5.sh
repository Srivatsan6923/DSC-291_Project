#!/usr/bin/env bash
set -Eeuo pipefail

echo "Figures S4 and S5 - Deeper models (1-4 layers) on AG-News and SST-2."
echo "Training with lr=100, then verifying 1000 samples with LipsLev only."
echo "Outputs are isolated under results_ConvLips*/figures_s4_s5_* to avoid parallel-run clashes."

LR=100
SEED=1
OUTPUT_FOLDER=figures_s4_s5

result_dir() {
  local dataset="$1"
  local p="$2"
  local layers="$3"
  local kernel_size="$4"
  local p_dir=""

  if [ "$p" != "inf" ]; then
    p_dir="_p${p}.0"
  fi

  printf 'results_ConvLips%s/%s_%s_valid1000_optSGD_schcyclic_reduce-sum_nl%s_bs128_es150_hs100_ks%s_lr100.0_lm0_mp0_lips_reg0_steps_dec1000000_seed%s' \
    "$p_dir" "$OUTPUT_FOLDER" "$dataset" "$layers" "$kernel_size" "$SEED"
}

for dataset in ag_news sst2; do
  if [ "$dataset" = "ag_news" ]; then
    n_classes=4
    kernel_size=10
  else
    n_classes=2
    kernel_size=5
  fi

  for layers in 1 2 3 4; do
    for p in 1 2 inf; do
      python -u train.py --dataset "$dataset" --n_classes "$n_classes" --p "$p" --seed "$SEED" \
        --n_layers "$layers" --kernel_size "$kernel_size" --lr "$LR" --output_folder "$OUTPUT_FOLDER"

      model_path="$(result_dir "$dataset" "$p" "$layers" "$kernel_size")/weights_best.pt"
      python -u verify.py --dataset "$dataset" --model_path "$model_path" --n_samples 1000 --method ours
    done
  done
done
