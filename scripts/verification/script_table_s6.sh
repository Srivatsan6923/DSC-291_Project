#!/usr/bin/env bash
set -Eeuo pipefail

echo "Table S6 - Latency by p and number of layers for AG-News."
echo "Reusing LipsLev outputs produced by script_figures_s4_s5.sh; no retraining or re-verification."

DATASET=ag_news
KERNEL_SIZE=10
SEED=1
OUTPUT_FOLDER=figures_s4_s5
SUMMARY_DIR=table_s6_outputs
SUMMARY_CSV="${SUMMARY_DIR}/table_s6_latency_ag_news.csv"
WAIT_SECONDS="${TABLE_S6_WAIT_SECONDS:-86400}"
POLL_SECONDS="${TABLE_S6_POLL_SECONDS:-60}"

result_dir() {
  local p="$1"
  local layers="$2"
  local p_dir=""

  if [ "$p" != "inf" ]; then
    p_dir="_p${p}.0"
  fi

  printf 'results_ConvLips%s/%s_%s_valid1000_optSGD_schcyclic_reduce-sum_nl%s_bs128_es150_hs100_ks%s_lr100.0_lm0_mp0_lips_reg0_steps_dec1000000_seed%s' \
    "$p_dir" "$OUTPUT_FOLDER" "$DATASET" "$layers" "$KERNEL_SIZE" "$SEED"
}

wait_for_file() {
  local path="$1"
  local waited=0

  while [ ! -s "$path" ]; do
    if [ "$waited" -ge "$WAIT_SECONDS" ]; then
      echo "Missing after waiting ${WAIT_SECONDS}s: $path" >&2
      return 1
    fi

    echo "Waiting for S4/S5 LipsLev output: $path"
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
  done
}

mkdir -p "$SUMMARY_DIR"
printf 'dataset,p,n_layers,mean_latency_seconds,n_timed_samples,source_csv\n' > "$SUMMARY_CSV"

for layers in 1 2 3 4; do
  for p in 1 2 inf; do
    csv_path="$(result_dir "$p" "$layers")/_results_lipslev.csv"
    wait_for_file "$csv_path"

    python - "$DATASET" "$p" "$layers" "$csv_path" >> "$SUMMARY_CSV" <<'PY'
import csv
import sys

dataset, p, layers, csv_path = sys.argv[1:]
times = []

with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        value = row.get("time", "")
        if value not in ("", "None", "nan"):
            times.append(float(value))

mean_latency = sum(times) / len(times) if times else float("nan")
print(f"{dataset},{p},{layers},{mean_latency:.6f},{len(times)},{csv_path}")
PY
  done
done

echo "Wrote Table S6 latency summary: $SUMMARY_CSV"
