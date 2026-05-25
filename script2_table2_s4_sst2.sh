echo "Table 2 + Table S4 — Main results: SST-2 training (2 classes, kernel=5, 3 seeds, p in 1/2/inf)."
for seed in 1 2 3; do
  for p in 1 2 inf; do
    python train.py --dataset sst2 --n_classes 2 --p $p --seed $seed --kernel_size 5
  done
done
