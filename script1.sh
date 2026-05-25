echo "Table 2 + Table S4 — Main results: AG-News training (4 classes, kernel=10, 3 seeds, p in 1/2/inf)."
for seed in 1 2 3; do
  for p in 1 2 inf; do
    python train.py --dataset ag_news --n_classes 4 --p $p --seed $seed --kernel_size 10
  done
done
