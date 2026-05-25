echo "Table 2 + Table S4 — Main results: IMDB training (2 classes, kernel=10, 3 seeds, p in 1/2/inf)."
for seed in 1 2 3; do
  for p in 1 2 inf; do
    python train.py --dataset imdb --n_classes 2 --p $p --seed $seed --kernel_size 10
  done
done
