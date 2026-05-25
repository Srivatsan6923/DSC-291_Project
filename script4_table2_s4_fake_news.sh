echo "Table 2 + Table S4 — Main results: Fake-News training (2 classes, kernel=10, 3 seeds, p in 1/2/inf). Download train.csv and test.csv from Kaggle first."
for seed in 1 2 3; do
  for p in 1 2 inf; do
    python train.py --dataset fake-news --n_classes 2 --p $p --seed $seed --kernel_size 10
  done
done
