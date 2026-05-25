echo "Figure S3 — Learning rate grid search on AG-News and SST-2 (seed 1, p in 1/2/inf)."
for dataset in ag_news sst2; do
  for p in 1 2 inf; do
    for lr in 0.1 0.5 1 5 10 50 100 500 1000; do
      python train.py --dataset $dataset --n_classes 4 --p $p --seed 1 \
        --lr $lr --kernel_size 10
    done
  done
done

for p in 1 2 inf; do
  for lr in 0.1 0.5 1 5 10 50 100 500 1000; do
    python train.py --dataset sst2 --n_classes 2 --p $p --seed 1 \
      --lr $lr --kernel_size 5
  done
done
