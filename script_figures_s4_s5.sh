echo "Figures S4 and S5 — Deeper models (1–4 layers) on AG-News and SST-2: train then verify 1000 samples."
for dataset in ag_news sst2; do
  for layers in 1 2 3 4; do
    for p in 1 2 inf; do
      python train.py --dataset $dataset --n_classes 4 --p $p --seed 1 \
        --n_layers $layers --kernel_size 10
      python verify.py --dataset $dataset --p $p --seed 1 \
        --n_layers $layers --n_samples 1000
    done
  done
done
