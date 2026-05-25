echo "Table S6 — Latency by p and number of layers (AG-News, 1–4 layers, seed 1, train then verify 1000 samples)."
for layers in 1 2 3 4; do
  for p in 1 2 inf; do
    python train.py --dataset ag_news --n_classes 4 --p $p --seed 1 \
      --n_layers $layers --kernel_size 10
    python verify.py --dataset ag_news --p $p --seed 1 \
      --n_layers $layers --n_samples 1000
  done
done
