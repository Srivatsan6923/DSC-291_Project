echo "Table 3 — Regularization vs. enforced 1-Lipschitz on SST-2: ConvModel sweep with lips_reg (ConvLips baselines reuse Table 2 checkpoints)."
for p in 1 2 inf; do
  for lam in 0 0.001 0.01 0.1; do
    python train.py --dataset sst2 --n_classes 2 --p $p --seed 1 \
      --model_name Conv --lips_reg $lam --lr 0.01 --kernel_size 5
  done
done
