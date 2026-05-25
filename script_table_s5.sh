echo "Table S5 — Fake-News LipsLev verification up to k=10 (3 seeds, p in 1/2/inf, 1000 samples)."
for p in 1 2 inf; do
  for seed in 1 2 3; do
    python verify.py --dataset fake-news --p $p --seed $seed --n_samples 1000 --max_k 10
  done
done
