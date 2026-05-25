echo "Table 2 + Table S4 — Main results: verify all trained models with LipsLev + Charmer (1000 samples; fake-news 1000 for LipsLev)."
for dataset in ag_news sst2 imdb; do
  for p in 1 2 inf; do
    for seed in 1 2 3; do
      python verify.py --dataset $dataset --p $p --seed $seed --n_samples 1000
    done
  done
done

for p in 1 2 inf; do
  for seed in 1 2 3; do
    python verify.py --dataset fake-news --p $p --seed $seed --n_samples 1000
  done
done
