echo "Table 2 + Table S4 — Main results: BruteForce + IBP verification (k=1 only; 1000 samples for ag_news/sst2/imdb, 50 for fake-news)."
for dataset in ag_news sst2 imdb; do
  for p in 1 2 inf; do
    for seed in 1 2 3; do
      python verify.py --dataset $dataset --p $p --seed $seed --n_samples 1000 --methods bruteforce ibp
    done
  done
done

for p in 1 2 inf; do
  for seed in 1 2 3; do
    python verify.py --dataset fake-news --p $p --seed $seed --n_samples 50 --methods bruteforce ibp
  done
done
