echo "Table S7 — RS-Del comparison (1000 samples, all datasets and seeds)."
for dataset in ag_news sst2 imdb; do
  for p in 1 2 inf; do
    for seed in 1 2 3; do
      python verify.py --dataset $dataset --p $p --seed $seed \
        --n_samples 1000 --methods rsdel
    done
  done
done

for p in 1 2 inf; do
  for seed in 1 2 3; do
    python verify.py --dataset fake-news --p $p --seed $seed \
      --n_samples 1000 --methods rsdel
  done
done
