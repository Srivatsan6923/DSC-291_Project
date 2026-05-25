echo "Step 0 — Confirm fix: verify train_verifiable and model_name flags exist in train.py before running experiments."
grep "train_verifiable" train.py   # must return a line
grep "model_name" train.py | head -5
