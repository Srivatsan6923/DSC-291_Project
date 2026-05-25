# Paper reproduction scripts

Shell scripts in the repo root reproduce the tables and figures from the ICLR 2025 paper. Each script prints what it is for, then runs the commands.

Run everything from the repository root:

```bash
cd /path/to/LipsLev
```

Use `bash script_name.sh` (or `chmod +x script*.sh` and `./script_name.sh`).

---

## Prerequisites

1. **Python environment** — install dependencies: `pip install -r requirements.txt`
2. **Fake-News data** — download `train.csv` and `test.csv` from [Kaggle Fake News](https://www.kaggle.com/competitions/fake-news) into a folder named `fake-news/` (the loader in `utils.py` expects that path; if your files live in `fake-news/`, symlink: `ln -s fake-news fake-news`).
3. **Dataset flag** — training/verification code uses `--dataset fake-news` (hyphen). Scripts currently pass `fake-news`; align the flag with `train.py` / `utils.py` before running `script4_*` and fake-news verify scripts.
4. **Verify CLI** — verify scripts call `verify.py` with `--p`, `--seed`, `--methods`, `--max_k`, and `--n_layers`. The stock `verify.py` only accepts `--model_path` and runs all methods. Extend `verify.py` to match the scripts, or run verification manually with paths like `results/example_ag_news/weights_last.pt` (see [Verification outputs](#verification-outputs)).

Confirm defaults once before a long run:

```bash
python train.py --help
```

| Flag | AG-News | SST-2 | IMDB | Fake-News |
|------|---------|-------|------|-----------|
| `--n_classes` | 4 | 2 | 2 | 2 |
| `--kernel_size` | 10 | 5 | 10 | 10 |
| `--embed_size` | 150 | 150 | 150 | 150 |
| `--hidden_size` | 100 | 100 | 100 | 100 |
| `--epochs` | 30 | 30 | 30 | 30 |
| `--model_name` | ConvLips (default) | ConvLips | ConvLips | ConvLips |

---

## Recommended run order

Scripts depend on earlier checkpoints and CSVs. Run in this order:

| Step | Script | Paper output | Depends on |
|------|--------|--------------|------------|
| 0 | `script0_confirm_fix.sh` | — (sanity check) | — |
| 1 | `script1.sh` | Table 2, Table S4 | — |
| 2 | `script2_table2_s4_sst2.sh` | Table 2, Table S4 | — |
| 3 | `script3_table2_s4_imdb.sh` | Table 2, Table S4 | — |
| 4 | `script4_table2_s4_fake-news.sh` | Table 2, Table S4 | Fake-News CSVs |
| 5 | `script5_table2_s4_verify_lipslev.sh` | Table 2, Table S4 | Steps 1–4 |
| 6 | `script6_table2_s4_verify_bruteforce_ibp.sh` | Table 2, Table S4 | Steps 1–4 (slow) |
| 7 | `script_table3.sh` | Table 3 | — (ConvLips SST-2 from step 2 is reused) |
| 8 | `script_table_s5.sh` | Table S5 | Step 4 |
| 9 | `script_table_s6.sh` | Table S6 | — |
| 10 | `script_table_s7.sh` | Table S7 | Steps 1–4 |
| 11 | `script_figure_s3.sh` | Figure S3 | — |
| 12 | `script_figures_s4_s5.sh` | Figures S4, S5 | — |
| 13 | `script_figure1_s2.sh` | Figure 1, Figure S2 | Step 5 (LipsLev CSVs) |

Optional parallelism: steps 1–4 are independent per dataset; step 7 can run anytime after step 2; steps 9–12 are independent of Table 2 verify except where noted.

---

## Script reference

### `script0_confirm_fix.sh`

**Command**

```bash
bash script0_confirm_fix.sh
```

**Purpose** — Step 0: confirm `train_verifiable` and `model_name` exist in `train.py` before launching jobs.

**Expected output** — Terminal only: one line containing `train_verifiable`, plus the first few `model_name` lines from `train.py`. No files written. Exit if `train_verifiable` grep returns nothing.

---

### `script1.sh` — AG-News training

**Command**

```bash
bash script1.sh
```

**Purpose** — Table 2 + Table S4: train ConvLips on AG-News (4 classes, kernel 10, seeds 1–3, p ∈ {1, 2, inf}).

**Runs** — 9 training jobs (3 seeds × 3 norms).

**Expected output** — Under `results_ConvLips/` or `results_ConvLips_p1/` / `results_ConvLips_p2/`:

- Directory per run, e.g.  
  `results_ag_news_valid1000_optSGD_schcyclic_reduce-sum_nl1_bs128_es150_hs100_ks10_lr1_lm0_mp0_lips_reg0_steps_dec1000000_seed{N}/`
- `log.csv` — per-epoch train/valid metrics and `valid_verified_1/2/3`
- `weights_best.pt`, `weights_best_dict.pt`, `weights_last.pt`, `weights_last_dict.pt`

**Console** — Per epoch: train loss, validation accuracy, verified accuracy (k=1 during training).

**Rough runtime** — GPU: tens of minutes per run; CPU: much longer.

---

### `script2_table2_s4_sst2.sh` — SST-2 training

**Command**

```bash
bash script2_table2_s4_sst2.sh
```

**Purpose** — Table 2 + Table S4: SST-2 (2 classes, kernel 5, 3 seeds, 3 norms).

**Runs** — 9 training jobs.

**Expected output** — Same layout as `script1.sh`, with `results_sst2_..._ks5_...` in the run folder name.

---

### `script3_table2_s4_imdb.sh` — IMDB training

**Command**

```bash
bash script3_table2_s4_imdb.sh
```

**Purpose** — Table 2 + Table S4: IMDB (2 classes, kernel 10, 3 seeds, 3 norms).

**Runs** — 9 training jobs.

**Expected output** — Same as above with `results_imdb_..._ks10_...`.

---

### `script4_table2_s4_fake-news.sh` — Fake-News training

**Command**

```bash
bash script4_table2_s4_fake-news.sh
```

**Purpose** — Table 2 + Table S4: Fake-News (2 classes, kernel 10, 3 seeds, 3 norms).

**Runs** — 9 training jobs.

**Expected output** — Same checkpoint/`log.csv` layout with `results_fake-news_...` (once dataset flag matches `fake-news`).

**Prerequisite** — `fake-news/train.csv` and `fake-news/test.csv` present.

---

### `script5_table2_s4_verify_lipslev.sh` — LipsLev + Charmer

**Command**

```bash
bash script5_table2_s4_verify_lipslev.sh
```

**Purpose** — Table 2 + Table S4: verify all Table 2 models with LipsLev and Charmer (1000 test samples; all datasets and seeds).

**Runs** — 36 verification jobs (27 for ag_news/sst2/imdb + 9 for fake-news).

**Expected output** — Next to each checkpoint directory (same folder as `weights_last.pt`):

| File | Content |
|------|---------|
| `_results_lipslev.csv` | Per-sample `sentence`, `true_label`, `pred_label`, `margin`, `radius`, `time` |
| `_results_charmer.csv` | Charmer adversarial robustness (k≤2) |

**Console** — Printed aggregate verified accuracies, e.g. `ours (0.42, 0.31, 0.18)` for k=1,2,3.

**Rough runtime** — LipsLev: minutes–hours per 1000 samples on GPU; Charmer: additional time per sample.

---

### `script6_table2_s4_verify_bruteforce_ibp.sh` — BruteForce + IBP

**Command**

```bash
bash script6_table2_s4_verify_bruteforce_ibp.sh
```

**Purpose** — Table 2 + Table S4: BruteForce and IBP baselines (k=1 only). 1000 samples for ag_news/sst2/imdb; **50** samples for fake-news.

**Runs** — 36 jobs (27 + 9), each much slower than LipsLev.

**Expected output** — Per checkpoint directory:

| File | Content |
|------|---------|
| `_results_bruteforce.csv` | Oracle / brute-force verified accuracy |
| `_results_ibp.csv` | IBP bounds |

**Console** — Lines like `bf 0.12`, `ibp 0.08`.

**Rough runtime** — Hours per configuration on ag_news/sst2/imdb at 1000 samples; fake-news capped at 50 samples but still slow.

---

### `script_table3.sh` — Regularization sweep

**Command**

```bash
bash script_table3.sh
```

**Purpose** — Table 3: SST-2 with **Conv** (not ConvLips), `lips_reg` ∈ {0, 0.001, 0.01, 0.1}, lr=0.01, seed 1, all p. ConvLips baselines come from `script2_table2_s4_sst2.sh` (do not retrain).

**Runs** — 12 training jobs (3 p × 4 λ).

**Expected output** — Under `results_Conv_p{p}/` (p in path for Conv model):

- Same `log.csv` + weights as training scripts
- Folder names include `lips_reg{L}` in the path

**Use** — Compare `valid_verified_*` and test accuracy in `log.csv` against ConvLips runs from step 2.

---

### `script_table_s5.sh` — Fake-News k up to 10

**Command**

```bash
bash script_table_s5.sh
```

**Purpose** — Table S5: LipsLev on fake-news with `--max_k 10` (1000 samples, all p and seeds).

**Runs** — 9 verification jobs.

**Expected output** — Extended LipsLev metrics / CSV (depends on `verify.py` supporting `--max_k`). At minimum, updated `_results_lipslev.csv` with radii usable to compute verified accuracy for k=1…10.

---

### `script_table_s6.sh` — Latency vs depth and p

**Command**

```bash
bash script_table_s6.sh
```

**Purpose** — Table S6: AG-News with 1–4 layers, seed 1, all p; train then verify 1000 samples per config.

**Runs** — 12 train+verify pairs (4 layers × 3 p).

**Expected output**

- Training: `results_ConvLips[_p{p}]/..._nl{L}_.../` with `log.csv` and weights
- Verification: `_results_lipslev.csv` with per-sample `time` column for latency analysis

**Use** — Aggregate `time` from CSVs (and optionally train-time from logs) vs p and `n_layers`.

---

### `script_table_s7.sh` — RS-Del

**Command**

```bash
bash script_table_s7.sh
```

**Purpose** — Table S7: RS-Del verification only (`--methods rsdel`), 1000 samples, all datasets/seeds/p.

**Runs** — 36 verification jobs.

**Expected output** — `_results_rsdel.csv` per checkpoint directory (randomized smoothing baseline).

---

### `script_figure_s3.sh` — Learning-rate grid

**Command**

```bash
bash script_figure_s3.sh
```

**Purpose** — Figure S3: LR grid on AG-News and SST-2 (seed 1, all p, LR ∈ {0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000}).

**Runs** — 54 AG-News + 27 SST-2 = 81 training jobs (note: first loop uses `n_classes 4` for both datasets in the script; SST-2 block overrides with `n_classes 2` and `kernel_size 5`).

**Expected output** — Many run directories with `lr{LR}` in the folder name and `log.csv` recording best valid/test accuracy for each LR.

**Use** — Plot best valid or test accuracy vs learning rate for each (dataset, p).

---

### `script_figures_s4_s5.sh` — Deeper models

**Command**

```bash
bash script_figures_s4_s5.sh
```

**Purpose** — Figures S4 & S5: AG-News and SST-2, layers 1–4, seed 1, all p; train + verify 1000 samples.

**Runs** — 24 train+verify pairs (2 datasets × 4 layers × 3 p).

**Expected output** — Checkpoints and `_results_lipslev.csv` under paths with `_nl{L}_`; use verified accuracy vs depth for the figures.

**Note** — SST-2 loop uses `--n_classes 4` in the script; for correct SST-2 runs use `--n_classes 2` and `--kernel_size 5` (as in `script2_table2_s4_sst2.sh`).

---

### `script_figure1_s2.sh` — Sentence length plots

**Command**

```bash
bash script_figure1_s2.sh
```

**Purpose** — Figure 1 + Figure S2: verified accuracy vs sentence length from existing LipsLev CSVs (no training).

**Prerequisite** — `script5_table2_s4_verify_lipslev.sh` finished for ag_news and sst2 (seed 1, p=2).

**Expected output** — PNG files in the repo root:

- `fig1_ag_news.png`
- `fig1_sst2.png`

**Note** — The plotting snippet globs `results/*{dataset}*p2*seed1*lipslev*.csv`. Actual CSVs live under nested `results_ConvLips_p2/.../_results_lipslev.csv`. Adjust the glob or pass explicit paths if the script finds no files.

---

## Training output layout

ConvLips checkpoints (default scripts) follow this pattern:

```text
results_ConvLips[_p{p}]/results_{dataset}_valid1000_optSGD_schcyclic_reduce-sum_nl{n_layers}_bs128_es150_hs100_ks{kernel}_lr{lr}_lm0_mp0_lips_reg0_steps_dec1000000_seed{seed}/
├── log.csv
├── weights_best.pt
├── weights_best_dict.pt
├── weights_last.pt
└── weights_last_dict.pt
```

- `p=inf` → parent folder `results_ConvLips/` (no `_p` suffix).
- `p=1` or `p=2` → `results_ConvLips_p1/` or `results_ConvLips_p2/`.

`log.csv` columns useful for tables: `valid_acc`, `valid_verified_1`, `valid_verified_2`, `valid_verified_3`, `test_acc` (when validation improves).

---

## Verification outputs

When `verify.py` is pointed at `.../weights_last.pt`, outputs are written **in that checkpoint directory**:

| File | Method |
|------|--------|
| `_results_lipslev.csv` | LipsLev certified radius |
| `_results_charmer.csv` | Charmer (k≤2) |
| `_results_bruteforce.csv` | Brute force |
| `_results_ibp.csv` | IBP |
| `_results_rsdel.csv` | RS-Del |

Example manual verify (current CLI):

```bash
python verify.py \
  --dataset ag_news \
  --model_path results_ConvLips_p2/results_ag_news_valid1000_..._seed1/weights_last.pt \
  --n_classes 4 \
  --n_samples 1000
```

See `results/example_ag_news/` for sample CSV format.

---

## Paper mapping summary

| Paper item | Scripts |
|------------|---------|
| Table 2 | `script1`–`script6` (train + all verify) |
| Table 3 | `script_table3.sh` + ConvLips from `script2` |
| Table S4 | Same as Table 2 (extended / supplementary metrics) |
| Table S5 | `script_table_s5.sh` |
| Table S6 | `script_table_s6.sh` |
| Table S7 | `script_table_s7.sh` |
| Figure 1 | `script_figure1_s2.sh` |
| Figure S2 | `script_figure1_s2.sh` |
| Figure S3 | `script_figure_s3.sh` |
| Figure S4 | `script_figures_s4_s5.sh` |
| Figure S5 | `script_figures_s4_s5.sh` |

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| `fake-news` dataset not found | Use `--dataset fake-news` and `fake-news/` data path |
| Verify scripts fail on unknown flags | Extend `verify.py` or call with `--model_path` |
| Figure script finds no CSVs | Glob nested `results_ConvLips*/**/_results_lipslev.csv` |
| IMDB validation order | `utils.py` TODO: validation set may be label-ordered |
| Out of disk | Each run saves full model `.pt` files (~MB–GB per run) |
