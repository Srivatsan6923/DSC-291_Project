# Experiment Reproduction Guide

This file explains how to run the experiment scripts in this repository and how
each script maps to the reproduced tables and figures.

Run all commands from the repository root:

```bash
cd /path/to/repository
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For Fake-News experiments, place the Kaggle Fake News files here:

```text
fake-news/train.csv
fake-news/test.csv
```

---

## Script Layout

```text
scripts/
├── training/
│   ├── script1_table2_s4_agnews.sh
│   ├── script2_table2_s4_sst2.sh
│   ├── script3_table2_s4_imdb.sh
│   └── script4_table2_s4_fake_news.sh
└── verification/
    ├── script5_table2_s4_verify_lipslev.sh
    ├── script6_table2_s4_verify_bruteforce_ibp.sh
    ├── script_table3.sh
    ├── script_table_s5.sh
    ├── script_table_s6.sh
    ├── script_table_s7.sh
    ├── script_figure_s3.sh
    ├── script_figures_s4_s5.sh
    ├── fig1_figs2.py
    ├── generate_figure_s3.py
    └── generate_figures_s4_s5.py
```

---

## Main Run Order

Run training first:

```bash
bash scripts/training/script1_table2_s4_agnews.sh
bash scripts/training/script2_table2_s4_sst2.sh
bash scripts/training/script3_table2_s4_imdb.sh
bash scripts/training/script4_table2_s4_fake_news.sh
```

Then run verification:

```bash
bash scripts/verification/script5_table2_s4_verify_lipslev.sh
bash scripts/verification/script6_table2_s4_verify_bruteforce_ibp.sh
bash scripts/verification/script_table_s7.sh
```

Then run the remaining table and figure scripts as needed:

```bash
bash scripts/verification/script_table3.sh
bash scripts/verification/script_table_s5.sh
bash scripts/verification/script_figures_s4_s5.sh
bash scripts/verification/script_table_s6.sh
bash scripts/verification/script_figure_s3.sh
python scripts/verification/generate_figure_s3.py
python scripts/verification/generate_figures_s4_s5.py
python scripts/verification/fig1_figs2.py
```

---

## Training Scripts

### AG-News

```bash
bash scripts/training/script1_table2_s4_agnews.sh
```

Runs:

```text
dataset: ag_news
n_classes: 4
kernel_size: 10
p: 1, 2, inf
seed: 1, 2, 3
```

### SST-2

```bash
bash scripts/training/script2_table2_s4_sst2.sh
```

Runs:

```text
dataset: sst2
n_classes: 2
kernel_size: 5
p: 1, 2, inf
seed: 1, 2, 3
```

### IMDB

```bash
bash scripts/training/script3_table2_s4_imdb.sh
```

Runs:

```text
dataset: imdb
n_classes: 2
kernel_size: 10
lr: 100
valid_size: 1000
p: 1, 2, inf
seed: 1, 2, 3
```

### Fake-News

```bash
bash scripts/training/script4_table2_s4_fake_news.sh
```

Runs:

```text
dataset: fake-news
n_classes: 2
kernel_size: 10
lr: 100
valid_size: 1000
p: 1, 2, inf
seed: 1, 2, 3
```

---

## Verification Scripts

### LipsLev and Charmer

```bash
bash scripts/verification/script5_table2_s4_verify_lipslev.sh
```

Runs LipsLev and Charmer on the main `lr=100`, `nl=1` checkpoints.

Defaults:

```text
datasets: ag_news, sst2, imdb, fake-news
p: inf, 1.0, 2.0
seed: 1, 2, 3
LipsLev samples: 1000
Charmer samples: 1000, except fake-news uses 50
```

Outputs:

```text
_lr100_best_results_lipslev.csv
_lr100_best_results_charmer.csv
```

Run a subset:

```bash
DATASETS="ag_news sst2" PS="2.0" SEEDS="1" \
  bash scripts/verification/script5_table2_s4_verify_lipslev.sh
```

### Brute-Force and IBP

```bash
bash scripts/verification/script6_table2_s4_verify_bruteforce_ibp.sh
```

Runs:

```text
AG-News:   seeds 1,2,3; 1000 samples
SST-2:     seeds 1,2,3; 1000 samples
IMDB:      seed 1 only; 1000 samples
Fake-News: seed 1 only; 50 samples
```

Outputs:

```text
_lr100_best_results_bruteforce.csv
_lr100_best_results_ibp.csv
```

### RS-Del for Table S7

```bash
bash scripts/verification/script_table_s7.sh
```

Runs RS-Del on all four datasets, all p values, and all seeds.

Outputs:

```text
_lr100_best_results_rsdel.csv
```

Run only one dataset:

```bash
DATASETS="imdb" bash scripts/verification/script_table_s7.sh
```

---

## Tables

### Table 2 and Table S4

Use the four training scripts plus:

```bash
bash scripts/verification/script5_table2_s4_verify_lipslev.sh
bash scripts/verification/script6_table2_s4_verify_bruteforce_ibp.sh
```

Main result CSVs:

```text
_lr100_best_results_lipslev.csv
_lr100_best_results_charmer.csv
_lr100_best_results_bruteforce.csv
_lr100_best_results_ibp.csv
```

### Table 3

```bash
bash scripts/verification/script_table3.sh
```

Runs SST-2 `Conv` models with:

```text
p: inf, 1, 2
seed: 1, 2, 3
lips_reg: 0, 0.001, 0.01, 0.1
lr: 0.01
kernel_size: 5
```

### Table S5

```bash
bash scripts/verification/script_table_s5.sh
```

Runs Fake-News LipsLev verification:

```text
p: 1, 2, inf
seed: 1, 2, 3
n_samples: 1000
suffix: _s5_k10
```

Output:

```text
_s5_k10_results_lipslev.csv
```

### Table S6

First generate the deeper-model LipsLev CSVs:

```bash
bash scripts/verification/script_figures_s4_s5.sh
```

Then summarize latency:

```bash
bash scripts/verification/script_table_s6.sh
```

Output:

```text
table_s6_outputs/table_s6_latency_ag_news.csv
```

### Table S7

Run LipsLev/Charmer and RS-Del:

```bash
bash scripts/verification/script5_table2_s4_verify_lipslev.sh
bash scripts/verification/script_table_s7.sh
```

Compare:

```text
_lr100_best_results_lipslev.csv
_lr100_best_results_rsdel.csv
```

---

## Figures

### Figure S3

Run the learning-rate grid:

```bash
bash scripts/verification/script_figure_s3.sh
```

Then generate the plot:

```bash
python scripts/verification/generate_figure_s3.py
```

Outputs:

```text
figure_s3_lr_selection.pdf
figure_s3_lr_selection.png
```

### Figures S4 and S5

Train and verify deeper models:

```bash
bash scripts/verification/script_figures_s4_s5.sh
```

Generate plots:

```bash
python scripts/verification/generate_figures_s4_s5.py
```

Outputs:

```text
figure_s4_ag_news.pdf
figure_s4_ag_news.png
figure_s5_sst2.pdf
figure_s5_sst2.png
```

### Figure 1 and Figure S2

```bash
python scripts/verification/fig1_figs2.py
```

Outputs:

```text
figure1_reproduction.png
figure1_reproduction.pdf
figureS2_reproduction.png
figureS2_reproduction.pdf
figure1_average_lengths.csv
```

---

## Output Files

Training directories follow this pattern:

```text
results_ConvLips[_p{p}]/{output_folder}_{dataset}_valid1000_optSGD_schcyclic_reduce-sum_nl{layers}_bs128_es150_hs100_ks{kernel}_lr{lr}_lm0_mp0_lips_reg0_steps_dec1000000_seed{seed}/
```

Each training directory contains:

```text
log.csv
weights_best.pt
weights_best_dict.pt
weights_last.pt
weights_last_dict.pt
```

Verification CSVs are written in the same directory as the model checkpoint.

Common verification outputs:

```text
_results_lipslev.csv
_results_charmer.csv
_results_bruteforce.csv
_results_ibp.csv
_results_rsdel.csv
_lr100_best_results_lipslev.csv
_lr100_best_results_charmer.csv
_lr100_best_results_bruteforce.csv
_lr100_best_results_ibp.csv
_lr100_best_results_rsdel.csv
```

---

## Paper Mapping

| Paper item | Scripts |
|------------|---------|
| Table 2 | `scripts/training/script1-4_*`, `script5_*`, `script6_*` |
| Table 3 | `scripts/verification/script_table3.sh` |
| Table S4 | Same scripts as Table 2 |
| Table S5 | `scripts/verification/script_table_s5.sh` |
| Table S6 | `scripts/verification/script_figures_s4_s5.sh`, then `script_table_s6.sh` |
| Table S7 | `scripts/verification/script5_*`, `script_table_s7.sh` |
| Figure 1 | `scripts/verification/fig1_figs2.py` |
| Figure S2 | `scripts/verification/fig1_figs2.py` |
| Figure S3 | `scripts/verification/script_figure_s3.sh`, then `generate_figure_s3.py` |
| Figure S4 | `scripts/verification/script_figures_s4_s5.sh`, then `generate_figures_s4_s5.py` |
| Figure S5 | `scripts/verification/script_figures_s4_s5.sh`, then `generate_figures_s4_s5.py` |
