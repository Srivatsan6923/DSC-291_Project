"""
Reproduce Figure S3 from LipsLev: learning rate selection for SST-2 and AG-News.
Reads _results_lipslev.csv files across (dataset, p, lr, seed) and plots
clean + verified-k=1 accuracy vs learning rate.

Layout (matches paper):
    [ SST-2 Clean Acc  ] [ AG-News Clean Acc ]
    [ SST-2 Ver. Acc.  ] [ AG-News Ver. Acc. ]   ← legend here

CSV columns: sentence, true_label, pred_label, margin, radius, time
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG ─────────────────────────────────────────────────────────
ROOT = "/home/srivatunix2025/DSC-291_Project"

LRS = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
SEEDS = [1, 2, 3]

P_SPECS = [
    ("inf", r"$\infty$", "#1f77b4"),  # blue
    (1,     "1",         "#d62728"),  # red
    (2,     "2",         "#2ca02c"),  # green
]

# (dataset_arg, display_name, kernel_size)
DATASETS = [
    ("sst2",    "SST-2",    5),
    ("ag_news", "AG-News",  10),
]

# ── PATH BUILDING ─────────────────────────────────────────────────
def folder_suffix_for_p(p):
    return "" if p == "inf" else f"_p{float(p)}"

def get_csv_path(dataset, p, lr, seed, kernel_size):
    suffix = folder_suffix_for_p(p)
    folder = (
        f"results_ConvLips{suffix}/"
        f"figure_s3_{dataset}_valid1000_optSGD_schcyclic_reduce-sum_"
        f"nl1_bs128_es150_hs100_ks{kernel_size}_lr{lr}_lm0_mp0_"
        f"lips_reg0_steps_dec1000000_seed{seed}"
    )
    return os.path.join(ROOT, folder, "_results_lipslev.csv")

# ── ACCURACY COMPUTATION ──────────────────────────────────────────
def compute_accuracies(csv_path):
    """Returns (clean_acc, verified_k1) as fractions, or (None, None) on bad CSV."""
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"  ✗ unreadable CSV ({type(e).__name__}): {csv_path}")
        return None, None
    if len(df) == 0 or "pred_label" not in df.columns:
        print(f"  ✗ no data rows: {csv_path}")
        return None, None
    correct = df["pred_label"] == df["true_label"]
    clean_acc = correct.mean()
    verified_k1 = (correct & (df["radius"] >= 1)).mean()
    return clean_acc, verified_k1

def aggregate_seeds(dataset, p, lr, kernel_size, seeds=SEEDS):
    cleans, verifieds = [], []
    for seed in seeds:
        path = get_csv_path(dataset, p, lr, seed, kernel_size)
        if not os.path.exists(path):
            print(f"  ⚠ missing: {dataset} p={p} lr={lr} seed={seed}")
            continue
        c, v = compute_accuracies(path)
        if c is None:
            continue
        cleans.append(c)
        verifieds.append(v)
    if not cleans:
        return np.nan, np.nan, np.nan, np.nan
    return np.mean(cleans), np.std(cleans), np.mean(verifieds), np.std(verifieds)
# ── PLOTTING ──────────────────────────────────────────────────────
def plot_figure_s3(save_path):
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset":  "cm",
        "axes.labelsize":    13,
        "axes.titlesize":    14,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   12,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True)

    for col_idx, (dataset, label_ds, kernel_size) in enumerate(DATASETS):
        print(f"\n=== {label_ds} (kernel_size={kernel_size}) ===")

        # ── Collect data ──
        data = {}
        for p, _, _ in P_SPECS:
            clean_m, clean_s, ver_m, ver_s = [], [], [], []
            for lr in LRS:
                cm, cs, vm, vs = aggregate_seeds(dataset, p, lr, kernel_size)
                clean_m.append(cm); clean_s.append(cs)
                ver_m.append(vm);   ver_s.append(vs)
            data[p] = tuple(np.array(x) for x in (clean_m, clean_s, ver_m, ver_s))
            print(f"  p={p:>3} | clean    = {[f'{x:.3f}' for x in clean_m]}")
            print(f"          | verified = {[f'{x:.3f}' for x in ver_m]}")

        # ── Plot ──
        ax_clean = axes[0, col_idx]
        ax_ver   = axes[1, col_idx]

        for p, label, color in P_SPECS:
            cm, cs, vm, vs = data[p]
            # top row: clean
            ax_clean.plot(LRS, cm, color=color, linewidth=1.8, label=label)
            ax_clean.fill_between(LRS, cm - cs, cm + cs,
                                  color=color, alpha=0.25, linewidth=0)
            # bottom row: verified
            ax_ver.plot(LRS, vm, color=color, linewidth=1.8, label=label)
            ax_ver.fill_between(LRS, vm - vs, vm + vs,
                                color=color, alpha=0.25, linewidth=0)

        # title only on top row
        ax_clean.set_title(label_ds)
        # y-labels only on leftmost column
        if col_idx == 0:
            ax_clean.set_ylabel("Clean Acc.")
            ax_ver.set_ylabel("Ver. Acc.")
        # x-label only on bottom row
        ax_ver.set_xlabel("Learning rate")

        # log scale on x-axis with custom tick labels
        for ax in (ax_clean, ax_ver):
            ax.set_xscale("log")
            ax.set_xticks(LRS)
            ax.set_xticklabels([f"{lr:g}" for lr in LRS])
            ax.minorticks_off()
            ax.tick_params(axis="x", which="both", bottom=True)

    # Legend in bottom-right subplot (matches paper)
    legend = axes[1, 1].legend(loc="lower right", frameon=True,
                                edgecolor="black", framealpha=1.0,
                                fancybox=False)
    legend.get_frame().set_linewidth(0.6)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = save_path.replace(".pdf", f".{ext}")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  saved → {out}")
    plt.close()

# ── MAIN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_figure_s3("figure_s3_lr_selection.pdf")
    print("\n✓ Done.")