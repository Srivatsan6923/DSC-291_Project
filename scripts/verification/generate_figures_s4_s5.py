"""
Reproduce Figures S4 (AG-News) and S5 (SST-2) from LipsLev.
Reads _results_lipslev.csv files across (p, n_layers, seed) and plots
clean + verified-k=1 accuracy vs number of layers.

CSV columns: sentence, true_label, pred_label, margin, radius, time
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG ─────────────────────────────────────────────────────────
ROOT = "/home/srivatunix2025/DSC-291_Project"
LAYERS = [1, 2, 3, 4]
SEEDS = [1, 2, 3]

# Color/label spec for the three p values
P_SPECS = [
    ("inf", r"$\infty$", "#1f77b4"),  # blue
    (1,     "1",         "#d62728"),  # red
    (2,     "2",         "#2ca02c"),  # green
]

# ── PATH BUILDING ─────────────────────────────────────────────────
def folder_suffix_for_p(p):
    """p='inf' → '' (results_ConvLips/); p=1 → '_p1.0'; p=2 → '_p2.0'"""
    if p == "inf":
        return ""
    return f"_p{float(p)}"

def get_csv_path(dataset, p, n_layers, seed, kernel_size):
    suffix = folder_suffix_for_p(p)
    folder = (
        f"results_ConvLips{suffix}/"
        f"figures_s4_s5_{dataset}_valid1000_optSGD_schcyclic_reduce-sum_"
        f"nl{n_layers}_bs128_es150_hs100_ks{kernel_size}_lr100.0_lm0_mp0_"
        f"lips_reg0_steps_dec1000000_seed{seed}"
    )
    return os.path.join(ROOT, folder, "_results_lipslev.csv")

# ── ACCURACY COMPUTATION ──────────────────────────────────────────
def compute_accuracies(csv_path):
    """Returns (clean_acc%, verified_acc_k1%)."""
    df = pd.read_csv(csv_path)
    correct = df["pred_label"] == df["true_label"]
    clean_acc = correct.mean() * 100
    verified_k1 = (correct & (df["radius"] >= 1)).mean() * 100
    return clean_acc, verified_k1

def aggregate_seeds(dataset, p, n_layers, kernel_size, seeds=SEEDS):
    """Mean/std of (clean, verified) across seeds. Returns NaN if all missing."""
    cleans, verifieds = [], []
    for seed in seeds:
        path = get_csv_path(dataset, p, n_layers, seed, kernel_size)
        if not os.path.exists(path):
            print(f"  ⚠ missing: {dataset} p={p} nl={n_layers} seed={seed}")
            continue
        c, v = compute_accuracies(path)
        cleans.append(c)
        verifieds.append(v)
    if not cleans:
        return np.nan, np.nan, np.nan, np.nan
    return np.mean(cleans), np.std(cleans), np.mean(verifieds), np.std(verifieds)

# ── PLOTTING ──────────────────────────────────────────────────────
def plot_figure(dataset, kernel_size, save_path, fig_label):
    print(f"\n=== {fig_label}: {dataset} (kernel_size={kernel_size}) ===")

    # Collect all data
    data = {}
    for p, _, _ in P_SPECS:
        clean_m, clean_s, ver_m, ver_s = [], [], [], []
        for nl in LAYERS:
            cm, cs, vm, vs = aggregate_seeds(dataset, p, nl, kernel_size)
            clean_m.append(cm); clean_s.append(cs)
            ver_m.append(vm);   ver_s.append(vs)
        data[p] = tuple(np.array(x) for x in (clean_m, clean_s, ver_m, ver_s))
        print(f"  p={p:>3} | clean={[f'{x:.1f}' for x in clean_m]} "
              f"| verified={[f'{x:.1f}' for x in ver_m]}")

    # Paper-matching style (serif font, Computer Modern math)
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset":  "cm",
        "axes.labelsize":    14,
        "axes.titlesize":    15,
        "xtick.labelsize":   12,
        "ytick.labelsize":   12,
        "legend.fontsize":   13,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for p, label, color in P_SPECS:
        cm, cs, vm, vs = data[p]
        # Clean (left)
        axes[0].plot(LAYERS, cm, color=color, linewidth=2.0)
        axes[0].fill_between(LAYERS, cm - cs, cm + cs,
                             color=color, alpha=0.25, linewidth=0)
        # Verified k=1 (right)
        axes[1].plot(LAYERS, vm, color=color, linewidth=2.0, label=label)
        axes[1].fill_between(LAYERS, vm - vs, vm + vs,
                             color=color, alpha=0.25, linewidth=0)

    axes[0].set_title("Clean")
    axes[1].set_title(r"Verified ($k=1$)")
    for ax in axes:
        ax.set_xlabel("Layers")
        ax.set_ylabel("Acc.")
        ax.set_xticks(LAYERS)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    # Legend in right plot, with black thin border (matches paper)
    legend = axes[1].legend(loc="best", frameon=True, edgecolor="black",
                            framealpha=1.0, fancybox=False)
    legend.get_frame().set_linewidth(0.6)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = save_path.replace(".pdf", f".{ext}")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  saved → {out}")
    plt.close()

# ── MAIN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_figure(
        dataset="ag_news", kernel_size=10,
        save_path="figure_s4_ag_news.pdf",
        fig_label="Figure S4 (AG-News)",
    )
    plot_figure(
        dataset="sst2", kernel_size=5,
        save_path="figure_s5_sst2.pdf",
        fig_label="Figure S5 (SST-2)",
    )
    print("\n✓ Done.")