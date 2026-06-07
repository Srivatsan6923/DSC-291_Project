import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec

GREEN = "#2ca02c"
RED = "#d62728"

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "cm",
})

def load_results():
    """Main results dirs, nl1, single seed, correctly-classified only — matches the paper."""
    rows = []
    for f in Path(".").rglob("_results_*.csv"):
        path = str(f)
        exp  = f.parent.name

        if "results_ConvLips_p2.0" not in path: continue
        if not exp.startswith("results_"):  continue   
        if "_seed1" not in exp:             continue
        if "lr100.0" not in path:           continue
        if "ag_news" not in path and "sst2" not in path: continue

        if   "bruteforce" in path: method = "BruteF"
        elif "lipslev"    in path: method = "LipsLev"
        else: continue

        dataset = "AG-News" if "ag_news" in path else "SST-2"

        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "pred_label" not in df.columns or len(df) == 0:
            continue

        df["sentence"] = df["sentence"].astype(str)
        df["length"]   = df["sentence"].str.len()
        df["correct"]  = df["true_label"] == df["pred_label"]

        df = df[df["correct"]].copy()                     
        df["verified"] = df["radius"].fillna(0) >= 1      
                                                          

        df["dataset"] = dataset
        df["method"]  = method
        rows.append(df[["dataset", "method", "length", "verified"]])

    out = pd.concat(rows, ignore_index=True)
    print(out.groupby(["dataset", "method"]).size())  
    return out

all_df = load_results()
# ────────────────────────────────────────────────────────────
# FIGURE 1 — Ver. Acc. vs sentence length + average-length table
# ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 4))
gs = GridSpec(1, 3, width_ratios=[1.1, 1.1, 1.5], wspace=0.42)

plot_cfg = {
    "AG-News": dict(lo=80, hi=520, nbins=15, ylim=(0, 1.05)),
    "SST-2":   dict(lo=0,  hi=240, nbins=15, ylim=(0, 0.70)),
}

for i, dataset in enumerate(["AG-News", "SST-2"]):
    ax = fig.add_subplot(gs[0, i])
    cfg = plot_cfg[dataset]
    sub = all_df[(all_df.dataset == dataset) & (all_df.method == "LipsLev")].copy()
    sub = sub[(sub["length"] >= cfg["lo"]) & (sub["length"] <= cfg["hi"])]

    bins = np.linspace(cfg["lo"], cfg["hi"], cfg["nbins"] + 1)
    sub["bin"] = pd.cut(sub["length"], bins=bins, include_lowest=True)
    plot_df = sub.groupby("bin", observed=True)["verified"].mean().reset_index()
    plot_df["mid"]   = plot_df["bin"].apply(lambda x: x.mid).astype(float)
    plot_df["width"] = plot_df["bin"].apply(lambda x: x.right - x.left).astype(float)

    ax.bar(plot_df["mid"], plot_df["verified"], width=plot_df["width"] * 1.01,
           color=GREEN, edgecolor="none", align="center")           # *1.01 → bars touch
    ax.set_title(dataset, fontsize=16)
    ax.set_xlabel("Sentence length", fontsize=13)
    ax.set_ylabel("Ver. Acc.", fontsize=13)
    ax.set_ylim(*cfg["ylim"])
    ax.set_xlim(cfg["lo"], cfg["hi"])
    ax.tick_params(labelsize=11)

# Table (right) — averages from unfiltered (within-config) data
avg = (
    all_df.groupby(["dataset", "method", "verified"])["length"]
    .mean()
    .reset_index()
)
def val(dataset, method, verified):
    x = avg[(avg.dataset == dataset) &
            (avg.method == method) &
            (avg.verified == verified)]["length"]
    return float(x.iloc[0]) if len(x) else float("nan")

ax_tbl = fig.add_subplot(gs[0, 2])
ax_tbl.set_xlim(0, 1); ax_tbl.set_ylim(0, 1); ax_tbl.axis("off")

x_ds, x_m, x_no, x_yes = 0.05, 0.32, 0.66, 0.86
y = dict(top_rule=0.95, avg=0.88, under_avg=0.82, ver=0.76, under_ver=0.70,
         col=0.62, under_col=0.55,
         r1=0.45, r2=0.36, mid_rule=0.30, r3=0.22, r4=0.13, bot_rule=0.07)

ax_tbl.plot([0, 1], [y["top_rule"]]*2, color="black", lw=1.0)
ax_tbl.text(0.5, y["avg"], "Average Length", ha="center", va="center", fontsize=13)
ax_tbl.plot([0, 1], [y["under_avg"]]*2, color="black", lw=0.5)
ax_tbl.text((x_no + x_yes)/2, y["ver"], "Verified", ha="center", va="center", fontsize=12)
ax_tbl.plot([x_no - 0.07, x_yes + 0.07], [y["under_ver"]]*2, color="black", lw=0.5)
ax_tbl.text(x_ds,  y["col"], "Dataset", ha="left",   va="center", fontsize=12)
ax_tbl.text(x_m,   y["col"], "Method",  ha="left",   va="center", fontsize=12)
ax_tbl.text(x_no,  y["col"], "✗", ha="center", va="center", fontsize=14, color=RED)
ax_tbl.text(x_yes, y["col"], "✓", ha="center", va="center", fontsize=14, color=GREEN)
ax_tbl.plot([0, 1], [y["under_col"]]*2, color="black", lw=0.6)

rows = [
    (y["r1"], "AG-News", "BruteF",  val("AG-News", "BruteF",  False), val("AG-News", "BruteF",  True)),
    (y["r2"], "",        "LipsLev", val("AG-News", "LipsLev", False), val("AG-News", "LipsLev", True)),
    (y["r3"], "SST-2",   "BruteF",  val("SST-2",   "BruteF",  False), val("SST-2",   "BruteF",  True)),
    (y["r4"], "",        "LipsLev", val("SST-2",   "LipsLev", False), val("SST-2",   "LipsLev", True)),
]
for yi, ds, m, no, yes in rows:
    ax_tbl.text(x_ds, yi, ds, ha="left", va="center", fontsize=11)
    ax_tbl.text(x_m,  yi, m,  ha="left", va="center", fontsize=11, family="monospace")
    no_s  = f"{no:.1f}"  if not np.isnan(no)  else "—"
    yes_s = f"{yes:.1f}" if not np.isnan(yes) else "—"
    ax_tbl.text(x_no,  yi, no_s,  ha="center", va="center", fontsize=11)
    ax_tbl.text(x_yes, yi, yes_s, ha="center", va="center", fontsize=11)

ax_tbl.plot([0, 1], [y["mid_rule"]]*2, color="gray", lw=0.4)
ax_tbl.plot([0, 1], [y["bot_rule"]]*2, color="black", lw=1.0)

fig.savefig("figure1_reproduction.png", dpi=300, bbox_inches="tight")
fig.savefig("figure1_reproduction.pdf",            bbox_inches="tight")
plt.close(fig)
print("✓ figure1_reproduction.png/pdf saved")

# ────────────────────────────────────────────────────────────
# FIGURE S2 — side-by-side histograms (absolute counts)
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
fig.subplots_adjust(wspace=0.32)

hist_cfg = {
    "AG-News": dict(lo=80, hi=850, nbins=20),
    "SST-2":   dict(lo=0,  hi=250, nbins=25),
}

for ax, dataset in zip(axes, ["AG-News", "SST-2"]):
    cfg = hist_cfg[dataset]
    sub = all_df[(all_df.dataset == dataset) & (all_df.method == "LipsLev")].copy()
    sub = sub[(sub["length"] >= cfg["lo"]) & (sub["length"] <= cfg["hi"])]

    edges   = np.linspace(cfg["lo"], cfg["hi"], cfg["nbins"] + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bw      = edges[1] - edges[0]

    not_v, _ = np.histogram(sub[~sub.verified]["length"], bins=edges)
    yes_v, _ = np.histogram(sub[ sub.verified]["length"], bins=edges)

    dodge = bw * 0.42
    ax.bar(centers - dodge/2, not_v, width=dodge, color=RED,   label="Not verified", edgecolor="none")
    ax.bar(centers + dodge/2, yes_v, width=dodge, color=GREEN, label="Verified",     edgecolor="none")

    ax.set_title(dataset, fontsize=16)
    ax.set_xlabel("Sentence length", fontsize=13)
    ax.set_ylabel("Freq.", fontsize=13)
    ax.set_xlim(cfg["lo"], cfg["hi"])
    ax.tick_params(labelsize=11)

    legend = ax.legend(loc="upper right", frameon=True, edgecolor="black",
                       framealpha=1.0, fancybox=False, fontsize=11)
    legend.get_frame().set_linewidth(0.6)

fig.savefig("figureS2_reproduction.png", dpi=300, bbox_inches="tight")
fig.savefig("figureS2_reproduction.pdf",            bbox_inches="tight")
plt.close(fig)
print("✓ figureS2_reproduction.png/pdf saved")