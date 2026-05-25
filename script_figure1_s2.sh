echo "Figure 1 + Figure S2 — Sentence length analysis: post-process Table 2 verify CSVs in results/ (no extra train/verify runs)."
python -c "
import pandas as pd, matplotlib.pyplot as plt, glob

for dataset in ['ag_news', 'sst2']:
    files = glob.glob(f'results/*{dataset}*p2*seed1*lipslev*.csv')
    df = pd.concat([pd.read_csv(f) for f in files])
    df['verified'] = df['certified_radius'] >= 1
    df.groupby(pd.cut(df['length'], bins=20))['verified'].mean().plot()
    plt.xlabel('Sentence length'); plt.ylabel('Verified Acc.')
    plt.title(dataset); plt.savefig(f'fig1_{dataset}.png'); plt.clf()
"
