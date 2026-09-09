import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
dd = pd.read_csv('data/data_dictionary.csv')
F = [c for c in tr.columns if c.startswith('feature_')]
print("train", tr.shape, "test", te.shape)
desc = dict(zip(dd.feature, dd.category + " | " + dd.description))
rows=[]
for c in F:
    s = tr[c]
    rows.append(dict(feat=c, desc=desc[c], miss=s.isna().mean(),
        min=s.min(), q01=s.quantile(.01), q25=s.quantile(.25), med=s.median(),
        q75=s.quantile(.75), q99=s.quantile(.99), max=s.max(),
        mean=s.mean(), std=s.std(), skew=s.skew(), nuniq=s.nunique()))
prof = pd.DataFrame(rows)
print(prof.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
print("\n=== overall missing frac (train):", tr[F].isna().mean().mean())
print("=== overall missing frac (test):", te[F].isna().mean().mean())
print("=== observed-per-row distribution (train):")
print(tr[F].notna().sum(axis=1).value_counts().sort_index().to_string())
prof.to_csv('reports/profile_train.csv', index=False)
