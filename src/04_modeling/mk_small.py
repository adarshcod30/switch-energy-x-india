import numpy as np, pandas as pd, json
X=pd.read_parquet('artifacts/Xv2.parquet').iloc[400000:].reset_index(drop=True)
d={c:X[c].values.astype(np.float64) for c in X.columns}
P={'f13':d['feature_13'],'f14':d['feature_14'],'f07':d['feature_07'],'f01':d['feature_01'],
   'f02':d['feature_02'],'f03':d['feature_03'],'f13*f15':d['feature_13']*d['feature_15'],
   'f14*f15':d['feature_14']*d['feature_15'],'usable':d['d_usable'],'f21':d['feature_21']}
C={'s24_small5a':{'f07':-0.9915,'f01':0.01807,'f02':0.741,'f03':-0.0137,'f13*f15':0.5457},
   's25_small5b':{'f13':0.5151,'f07':-0.9855,'f01':0.01771,'f02':0.7249,'f03':-0.01372},
   's26_small5c':{'f07':-0.9892,'f01':0.01784,'f02':0.7406,'f03':-0.01293,'usable':0.622}}
sub=pd.read_csv('data/sample_submission.csv')
for nm,co in C.items():
    f=sum(b*P[t] for t,b in co.items()); p=f-f.mean()+3.9
    sub['prediction']=p; sub.to_csv(f'subs/{nm}.csv',index=False)
    print(f"{nm}: sd={p.std():.3f} min={p.min():.2f} max={p.max():.2f}")
