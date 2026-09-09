import pandas as pd, numpy as np, sys
X=pd.read_parquet('artifacts/Xv2.parquet'); ntr=400000
d={c:X[c].values.astype(np.float64) for c in X.columns}
v3=np.clip(d['feature_18'],0,None)**1.5
B={'w_f10':d['feature_10'],'w_f17':d['feature_17'],'w_f18':d['feature_18'],
   'w_rhov3':d['feature_10']*v3,'w_f16':d['feature_16'],'w_f05':d['feature_05'],
   'w_f14':d['feature_14'],'w_v3':v3}
sub=pd.read_csv('data/sample_submission.csv')
for n in sys.argv[1:]:
    a=np.nan_to_num(B[n][ntr:]); a=(a-a.mean())/a.std()*4.6+3.9
    sub['prediction']=a; sub.to_csv(f'subs/{n}.csv',index=False); print(n)
