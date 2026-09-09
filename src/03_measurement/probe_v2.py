import pandas as pd, numpy as np, sys
X=pd.read_parquet('artifacts/Xv2.parquet'); ntr=400000
sub=pd.read_csv('data/sample_submission.csv')
for f in sys.argv[1:]:
    a=X['feature_'+f].values[ntr:].astype(float)
    a=(a-a.mean())/a.std()*4.6+3.9
    sub['prediction']=a; sub.to_csv(f'subs/v2_{f}.csv',index=False); print('v2_'+f)
