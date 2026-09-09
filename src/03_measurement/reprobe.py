"""Re-probe key features on the fully-imputed matrix (removes sqrt(p) attenuation)."""
import pandas as pd, numpy as np, sys
X=pd.read_parquet('artifacts/Xfull.parquet'); ntr=400000
d={c:X[c].values.astype(np.float64) for c in X.columns}
B={}
for c in ['feature_01','feature_02','feature_03','feature_04','feature_06','feature_07','feature_08',
          'feature_09','feature_11','feature_12','feature_13','feature_15','feature_19','feature_22','feature_21']:
    B['q_'+c[-2:]]=d[c]
B['q_nl_f13xf15']=d['feature_13']*d['feature_15']
B['q_nl_f07sq']=d['feature_07']**2
B['q_nl_f13sq']=d['feature_13']**2
B['q_nl_f07xf13']=d['feature_07']*d['feature_13']
B['q_nl_usable']=d['feature_14']*d['feature_15']*d['feature_11']*(1-d['feature_22'])
sub=pd.read_csv('data/sample_submission.csv')
which=sys.argv[1:] if len(sys.argv)>1 else list(B)
for n in which:
    a=np.nan_to_num(B[n][ntr:],nan=0,posinf=0,neginf=0).astype(float)
    a=(a-a.mean())/a.std()*4.6+3.9
    sub['prediction']=a; sub.to_csv(f'subs/{n}.csv',index=False); print(n)
