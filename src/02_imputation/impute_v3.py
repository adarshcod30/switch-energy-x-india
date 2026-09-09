"""Xv3 = Xv2 with phase-aware re-imputation of the periodic columns (f08,f09,f11,f19)."""
import pandas as pd, numpy as np, lightgbm as lgb, time
from sklearn.model_selection import KFold
F=[f'feature_{i:02d}' for i in range(1,26)]
tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv')
R=pd.concat([tr,te],ignore_index=True).reset_index(drop=True)[F]
n=len(R); rid=np.arange(n,dtype=float)
PH={}
for nm,P in [('p08',83333.3),('p09',50000.0),('p24',250000.0),('p250',250000.0/2)]:
    PH[f'{nm}_s']=np.sin(2*np.pi*rid/P); PH[f'{nm}_c']=np.cos(2*np.pi*rid/P)
PH=pd.DataFrame(PH)
X2=pd.read_parquet('artifacts/Xv2.parquet')
MASK=pd.DataFrame({f'm_{c}':R[c].isna().astype(np.int8) for c in F}); MASK['n_obs']=R.notna().sum(axis=1)
PARAMS=dict(objective='l2',learning_rate=0.04,num_leaves=255,min_data_in_leaf=30,
            feature_fraction=0.85,bagging_fraction=0.85,bagging_freq=1,verbose=-1,num_threads=8,lambda_l2=1.0)
Z=X2[F].copy(); t0=time.time()
print(f"{'col':12s} {'OOF R2 (no phase)':>18s} {'OOF R2 (+phase)':>16s}")
for c in ['feature_08','feature_09','feature_11','feature_19']:
    o=R[c].notna().values; m=~o
    base=pd.concat([R.drop(columns=[c]),MASK.drop(columns=[f'm_{c}'])],axis=1)
    A0=base.values.astype(np.float32)
    A1=pd.concat([base,PH],axis=1).values.astype(np.float32)
    y=R[c].values
    idx=np.where(o)[0]; rng=np.random.default_rng(0); idx=rng.permutation(idx)[:120000]
    r2s=[]
    for A in (A0,A1):
        kf=KFold(3,shuffle=True,random_state=0); oof=np.zeros(len(idx))
        for a,b in kf.split(idx):
            mdl=lgb.train(PARAMS,lgb.Dataset(A[idx[a]],y[idx[a]]),num_boost_round=400)
            oof[b]=mdl.predict(A[idx[b]])
        r=y[idx]-oof; r2s.append(1-r.var()/y[idx].var())
    print(f"  {c:10s} {r2s[0]:18.5f} {r2s[1]:16.5f}   gain={r2s[1]-r2s[0]:+.5f}")
    mdl=lgb.train(PARAMS,lgb.Dataset(A1[o],y[o]),num_boost_round=900)
    Z.loc[m,c]=mdl.predict(A1[m]); Z.loc[o,c]=y[o]
    print(f"     refilled {m.sum()} [{time.time()-t0:.0f}s]",flush=True)
Z=Z.astype(np.float64)
D2=pd.DataFrame({'d_rho':Z.feature_05*100/(287.05*(Z.feature_02+273.15)),
                 'd_usable':Z.feature_14*Z.feature_15*Z.feature_11*(1-Z.feature_22)})
out=pd.concat([Z,D2,PH],axis=1); out.to_parquet('artifacts/Xv3.parquet')
print("saved artifacts/Xv3.parquet",out.shape)
