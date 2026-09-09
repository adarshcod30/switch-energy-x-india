"""Full self-supervised GBM imputation of all 25 features, 2 iterative passes."""
import pandas as pd, numpy as np, lightgbm as lgb, time
F=[f'feature_{i:02d}' for i in range(1,26)]
tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv')
full=pd.concat([tr,te],ignore_index=True).reset_index(drop=True)
obs_mask={c:full[c].notna().values for c in F}
def derived(Z):
    D=pd.DataFrame(index=Z.index)
    D['d_v18']=np.sqrt(np.clip(Z.feature_18,0,None))
    D['d_G17']=Z.feature_17/np.clip(Z.feature_03/100,.05,None)
    D['d_Tp15']=(1.10421-Z.feature_15)/0.00418073
    D['d_gross21']=(Z.feature_21-49.9867)/0.899958+Z.feature_09
    D['d_1213']=Z.feature_12+Z.feature_13
    D['d_14m13']=Z.feature_14-Z.feature_13
    D['d_14m12']=Z.feature_14-Z.feature_12
    D['d_rho']=Z.feature_05*100/(287.05*(Z.feature_02+273.15))
    D['d_Tphys']=Z.feature_02+0.0219*Z.feature_01-0.348*Z.feature_04
    D['d_v3']=np.clip(Z.feature_04,0,None)**3
    D['d_G1mC']=Z.feature_01*(1-Z.feature_06)
    D['d_ds']=Z.feature_09*Z.feature_08
    return D
MASK=pd.DataFrame({f'm_{c}':full[c].isna().astype(np.int8) for c in F})
MASK['n_obs']=full[F].notna().sum(axis=1)
Z=full[F].copy()
med=Z.median()
PARAMS=dict(objective='l2',learning_rate=0.06,num_leaves=255,min_data_in_leaf=40,
            feature_fraction=0.85,bagging_fraction=0.85,bagging_freq=1,verbose=-1,
            num_threads=8,lambda_l2=1.0)
t0=time.time()
for it in range(2):
    Zf=Z.fillna(med) if it==0 else Z.copy()
    for c in F:
        o=obs_mask[c]; m=~o
        if m.sum()==0: continue
        base=Zf.drop(columns=[c])
        Xall=pd.concat([base,derived(Zf),MASK.drop(columns=[f'm_{c}'])],axis=1).values.astype(np.float32)
        mdl=lgb.train(PARAMS,lgb.Dataset(Xall[o],full[c].values[o]),num_boost_round=600)
        Z.loc[m,c]=mdl.predict(Xall[m])
        Z.loc[o,c]=full[c].values[o]
    print(f"  pass {it+1} done [{time.time()-t0:.0f}s]",flush=True)
Z=Z.astype(np.float64)
out=pd.concat([Z,derived(Z)],axis=1)
out.to_parquet('artifacts/Xfull.parquet')
print("saved artifacts/Xfull.parquet",out.shape,"NaNs:",int(out.isna().sum().sum()))
