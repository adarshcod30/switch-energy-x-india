"""Leak-free single-pass GBM imputation. Derived cols computed from RAW (NaN-bearing) data;
any derived col depending on the target is dropped for that target."""
import pandas as pd, numpy as np, lightgbm as lgb, time
F=[f'feature_{i:02d}' for i in range(1,26)]
tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv')
full=pd.concat([tr,te],ignore_index=True).reset_index(drop=True)
R=full[F]
DERIV={  # name -> (values, set of source features)
 'd_v18'    :(np.sqrt(np.clip(R.feature_18,0,None)),                    {'feature_18'}),
 'd_G17'    :(R.feature_17/np.clip(R.feature_03/100,.05,None),          {'feature_17','feature_03'}),
 'd_Tp15'   :((1.10421-R.feature_15)/0.00418073,                        {'feature_15'}),
 'd_gross21':((R.feature_21-49.9867)/0.899958+R.feature_09,             {'feature_21','feature_09'}),
 'd_g21b'   :((R.feature_21-49.402)/0.90027,                            {'feature_21'}),
 'd_1213'   :(R.feature_12+R.feature_13,                                {'feature_12','feature_13'}),
 'd_14m13'  :(R.feature_14-R.feature_13,                                {'feature_14','feature_13'}),
 'd_14m12'  :(R.feature_14-R.feature_12,                                {'feature_14','feature_12'}),
 'd_rho'    :(R.feature_05*100/(287.05*(R.feature_02+273.15)),          {'feature_05','feature_02'}),
 'd_Tphys'  :(R.feature_02+0.0219*R.feature_01-0.348*R.feature_04,      {'feature_02','feature_01','feature_04'}),
 'd_v3'     :(np.clip(R.feature_04,0,None)**3,                          {'feature_04'}),
 'd_v3b'    :(np.clip(R.feature_18,0,None)**1.5,                        {'feature_18'}),
 'd_G1mC'   :(R.feature_01*(1-R.feature_06),                            {'feature_01','feature_06'}),
 'd_ds'     :(R.feature_09*R.feature_08,                                {'feature_09','feature_08'}),
 'd_TfromTp':(R.feature_07-0.0219*R.feature_01+0.348*R.feature_04,      {'feature_07','feature_01','feature_04'}),
}
DF=pd.DataFrame({k:v[0] for k,v in DERIV.items()})
MASK=pd.DataFrame({f'm_{c}':R[c].isna().astype(np.int8) for c in F}); MASK['n_obs']=R.notna().sum(axis=1)
PARAMS=dict(objective='l2',learning_rate=0.05,num_leaves=255,min_data_in_leaf=30,
            feature_fraction=0.85,bagging_fraction=0.85,bagging_freq=1,verbose=-1,
            num_threads=8,lambda_l2=1.0)
Z=R.copy(); t0=time.time()
for c in F:
    o=R[c].notna().values; m=~o
    if m.sum()==0: continue
    drop_d=[k for k,(v,src) in DERIV.items() if c in src]
    Xall=pd.concat([R.drop(columns=[c]), DF.drop(columns=drop_d), MASK.drop(columns=[f'm_{c}'])],axis=1)
    A=Xall.values.astype(np.float32)
    mdl=lgb.train(PARAMS,lgb.Dataset(A[o],R[c].values[o]),num_boost_round=800)
    Z.loc[m,c]=mdl.predict(A[m])
    print(f"  {c}: {m.sum():6d} filled, dropped_derived={drop_d} [{time.time()-t0:.0f}s]",flush=True)
Z=Z.astype(np.float64)
D2=pd.DataFrame({k:v for k,v in {
 'd_rho':Z.feature_05*100/(287.05*(Z.feature_02+273.15)),
 'd_usable':Z.feature_14*Z.feature_15*Z.feature_11*(1-Z.feature_22)}.items()})
pd.concat([Z,D2],axis=1).to_parquet('artifacts/Xv2.parquet')
print("saved artifacts/Xv2.parquet","NaNs:",int(Z.isna().sum().sum()))
