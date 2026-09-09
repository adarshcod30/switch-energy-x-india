"""SWITCH ENERGY-X: physics reconstruction of NDEM.
Stage 1: self-supervised imputation of every feature (train+test, no labels used).
Stage 2: plug imputed values into the recovered physics equation.
"""
import pandas as pd, numpy as np, lightgbm as lgb, os, json, time
os.makedirs('artifacts',exist_ok=True); os.makedirs('subs',exist_ok=True)
F=[f'feature_{i:02d}' for i in range(1,26)]
tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv')
ntr=len(tr)
full=pd.concat([tr,te],ignore_index=True).reset_index(drop=True)

def build_X(full):
    X=full[F].copy()
    X['d_v18']=np.sqrt(np.clip(full.feature_18,0,None))
    X['d_G17']=full.feature_17/(full.feature_03/100)
    X['d_Tp15']=(1.10421-full.feature_15)/0.00418073
    X['d_gross21']=(full.feature_21-49.9867)/0.899958+full.feature_09
    X['d_gross21b']=(full.feature_21-49.402)/0.90027
    X['d_1213']=full.feature_12+full.feature_13
    X['d_rho']=full.feature_05*100/(287.05*(full.feature_02+273.15))
    X['d_Tp_phys']=full.feature_02+0.0219*full.feature_01-0.348*full.feature_04
    X['d_v3']=np.clip(full.feature_04,0,None)**3
    X['d_v3b']=np.clip(full.feature_18,0,None)**1.5
    X['d_G1mC']=full.feature_01*(1-full.feature_06)
    X['d_ds']=full.feature_09*full.feature_08
    X['n_obs']=full[F].notna().sum(axis=1)
    for c in F: X[f'm_{c}']=full[c].isna().astype(np.int8)
    return X
X=build_X(full)
COLS=list(X.columns)
PARAMS=dict(objective='l2',learning_rate=0.05,num_leaves=255,min_data_in_leaf=40,
            feature_fraction=0.85,bagging_fraction=0.85,bagging_freq=1,verbose=-1,
            num_threads=8,lambda_l2=1.0)
NEED=['feature_14','feature_15','feature_11','feature_22','feature_09','feature_08',
      'feature_19','feature_12','feature_13','feature_01','feature_21']
filled={}
t0=time.time()
for tgt in NEED:
    obs=full[tgt].notna().values; mis=~obs
    feats=[c for c in COLS if c not in (tgt,f'm_{tgt}')]
    v=full[tgt].values.copy()
    if mis.sum()>0:
        ds=lgb.Dataset(X.loc[obs,feats].values.astype(np.float32), v[obs])
        m=lgb.train(PARAMS,ds,num_boost_round=700)
        v[mis]=m.predict(X.loc[mis,feats].values.astype(np.float32))
    filled[tgt]=v
    print(f"  filled {tgt}: {mis.sum():6d} missing  [{time.time()-t0:.0f}s]",flush=True)
imp=pd.DataFrame(filled)
imp.to_parquet('artifacts/imputed.parquet')
print("saved artifacts/imputed.parquet", imp.shape)

G=imp.feature_14.values; TL=imp.feature_15.values; IE=imp.feature_11.values
XL=imp.feature_22.values; DS=imp.feature_19.values; DE=imp.feature_09.values; SOC=imp.feature_08.values
usable=G*TL*IE*(1-XL)
np.save('artifacts/usable.npy',usable)
for nm,arr in [('usable',usable),('usable-0.25f19',usable-0.25*DS),('usable-f19',usable-DS),('usable-f09',usable-DE)]:
    a=arr[ntr:]
    print(f"  {nm:16s} TEST: mean={a.mean():.4f} std={a.std():.4f} min={a.min():.3f} max={a.max():.3f}")
sub=pd.read_csv('data/sample_submission.csv')
sub['prediction']=usable[ntr:]
sub.to_csv('subs/s01_usable_raw.csv',index=False)
print("wrote subs/s01_usable_raw.csv"); print(sub.head())
