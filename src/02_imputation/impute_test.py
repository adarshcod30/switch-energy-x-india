import pandas as pd, numpy as np, lightgbm as lgb, time
from sklearn.model_selection import KFold
tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv')
full=pd.concat([tr,te],ignore_index=True).reset_index(drop=True)
F=[f'feature_{i:02d}' for i in range(1,26)]
X=full[F].copy()
# physics-derived helper columns (all from observed cols, NaN-safe)
X['d_v_from18']=np.sqrt(np.clip(full.feature_18,0,None))
X['d_G_from17']=full.feature_17/(full.feature_03/100)
X['d_Tp_from15']=(1.10421-full.feature_15)/0.00418073
X['d_gross_from21']=(full.feature_21-49.9867)/0.899958+full.feature_09
X['d_gross_from21b']=(full.feature_21-49.402)/0.90027
X['d_sum1213']=full.feature_12+full.feature_13
X['d_rho']=full.feature_05*100/(287.05*(full.feature_02+273.15))
X['n_obs']=full[F].notna().sum(axis=1)
for c in F: X[f'm_{c}']=full[c].isna().astype(np.int8)
COLS=list(X.columns)
def oof_r2(target, drop, n=200000, seed=0):
    """OOF R2 for predicting `target` from all other cols."""
    obs=np.where(full[target].notna().values)[0]
    rng=np.random.default_rng(seed); obs=rng.permutation(obs)[:n]
    feats=[c for c in COLS if c not in drop]
    Xs=X.iloc[obs][feats].values.astype(np.float32); ys=full[target].values[obs]
    kf=KFold(3,shuffle=True,random_state=0); oof=np.zeros(len(ys))
    for a,b in kf.split(Xs):
        m=lgb.train(dict(objective='l2',learning_rate=0.08,num_leaves=127,min_data_in_leaf=40,
                         feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=8),
                    lgb.Dataset(Xs[a],ys[a]),num_boost_round=400)
        oof[b]=m.predict(Xs[b])
    r=ys-oof; return 1-r.var()/ys.var(), r.std(), ys.std()
t0=time.time()
for tgt in ['feature_14','feature_15','feature_11','feature_22','feature_09','feature_08','feature_19','feature_12','feature_13']:
    drop=[tgt,f'm_{tgt}']
    if tgt=='feature_14': drop+=['d_sum1213']          # keep it honest? no -> keep, it's legit
    drop=[tgt,f'm_{tgt}']
    r2,sd,ysd=oof_r2(tgt,drop)
    print(f"  impute {tgt}: OOF R2={r2:.5f}  resid_sd={sd:.5f}  (col sd={ysd:.4f})   [{time.time()-t0:.0f}s]")
