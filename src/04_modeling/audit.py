"""Where does the error come from? (a) moment residuals -> wrong formula, (b) imputation error."""
import numpy as np, pandas as pd, json, lightgbm as lgb
from sklearn.model_selection import KFold
SY=4.611
RHO=json.load(open('artifacts/RHO_full.json'))
X=pd.read_parquet('artifacts/Xv2.parquet'); Xte=X.iloc[400000:].reset_index(drop=True)
cols=sorted(RHO); P=Xte[cols].values.astype(np.float64); sd=P.std(0)
Z=(P-P.mean(0))/sd; Cz=np.corrcoef(Z,rowvar=False); cz=np.array([RHO[c] for c in cols])*SY
bz=np.linalg.solve(Cz+0.01*np.eye(len(cols)),cz)
implied=Cz@bz                      # cov(p, z_j)
print("(a) MOMENT RESIDUALS  (measured cov(y,z) vs model-implied cov(p,z)); big |resid| = missing structure")
print(f"{'feature':13s} {'measured':>10s} {'implied':>10s} {'resid':>9s}")
for i,c in enumerate(cols):
    print(f"  {c:11s} {cz[i]:10.4f} {implied[i]:10.4f} {cz[i]-implied[i]:+9.4f}")
print(f"  ||resid||/||meas|| = {np.linalg.norm(cz-implied)/np.linalg.norm(cz):.4f}")

print("\n(b) IMPUTATION ERROR BUDGET (OOF resid on observed rows, weighted by beta and missing frac)")
tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv')
full=pd.concat([tr,te],ignore_index=True).reset_index(drop=True)
F=[f'feature_{i:02d}' for i in range(1,26)]
raw_beta=bz/sd
tot=0.0; rows=[]
Xall=X.values.astype(np.float32); names=list(X.columns)
for i,c in enumerate(cols):
    o=full[c].notna().values
    mfrac=1-o.mean()
    idx=np.where(o)[0]; rng=np.random.default_rng(0); idx=rng.permutation(idx)[:60000]
    feats=[j for j,n in enumerate(names) if n!=c]
    A=Xall[np.ix_(idx,feats)]; y=full[c].values[idx]
    kf=KFold(3,shuffle=True,random_state=0); oof=np.zeros(len(y))
    for a,b in kf.split(A):
        mdl=lgb.train(dict(objective='l2',learning_rate=0.08,num_leaves=127,verbose=-1,num_threads=8,
                           feature_fraction=.8,bagging_fraction=.8,bagging_freq=1),
                      lgb.Dataset(A[a],y[a]),num_boost_round=250)
        oof[b]=mdl.predict(A[b])
    e=(y-oof).std()
    contrib=(raw_beta[i]**2)*mfrac*(e**2)
    tot+=contrib; rows.append((c,raw_beta[i],mfrac,e,np.sqrt(contrib)))
for c,b,mf,e,ct in sorted(rows,key=lambda t:-t[4]):
    print(f"  {c:11s} beta={b:+9.4f} miss={mf:.3f} imp_sd={e:8.4f}  -> RMSE contrib={ct:.4f}")
print(f"  TOTAL imputation-driven RMSE = {np.sqrt(tot):.4f}")
