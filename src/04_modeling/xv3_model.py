"""Best linear model rebuilt on Xv3 (phase-aware imputation), using measured rho."""
import numpy as np, pandas as pd, json
SY,MY=4.611,3.9
RHO=json.load(open('artifacts/RHO_full.json'))
X=pd.read_parquet('artifacts/Xv3.parquet').iloc[400000:].reset_index(drop=True)
cols=sorted(RHO); P=X[cols].values.astype(np.float64)
sd=P.std(0); Z=(P-P.mean(0))/sd
Cz=np.corrcoef(Z,rowvar=False); cz=np.array([RHO[c] for c in cols])*SY
sub=pd.read_csv('data/sample_submission.csv')
for lam,nm in [(0.01,'V3_lin01'),(0.03,'V3_lin03')]:
    bz=np.linalg.solve(Cz+lam*np.eye(len(cols)),cz); p=Z@bz+MY
    sub['prediction']=p; sub.to_csv(f'subs/{nm}.csv',index=False)
    print(f"  {nm}: sd={p.std():.3f} min={p.min():.2f} max={p.max():.2f}")
old=pd.read_parquet('artifacts/Xv2.parquet').iloc[400000:].reset_index(drop=True)
print("\n Xv2 vs Xv3 correlation on re-imputed cols:")
for c in ['feature_08','feature_09','feature_11','feature_19']:
    print(f"   {c}: corr={np.corrcoef(X[c],old[c])[0,1]:.5f}")
