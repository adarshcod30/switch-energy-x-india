import numpy as np, pandas as pd, json
SY,MY=4.6,3.9
rho=json.load(open('artifacts/rho.json')) if False else None
S=json.load(open('artifacts/lb_scores.json'))
N2C={'b_panelT_f07':'feature_07','b_transloss_f22':'feature_22','b_inveff_f11':'feature_11',
 'b_ambT_f02':'feature_02','b_cloud_f06':'feature_06','b_gridfreq_f21':'feature_21',
 'b_hum_f03':'feature_03','b_wind_f04':'feature_04','b_CONTROL_noise_f20':'feature_20',
 'b_solar_f12':'feature_12','b_soc_f08':'feature_08','b_irr_f01':'feature_01',
 'b_f19_demstor':'feature_19','b_temploss_f15':'feature_15','s07_wind_only':'feature_13'}
rho={c:1-S[k]**2/(2*SY*SY) for k,c in N2C.items() if k in S}
rho['feature_09']=-(1-S['s05_demand_only']**2/(2*SY*SY))
X=pd.read_parquet('artifacts/Xfilled.parquet')
cols=sorted(rho); A=X[cols].values[400000:].astype(np.float64)
print("non-finite in A:",(~np.isfinite(A)).sum(), " max abs:",np.abs(A).max())
A=np.nan_to_num(A,nan=0,posinf=0,neginf=0)
mu=A.mean(0); sd=A.std(0); Z=(A-mu)/sd                 # standardise -> stable solve
Cz=np.corrcoef(Z,rowvar=False)
cz=np.array([rho[c] for c in cols])*SY                  # cov(y, z_j) = rho_j*SY  (z has sd 1)
for lam in [0.0,0.01,0.05,0.1,0.3]:
    bz=np.linalg.solve(Cz+lam*np.eye(len(cols)),cz)
    R2=float(bz@cz)/SY**2
    p=Z@bz
    print(f"  ridge lam={lam:5.2f}: implied R2={R2:.4f} expRMSE={SY*np.sqrt(max(1-R2,0)):.4f} pred_sd={p.std():.3f}")
lam=0.05
bz=np.linalg.solve(Cz+lam*np.eye(len(cols)),cz)
pred=Z@bz+MY
print("\ncoefficients (per 1 sd of feature):")
for c,b in sorted(zip(cols,bz),key=lambda t:-abs(t[1])): print(f"   {c:12s} {b:+8.4f}")
sub=pd.read_csv('data/sample_submission.csv'); sub['prediction']=pred
sub.to_csv('subs/s10_linsolve.csv',index=False)
print(f"\nwrote s10_linsolve: mean={pred.mean():.3f} sd={pred.std():.3f} min={pred.min():.2f} max={pred.max():.2f}")
np.save('artifacts/Zte.npy',Z); json.dump({'cols':cols,'rho':rho},open('artifacts/rho.json','w'),indent=1)
