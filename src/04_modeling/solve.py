"""Solve best linear predictor from LB-measured correlations:  beta = Cov(X)^-1 cov(X,y)."""
import subprocess, re, numpy as np, pandas as pd, json, sys, os
SY,MY=4.6,3.9
NAME2COL={'b_panelT_f07':'feature_07','b_transloss_f22':'feature_22','b_inveff_f11':'feature_11',
 'b_ambT_f02':'feature_02','b_cloud_f06':'feature_06','b_gridfreq_f21':'feature_21',
 'b_hum_f03':'feature_03','b_wind_f04':'feature_04','b_CONTROL_noise_f20':'feature_20',
 'b_solar_f12':'feature_12','b_soc_f08':'feature_08','b_irr_f01':'feature_01',
 'b_f19_demstor':'feature_19','b_temploss_f15':'feature_15','s07_wind_only':'feature_13'}
def lb():
    out=subprocess.run(['python3','-m','kaggle','competitions','submissions','-c','switch-energy-x-india'],
                       capture_output=True,text=True).stdout
    r={}
    for ln in out.splitlines()[2:]:
        m=re.match(r'\s*(\d+)\s+(\S+?)\.csv\s+([\d\-]+ [\d:.]+)\s+(.*?)\s+SubmissionStatus\.(\w+)\s+([\d.]+)?',ln)
        if m and m.group(6): r.setdefault(m.group(2),float(m.group(6)))
    return r
S=lb(); json.dump(S,open('artifacts/lb_scores.json','w'),indent=1)
print(f"{'submission':26s} {'RMSE':>8s} {'rho':>8s}")
rho={}
for k,v in sorted(S.items()):
    r=1-v*v/(2*SY*SY); print(f"  {k:24s} {v:8.5f} {r:8.4f}")
    if k in NAME2COL: rho[NAME2COL[k]]=r
# f09 sign flip: s05 was -f09
if 's05_demand_only' in S: rho['feature_09']=-(1-S['s05_demand_only']**2/(2*SY*SY))
print("\nrho by column:",{k:round(v,4) for k,v in sorted(rho.items())})
X=pd.read_parquet('artifacts/Xfilled.parquet'); Xte=X.iloc[400000:]
cols=sorted(rho)
A=Xte[cols].values.astype(float)
C=np.cov(A,rowvar=False); sd=A.std(0)
c=np.array([rho[k]*SY*sd[i] for i,k in enumerate(cols)])
beta=np.linalg.solve(C+1e-9*np.eye(len(cols)),c)
pred=A@beta; b0=MY-pred.mean()
R2=float(beta@c/SY**2)
print(f"\nMultivariate solve on {len(cols)} features:")
for k,b in zip(cols,beta): print(f"   {k}: beta={b:+.5f}   (contrib sd={abs(b)*sd[cols.index(k)]:.3f})")
print(f"  implied R2={R2:.4f}  -> expected RMSE={SY*np.sqrt(max(1-R2,0)):.4f}")
np.save('artifacts/lin_pred.npy',pred+b0)
print(f"  pred: mean={(pred+b0).mean():.3f} sd={pred.std():.3f}")
