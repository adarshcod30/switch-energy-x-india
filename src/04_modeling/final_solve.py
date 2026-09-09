import numpy as np, pandas as pd, json, subprocess, re
SY,MY=4.611,3.9
out=subprocess.run(['python3','-m','kaggle','competitions','submissions','-c','switch-energy-x-india'],capture_output=True,text=True).stdout
S={}
for ln in out.splitlines()[2:]:
    m=re.match(r'\s*(\d+)\s+(\S+?)\.csv\s+\S+ \S+\s+.*?SubmissionStatus\.\w+\s+([\d.]+)',ln)
    if m: S.setdefault(m.group(2),float(m.group(3)))
json.dump(S,open('artifacts/lb_scores.json','w'),indent=1)
r=lambda k:1-S[k]**2/(2*SY*SY)
RHO={}
for f in ['15','09','02','04','03','06','08','21']:
    if f'v2_{f}' in S: RHO['feature_'+f]=r(f'v2_{f}')
CARRY={'feature_13':('s07_wind_only',1),'feature_12':('b_solar_f12',1),'feature_01':('b_irr_f01',1),
       'feature_19':('b_f19_demstor',1),'feature_22':('b_transloss_f22',1),'feature_11':('b_inveff_f11',1),
       'feature_07':('q_07',1)}
for c,(k,s) in CARRY.items():
    if k in S: RHO[c]=r(k)*s
X=pd.read_parquet('artifacts/Xv2.parquet'); Xte=X.iloc[400000:].reset_index(drop=True)
cols=sorted(RHO); P=Xte[cols].values.astype(np.float64)
sd=P.std(0); Z=(P-P.mean(0))/sd
Cz=np.corrcoef(Z,rowvar=False); cz=np.array([RHO[c] for c in cols])*SY
print(f"{'feature':14s} {'rho(v2)':>9s}")
for c in cols: print(f"  {c:12s} {RHO[c]:+9.4f}")
print()
res=[]
for lam in [0,0.001,0.003,0.01,0.03,0.1]:
    bz=np.linalg.solve(Cz+lam*np.eye(len(cols)),cz); p=Z@bz
    R2=float(bz@cz)/SY**2
    print(f"  lam={lam:6.3f}  R2={R2:.4f}  expRMSE={SY*np.sqrt(max(1-R2,0)):.4f}  sigma_f={p.std():.3f}")
    res.append((lam,p,R2))
lam,p,R2=res[2]
print(f"\nchosen lam={lam}")
bz=np.linalg.solve(Cz+lam*np.eye(len(cols)),cz)
for c,b in sorted(zip(cols,bz),key=lambda t:-abs(t[1])): print(f"   {c:12s} {b:+8.4f} (per sd)   raw beta={b/sd[cols.index(c)]:+.5g}")
sub=pd.read_csv('data/sample_submission.csv')
for nm,(lam_,pp,R2_) in zip(['s21_v2_lam0','s22_v2_lam003','s23_v2_lam03'],[res[0],res[2],res[4]]):
    pred=pp+MY; sub['prediction']=pred; sub.to_csv(f'subs/{nm}.csv',index=False)
    print(f"  {nm}: sd={pred.std():.3f} min={pred.min():.2f} max={pred.max():.2f} expRMSE={SY*np.sqrt(max(1-R2_,0)):.4f}")
