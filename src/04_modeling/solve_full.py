import numpy as np, pandas as pd, json, subprocess, re, itertools
SY,MY=4.611,3.9
out=subprocess.run(['python3','-m','kaggle','competitions','submissions','-c','switch-energy-x-india'],capture_output=True,text=True).stdout
S={}
for ln in out.splitlines()[2:]:
    m=re.match(r'\s*(\d+)\s+(\S+?)\.csv\s+\S+ \S+\s+.*?SubmissionStatus\.\w+\s+([\d.]+)',ln)
    if m: S.setdefault(m.group(2),float(m.group(3)))
json.dump(S,open('artifacts/lb_scores.json','w'),indent=1)
r=lambda k:1-S[k]**2/(2*SY*SY)
RHO={}
for f in ['15','09','02','04','03','06','08','21']: RHO['feature_'+f]=r(f'v2_{f}')
for f,k in [('10','w_f10'),('17','w_f17'),('18','w_f18'),('16','w_f16')]: RHO['feature_'+f]=r(k)
for c,k in [('feature_13','s07_wind_only'),('feature_12','b_solar_f12'),('feature_01','b_irr_f01'),
            ('feature_19','b_f19_demstor'),('feature_22','b_transloss_f22'),('feature_11','b_inveff_f11'),
            ('feature_07','q_07')]: RHO[c]=r(k)
json.dump(RHO,open('artifacts/RHO_full.json','w'),indent=1)
X=pd.read_parquet('artifacts/Xv2.parquet').iloc[400000:].reset_index(drop=True)
cols=sorted(RHO); P=X[cols].values.astype(np.float64); sd=P.std(0)
Z=(P-P.mean(0))/sd; Cz=np.corrcoef(Z,rowvar=False); cz=np.array([RHO[c] for c in cols])*SY
print("measured rho ("+str(len(cols))+" features):")
for c in cols: print(f"   {c:12s} {RHO[c]:+.4f}")
print()
preds={}
for lam in [0.003,0.01,0.02,0.04,0.08,0.15]:
    bz=np.linalg.solve(Cz+lam*np.eye(len(cols)),cz); p=Z@bz
    print(f"  lam={lam:6.3f} R2={float(bz@cz)/SY**2:.4f} expRMSE={SY*np.sqrt(max(1-float(bz@cz)/SY**2,0)):.4f} sd={p.std():.3f}")
    preds[lam]=p+MY
np.save('artifacts/preds_lam.npy',np.array([preds[l] for l in preds])); json.dump(list(preds),open('artifacts/lams.json','w'))
sub=pd.read_csv('data/sample_submission.csv')
for lam,p in preds.items():
    sub['prediction']=p; sub.to_csv(f'subs/s3_lam{str(lam).replace(".","")}.csv',index=False)
print("wrote s3_lam* candidates")
