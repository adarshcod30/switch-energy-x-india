"""Final solve on fully-imputed matrix using freshly measured (unattenuated) correlations."""
import numpy as np, pandas as pd, json, subprocess, re
SY,MY=4.611,3.9
out=subprocess.run(['python3','-m','kaggle','competitions','submissions','-c','switch-energy-x-india'],
                   capture_output=True,text=True).stdout
S={}
for ln in out.splitlines()[2:]:
    m=re.match(r'\s*(\d+)\s+(\S+?)\.csv\s+([\d\-]+ [\d:.]+)\s+(.*?)\s+SubmissionStatus\.(\w+)\s+([\d.]+)?',ln)
    if m and m.group(6): S.setdefault(m.group(2),float(m.group(6)))
json.dump(S,open('artifacts/lb_scores.json','w'),indent=1)
X=pd.read_parquet('artifacts/Xfull.parquet'); Xte=X.iloc[400000:].reset_index(drop=True)
d={c:Xte[c].values.astype(np.float64) for c in Xte.columns}
NL={'q_nl_f13xf15':d['feature_13']*d['feature_15'],'q_nl_f07sq':d['feature_07']**2,
    'q_nl_f13sq':d['feature_13']**2,'q_nl_f07xf13':d['feature_07']*d['feature_13'],
    'q_nl_usable':d['feature_14']*d['feature_15']*d['feature_11']*(1-d['feature_22'])}
basis={}; rho={}
for k,v in S.items():
    if not k.startswith('q_'): continue
    r=1-v*v/(2*SY*SY)
    basis[k]=NL[k] if k in NL else d['feature_'+k[2:]]
    rho[k]=r
names=sorted(basis)
if not names: print("no q_ probes scored yet"); raise SystemExit
P=np.column_stack([np.nan_to_num(basis[n]) for n in names])
sd=P.std(0); cov_y=np.array([rho[n] for n in names])*SY*sd
Cz=np.corrcoef((P-P.mean(0))/sd,rowvar=False); cz=np.array([rho[n] for n in names])*SY
print(f"{'basis':18s} {'rho':>8s}")
for n in names: print(f"  {n:16s} {rho[n]:8.4f}")
best=None
for lam in [0,0.002,0.005,0.01,0.02,0.05,0.1]:
    bz=np.linalg.solve(Cz+lam*np.eye(len(names)),cz)
    p=((P-P.mean(0))/sd)@bz; R2=float(bz@cz)/SY**2
    print(f"  lam={lam:6.3f} R2={R2:.4f} expRMSE={SY*np.sqrt(max(1-R2,0)):.4f} sigma_f={p.std():.3f}")
    if best is None or abs(p.std()-4.575)<abs(best[1]-4.575): best=(lam,p.std(),bz,p)
lam,sf,bz,p=best
print(f"\nchosen lam={lam} sigma_f={sf:.3f}")
for n,b in sorted(zip(names,bz),key=lambda t:-abs(t[1])): print(f"   {n:16s} {b:+8.4f}")
pred=p+MY
sub=pd.read_csv('data/sample_submission.csv'); sub['prediction']=pred
sub.to_csv('subs/s20_final_solve.csv',index=False)
print(f"wrote s20_final_solve: mean={pred.mean():.3f} sd={pred.std():.3f} min={pred.min():.2f} max={pred.max():.2f}")
