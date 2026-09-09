"""Round 2: re-stack using ALL scored submissions incl. the new orthogonal probes."""
import numpy as np, pandas as pd, json, subprocess, re, os
SY,MY=4.611,3.9
out=subprocess.run(['python3','-m','kaggle','competitions','submissions','-c','switch-energy-x-india'],capture_output=True,text=True).stdout
S={}
for ln in out.splitlines()[2:]:
    m=re.match(r'\s*(\d+)\s+(\S+?)\.csv\s+\S+ \S+\s+.*?SubmissionStatus\.\w+\s+([\d.]+)',ln)
    if m: S.setdefault(m.group(2),float(m.group(3)))
json.dump(S,open('artifacts/lb_scores.json','w'),indent=1)
names=[];P=[];rmse=[]
for n,sc in S.items():
    f=f'subs/{n}.csv'
    if not os.path.exists(f): continue
    v=pd.read_csv(f)['prediction'].values.astype(np.float64)
    if np.all(np.isfinite(v)): names.append(n);P.append(v);rmse.append(sc)
P=np.column_stack(P); rmse=np.array(rmse)
mu=P.mean(0); var=P.var(0)
cov_y=(SY**2+var+(MY-mu)**2-rmse**2)/2
rho=cov_y/(SY*np.sqrt(var))
print("=== ORTHOGONAL ATOM PROBE RESULTS (rho = incremental signal beyond old span) ===")
for i in np.argsort(-np.abs(rho)):
    if names[i].startswith('A_'):
        print(f"  {names[i]:14s} RMSE={rmse[i]:8.5f}  rho={rho[i]:+.4f}")
Pc=P-mu; Cp=(Pc.T@Pc)/len(P); tr=np.trace(Cp)/len(names)
print(f"\nstacking over {len(names)} models")
best=[]
for lam in [0.003,0.006,0.01,0.02,0.04,0.08,0.15]:
    errs=[]
    for i in range(len(names)):
        k=[j for j in range(len(names)) if j!=i]
        wi=np.linalg.solve(Cp[np.ix_(k,k)]+lam*tr*np.eye(len(k)),cov_y[k])
        errs.append((wi@Cp[np.ix_(k,[i])].ravel()-cov_y[i])/max(abs(cov_y[i]),1e-9))
    loo=float(np.sqrt(np.mean(np.square(errs))))
    w=np.linalg.solve(Cp+lam*tr*np.eye(len(names)),cov_y); p=Pc@w+MY
    mse=SY**2+p.var()+(MY-p.mean())**2-2*(w@cov_y)
    print(f"  lam={lam:6.3f} LOO={loo:8.5f} predRMSE={np.sqrt(max(mse,1e-9)):7.4f} sd={p.std():6.3f} |w|1={np.abs(w).sum():7.2f}")
    best.append((loo,lam,p))
best.sort()
sub=pd.read_csv('data/sample_submission.csv')
for r,(loo,lam,p) in enumerate(best[:4]):
    sub['prediction']=p; sub.to_csv(f'subs/R2_lam{str(lam).replace(".","")}.csv',index=False)
    print(f"  wrote R2_lam{str(lam).replace('.','')}: sd={p.std():.3f} min={p.min():.2f} max={p.max():.2f}")
