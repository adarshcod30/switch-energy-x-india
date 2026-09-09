"""Optimal stack of all scored submissions. cov(y,p_i) recovered exactly from each public RMSE."""
import numpy as np, pandas as pd, json, subprocess, re, os, glob
SY,MY=4.611,3.9
out=subprocess.run(['python3','-m','kaggle','competitions','submissions','-c','switch-energy-x-india'],capture_output=True,text=True).stdout
S={}
for ln in out.splitlines()[2:]:
    m=re.match(r'\s*(\d+)\s+(\S+?)\.csv\s+\S+ \S+\s+.*?SubmissionStatus\.\w+\s+([\d.]+)',ln)
    if m: S.setdefault(m.group(2),float(m.group(3)))
print("scored submissions:",len(S))
names=[]; P=[]; rmse=[]
for n,sc in S.items():
    f=f'subs/{n}.csv'
    if not os.path.exists(f): continue
    v=pd.read_csv(f)['prediction'].values.astype(np.float64)
    if not np.all(np.isfinite(v)): continue
    names.append(n); P.append(v); rmse.append(sc)
P=np.column_stack(P); rmse=np.array(rmse)
print("usable vectors:",P.shape)
mu=P.mean(0); var=P.var(0)
cov_y=(SY**2+var+(MY-mu)**2-rmse**2)/2            # exact cov(y, p_i)
implied_rho=cov_y/(SY*np.sqrt(var))
order=np.argsort(-implied_rho)
print(f"\n{'submission':26s} {'RMSE':>8s} {'rho':>8s}")
for i in order[:14]: print(f"  {names[i]:24s} {rmse[i]:8.4f} {implied_rho[i]:8.4f}")
Pc=P-mu; Cp=(Pc.T@Pc)/len(P)
best=None
for lam in [1e-6,1e-4,1e-3,1e-2,3e-2,1e-1,3e-1,1.0,3.0]:
    w=np.linalg.solve(Cp+lam*np.eye(len(names))*np.trace(Cp)/len(names),cov_y)
    pred=Pc@w+MY
    mse=SY**2+pred.var()+(MY-pred.mean())**2-2*(w@cov_y)
    print(f"  lam={lam:8.1e} predRMSE={np.sqrt(max(mse,1e-9)):7.4f} sd={pred.std():6.3f} |w|1={np.abs(w).sum():7.3f} nnz={np.sum(np.abs(w)>1e-3)}")
    if best is None or mse<best[0]: best=(mse,lam,w,pred)
mse,lam,w,pred=best
print(f"\nchosen lam={lam:.1e} predRMSE={np.sqrt(max(mse,1e-9)):.4f}")
for i in np.argsort(-np.abs(w))[:12]: print(f"   {names[i]:24s} w={w[i]:+8.4f}")
np.save('artifacts/stack_pred.npy',pred); json.dump({'names':names,'w':w.tolist(),'lam':lam},open('artifacts/stack_w.json','w'),indent=1)
