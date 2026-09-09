"""Final submission: lam-averaged stack over the LOO-optimal band."""
import numpy as np, pandas as pd, json, os
SY,MY=4.611,3.9
S=json.load(open('artifacts/lb_scores.json'))
names=[];P=[];rmse=[]
for n,sc in S.items():
    f=f'subs/{n}.csv'
    if not os.path.exists(f): continue
    v=pd.read_csv(f)['prediction'].values.astype(np.float64)
    if np.all(np.isfinite(v)): names.append(n);P.append(v);rmse.append(sc)
P=np.column_stack(P); rmse=np.array(rmse)
mu=P.mean(0); var=P.var(0)
cov_y=(SY**2+var+(MY-mu)**2-rmse**2)/2
Pc=P-mu; Cp=(Pc.T@Pc)/len(P); tr=np.trace(Cp)/len(names)
acc=[]; 
for lam in [0.003,0.005,0.008,0.012,0.02]:
    w=np.linalg.solve(Cp+lam*tr*np.eye(len(names)),cov_y)
    p=Pc@w+MY
    mse=SY**2+p.var()+(MY-p.mean())**2-2*(w@cov_y)
    acc.append(p); print(f"  lam={lam:6.3f} predRMSE={np.sqrt(max(mse,1e-9)):.4f} sd={p.std():.3f} |w|1={np.abs(w).sum():.2f}")
F=np.mean(acc,axis=0)
print("\npairwise corr across lam:", np.round(np.corrcoef(np.array(acc))[0],5))
# implied quality of the averaged blend
wbar=np.mean([np.linalg.solve(Cp+l*tr*np.eye(len(names)),cov_y) for l in [0.003,0.005,0.008,0.012,0.02]],axis=0)
mse=SY**2+F.var()+(MY-F.mean())**2-2*(wbar@cov_y)
print(f"AVERAGED stack: predRMSE={np.sqrt(max(mse,1e-9)):.4f} mean={F.mean():.4f} sd={F.std():.4f} min={F.min():.2f} max={F.max():.2f}")
sub=pd.read_csv('data/sample_submission.csv'); sub['prediction']=F
assert sub.shape==(100000,2) and sub.prediction.notna().all() and np.isfinite(F).all()
assert (sub.row_id.values==pd.read_csv('data/sample_submission.csv').row_id.values).all()
sub.to_csv('subs/FINAL.csv',index=False)
print("wrote subs/FINAL.csv"); print(sub.head())
print("corr with current best s22:",np.corrcoef(F,pd.read_csv('subs/s22_v2_lam003.csv')['prediction'])[0,1].round(5))
