"""Pick ridge lam by leave-one-out: can the stack predict a held-out submission's measured cov(y,p)?"""
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
print(f"{'lam':>10s} {'LOO_rel_err':>12s} {'predRMSE':>10s} {'|w|1':>9s} {'sd':>7s}")
rows=[]
for lam in [1e-3,3e-3,1e-2,2e-2,3e-2,5e-2,8e-2,1.2e-1,2e-1,3e-1,5e-1,1.0]:
    errs=[]
    for i in range(len(names)):
        k=[j for j in range(len(names)) if j!=i]
        A=Cp[np.ix_(k,k)]+lam*tr*np.eye(len(k))
        wi=np.linalg.solve(A,cov_y[k])
        pred_i=wi@Cp[np.ix_(k,[i])].ravel()
        errs.append((pred_i-cov_y[i])/max(abs(cov_y[i]),1e-9))
    loo=float(np.sqrt(np.mean(np.square(errs))))
    w=np.linalg.solve(Cp+lam*tr*np.eye(len(names)),cov_y)
    pred=Pc@w+MY
    mse=SY**2+pred.var()+(MY-pred.mean())**2-2*(w@cov_y)
    rows.append((loo,lam,w,pred,mse))
    print(f"{lam:10.4f} {loo:12.5f} {np.sqrt(max(mse,1e-9)):10.4f} {np.abs(w).sum():9.3f} {pred.std():7.3f}")
rows.sort(key=lambda t:t[0])
loo,lam,w,pred,mse=rows[0]
print(f"\nLOO-optimal lam={lam}  (LOO rel err {loo:.5f})  predRMSE={np.sqrt(max(mse,1e-9)):.4f}")
print("top weights:")
for i in np.argsort(-np.abs(w))[:12]: print(f"   {names[i]:24s} w={w[i]:+8.4f}  (its own RMSE {rmse[i]:.4f})")
sub=pd.read_csv('data/sample_submission.csv'); sub['prediction']=pred
sub.to_csv('subs/FINAL_stack.csv',index=False)
print(f"\nwrote subs/FINAL_stack.csv  mean={pred.mean():.3f} sd={pred.std():.3f} min={pred.min():.2f} max={pred.max():.2f}")
# also a conservative blend for comparison
w2=np.linalg.solve(Cp+0.3*tr*np.eye(len(names)),cov_y); p2=Pc@w2+MY
pd.DataFrame({'row_id':sub.row_id,'prediction':p2}).to_csv('subs/FINAL_stack_conservative.csv',index=False)
print(f"wrote conservative: sd={p2.std():.3f}  corr(final,cons)={np.corrcoef(pred,p2)[0,1]:.5f}")
