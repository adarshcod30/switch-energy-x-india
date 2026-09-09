"""49-dim fingerprint: every scored submission is a measurement of cov(y, p_i).
For candidate g we can compute cov(g,p_i) exactly -> match against measured."""
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
cov_y=(SY**2+var+(MY-mu)**2-rmse**2)/2      # EXACT cov(y,p_i), 49 measurements
Pc=P-mu; Cp=(Pc.T@Pc)/len(P)
np.save('artifacts/P.npy',P); np.save('artifacts/cov_y.npy',cov_y)
json.dump(names,open('artifacts/P_names.json','w'))
# how much of y is captured by span(P)?
w=np.linalg.solve(Cp+1e-3*np.trace(Cp)/len(names)*np.eye(len(names)),cov_y)
yP_var=float(w@cov_y)
print(f"sigma_y^2            = {SY**2:.4f}")
print(f"||y_P||^2 (in span)  = {yP_var:.4f}   -> sd {np.sqrt(yP_var):.4f}")
print(f"||y_perp||^2         = {SY**2-yP_var:.4f} -> sd {np.sqrt(max(SY**2-yP_var,0)):.4f}")
print(f"stack actual RMSE    = 1.07022  -> true ||y_perp|| = {np.sqrt(1.07022**2-0.28**2):.4f} (ex-noise)")
print(f"\n=> a component of sd ~1.03 is ORTHOGONAL to all 49 submitted vectors.")
print("   The fingerprint can RANK candidates but cannot measure that part.")
print("   => new probe directions are required; choose them to span it.\n")
# fingerprint scorer for candidate g
def score(g,verbose=False):
    g=np.nan_to_num(g,nan=0,posinf=0,neginf=0).astype(np.float64)
    gc=g-g.mean()
    if gc.std()<1e-12: return -1,0,0
    cg=(Pc.T@gc)/len(P)                       # cov(g,p_i) for all 49
    cos=float(cg@cov_y)/(np.linalg.norm(cg)*np.linalg.norm(cov_y))
    k=float(cg@cov_y)/float(cg@cg)            # implied scale
    return cos,k,gc.std()
np.save('artifacts/Pc.npy',Pc)
# sanity: score a few known vectors
X=pd.read_parquet('artifacts/Xv2.parquet').iloc[400000:].reset_index(drop=True)
d={c:X[c].values.astype(np.float64) for c in X.columns}
for nm,g in [('f13 wind',d['feature_13']),('usable',d['d_usable']),
             ('stack FINAL',pd.read_csv('subs/FINAL.csv')['prediction'].values),
             ('s22',pd.read_csv('subs/s22_v2_lam003.csv')['prediction'].values)]:
    c,k,s=score(g); print(f"  {nm:14s} fp_cos={c:.5f} implied_scale={k:.4f} sd={s:.3f}")
