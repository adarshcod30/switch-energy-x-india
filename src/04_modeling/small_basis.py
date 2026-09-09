"""Overdetermined fit: few physical basis terms, fit to ALL measured moments.
Exact objective: RMSE^2 = sy^2 + b'Sb - 2 b'c."""
import numpy as np, pandas as pd, json, itertools
SY,MY=4.611,3.9
S=json.load(open('artifacts/lb_scores.json')); r=lambda k:1-S[k]**2/(2*SY*SY)
RHO={}
for f in ['15','09','02','04','03','06','08','21']:
    if f'v2_{f}' in S: RHO['feature_'+f]=r(f'v2_{f}')
for c,k in [('feature_13','s07_wind_only'),('feature_12','b_solar_f12'),('feature_01','b_irr_f01'),
            ('feature_19','b_f19_demstor'),('feature_22','b_transloss_f22'),('feature_11','b_inveff_f11'),
            ('feature_07','q_07')]:
    if k in S: RHO[c]=r(k)
X=pd.read_parquet('artifacts/Xv2.parquet').iloc[400000:].reset_index(drop=True)
d={c:X[c].values.astype(np.float64) for c in X.columns}
mcols=sorted(RHO); Mx=X[mcols].values.astype(np.float64)
c_meas=np.array([RHO[c] for c in mcols])*SY*Mx.std(0)      # cov(y, x_j)
Mc=Mx-Mx.mean(0)
POOL={'f13':d['feature_13'],'f12':d['feature_12'],'f14':d['feature_14'],'f07':d['feature_07'],
 'f09':d['feature_09'],'f08':d['feature_08'],'f15':d['feature_15'],'f11':d['feature_11'],
 'f22':d['feature_22'],'f01':d['feature_01'],'f02':d['feature_02'],'f04':d['feature_04'],
 'f03':d['feature_03'],'f06':d['feature_06'],'f19':d['feature_19'],'f21':d['feature_21'],
 'f13*f15':d['feature_13']*d['feature_15'],'f14*f15':d['feature_14']*d['feature_15'],
 'f12*f15':d['feature_12']*d['feature_15'],'usable':d['d_usable'],
 'f07*f13':d['feature_07']*d['feature_13'],'f07sq':d['feature_07']**2,'f13sq':d['feature_13']**2}
names=list(POOL)
def evalset(ts):
    P=np.column_stack([POOL[t] for t in ts]); Pc=P-P.mean(0)
    M=((Pc.T@Mc)/(len(Mc)-1)).T                     # (nmom, k): cov(phi_i, x_j)
    beta,*_=np.linalg.lstsq(M,c_meas,rcond=None)    # OVERDETERMINED least squares
    Sig=np.cov(P,rowvar=False).reshape(len(ts),len(ts))
    cphi=M.T@np.linalg.lstsq(Mc.T@Mc/(len(Mc)-1)+1e-12*np.eye(len(mcols)),c_meas,rcond=None)[0]
    mse=SY**2+beta@Sig@beta-2*beta@cphi
    f=P@beta
    return beta,f,mse,f.std(),1-np.linalg.norm(c_meas-M@beta)/np.linalg.norm(c_meas)
res=[]
for k in (3,4,5):
    for combo in itertools.combinations(names,k):
        try: b,f,mse,sf,fit=evalset(list(combo))
        except Exception: continue
        if not np.isfinite(mse) or mse<=0: continue
        if not (4.30<=sf<=4.75): continue
        res.append((mse,combo,b,sf,fit))
res.sort(key=lambda t:t[0])
print(f"{'expRMSE':>8s} {'sigma_f':>8s} {'momfit':>7s}   formula")
for mse,combo,b,sf,fit in res[:15]:
    print(f"{np.sqrt(mse):8.4f} {sf:8.3f} {fit:7.4f}   "+"  ".join(f"{bb:+.4g}*{t}" for bb,t in zip(b,combo)))
json.dump([[float(np.sqrt(m)),list(c),[float(x) for x in b]] for m,c,b,_,_ in res[:30]],open('artifacts/small_top.json','w'),indent=1)
