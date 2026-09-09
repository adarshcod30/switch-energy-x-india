"""Method of moments: cov(y,x_j) = sum_i beta_i cov(phi_i, x_j).  16 moment eqs identify <=16 betas."""
import numpy as np, pandas as pd, json
J=json.load(open('artifacts/rho.json')); cols=J['cols']; rho=J['rho']
SY=4.611; MY=3.9
X=pd.read_parquet('artifacts/Xfilled.parquet').iloc[400000:].reset_index(drop=True)
d={c:X[c].values.astype(np.float64) for c in X.columns}
G=d['feature_01'];T=d['feature_02'];RH=d['feature_03'];V=d['feature_04'];C=d['feature_06']
TP=d['feature_07'];SOC=d['feature_08'];DE=d['feature_09'];IE=d['feature_11'];SOL=d['feature_12']
WND=d['feature_13'];GR=d['feature_14'];TL=d['feature_15'];DS=d['feature_19'];XL=d['feature_22']
A=X[cols].values.astype(np.float64)                    # moment features
sj=A.std(0)
cov_y=np.array([rho[c] for c in cols])*SY*sj           # measured cov(y, x_j)
def solve(basis, name, ridge=1e-6):
    P=np.column_stack([np.nan_to_num(b,nan=0,posinf=0,neginf=0) for b in basis.values()])
    M=np.array([[np.cov(P[:,i],A[:,j])[0,1] for j in range(len(cols))] for i in range(P.shape[1])]).T
    beta,*_=np.linalg.lstsq(M.T@M+ridge*np.eye(P.shape[1]), M.T@cov_y, rcond=None)
    f=P@beta
    resid=cov_y-M@beta
    sf=f.std()
    k=sf/SY                                 # sigma_f/sigma_y  (needs <=1)
    R2=min((sf/SY)**2,1.0)
    fit=1-np.linalg.norm(resid)/np.linalg.norm(cov_y)
    print(f"\n{name}:  moment-fit={fit:.4f}  sigma_f={sf:.3f} (sigma_y={SY})  impliedR2={R2:.4f}  expRMSE={SY*np.sqrt(max(1-R2,0)):.4f}")
    for n,b in sorted(zip(basis,beta),key=lambda t:-abs(t[1]*np.std(np.nan_to_num(basis[t[0]])))):
        print(f"     {n:26s} beta={b:+11.5f}   contrib_sd={abs(b)*np.nanstd(basis[n]):.3f}")
    return f,beta,fit
B1={'f13':WND,'f12':SOL,'f07':TP,'f09':DE,'f08':SOC,'f22':XL,'f11':IE,'f15':TL,
    'f01':G,'f02':T,'f04':V,'f06':C,'f03':RH,'f19':DS}
f1,_,_=solve(B1,'B1 all-linear (14 terms)')
B2={'f13':WND,'f12':SOL,'f07':TP,'f09':DE,'f19':DS,'f22':XL,'f11':IE,'f15':TL,
    'f13*f15':WND*TL,'f13sq':WND**2,'f07sq':TP**2,'f13*f09':WND*DE,'f12*f15':SOL*TL,'f04cu':V**3}
f2,_,_=solve(B2,'B2 with nonlinear (14 terms)')
B3={'f14':GR,'f14*f15':GR*TL,'f07':TP,'f09':DE,'f19':DS,'f22*f14':XL*GR,'f15':TL,
    'f12':SOL,'f13':WND,'f01':G,'f02':T,'f08':SOC,'f13*f15*f11':WND*TL*IE,'usable':GR*TL*IE*(1-XL)}
f3,_,_=solve(B3,'B3 physics-multiplicative (14 terms)')
best=max([(f1,'B1'),(f2,'B2'),(f3,'B3')],key=lambda t:t[0].std())
np.save('artifacts/mom_f1.npy',f1); np.save('artifacts/mom_f2.npy',f2); np.save('artifacts/mom_f3.npy',f3)
sub=pd.read_csv('data/sample_submission.csv')
for nm,f in [('s11_mom_linear',f1),('s12_mom_nonlin',f2),('s13_mom_physics',f3)]:
    p=f-f.mean()+MY; sub['prediction']=p; sub.to_csv(f'subs/{nm}.csv',index=False)
    print(f"  wrote {nm}: sd={p.std():.3f}")
