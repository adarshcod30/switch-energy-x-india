"""Match candidate formulas to the LB-measured correlation fingerprint. Zero submissions."""
import numpy as np, pandas as pd, json, itertools
J=json.load(open('artifacts/rho.json')); cols=J['cols']; rho=J['rho']
X=pd.read_parquet('artifacts/Xfilled.parquet').iloc[400000:].reset_index(drop=True)
A=X[cols].values.astype(np.float64)
Az=(A-A.mean(0))/A.std(0)
r_meas=np.array([rho[c] for c in cols])
d={c:X[c].values.astype(np.float64) for c in X.columns}
def fp(g):
    g=np.nan_to_num(g,nan=0,posinf=0,neginf=0); g=g-g.mean()
    s=g.std()
    if s<1e-12: return None,0
    return (Az*(g/s)[:,None]).mean(0), s
def score(g):
    r,_=fp(g)
    if r is None: return -1,0
    k=float(r@r_meas)/float(r@r) if r@r>0 else 0
    cos=float(r@r_meas)/(np.linalg.norm(r)*np.linalg.norm(r_meas))
    return cos,k
# building blocks
G=d['feature_01'];T=d['feature_02'];RH=d['feature_03'];V=d['feature_04'];P=d['feature_05']
C=d['feature_06'];TP=d['feature_07'];SOC=d['feature_08'];DE=d['feature_09'];RHO=d['feature_10']
IE=d['feature_11'];SOL=d['feature_12'];WND=d['feature_13'];GR=d['feature_14'];TL=d['feature_15']
PR=d['feature_16'];IH=d['feature_17'];V2=d['feature_18'];DS=d['feature_19'];XL=d['feature_22']
usable=GR*TL*IE*(1-XL)
CAND={
 'f13 wind':WND, 'f14 gross':GR, 'usable=f14*f15*f11*(1-f22)':usable,
 'f14*f15':GR*TL, 'f14*(1-f22)':GR*(1-XL), 'f14*f11':GR*IE,
 'f14*f15*f11*(1-f22) - f19':usable-DS,
 'f14 - 0.15*f07':GR-0.15*TP, 'f14 - 0.12*f07 - 8*f09':GR-0.12*TP-8*DE,
 'usable - 0.12*f07':usable-0.12*TP,
 'usable - 0.12*f07 - 2*f19':usable-0.12*TP-2*DS,
 'f14*f15*f11*(1-f22) -0.12f07 -1f09':usable-0.12*TP-DE,
 'f14*(1-(1-f15)) ':GR*TL,
 '0.5f13+2f12-0.12f07':0.5*WND+2*SOL-0.12*TP,
}
print(f"{'candidate':44s} {'cosine':>8s} {'k=sf/sy':>8s} {'impliedR2':>10s} {'expRMSE':>8s}")
res=[]
for n,g in CAND.items():
    cos,k=score(g)
    R2=cos*cos; res.append((cos,n,k))
    print(f"  {n:42s} {cos:8.4f} {k:8.4f} {R2:10.4f} {4.611*np.sqrt(max(1-R2,0)):8.4f}")
print("\nmeasured fingerprint r_meas:")
print("  "+"  ".join(f"{c[-2:]}:{rho[c]:+.3f}" for c in cols))
