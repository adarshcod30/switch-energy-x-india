"""Fingerprint-test capacity-weighted multiplicative forms on the current matrix."""
import numpy as np, pandas as pd, json, itertools
J=json.load(open('artifacts/rho.json')); cols=J['cols']; rho=J['rho']
X=pd.read_parquet('artifacts/Xfilled.parquet').iloc[400000:].reset_index(drop=True)
A=X[cols].values.astype(np.float64); Az=(A-A.mean(0))/A.std(0)
r=np.array([rho[c] for c in cols])
d={c:X[c].values.astype(np.float64) for c in X.columns}
SOL,WND,TL,IE,XL=d['feature_12'],d['feature_13'],d['feature_15'],d['feature_11'],d['feature_22']
DE,SOC,TP=d['feature_09'],d['feature_08'],d['feature_07']
def cos(g):
    g=np.nan_to_num(g); g=(g-g.mean())/g.std()
    rr=(Az*g[:,None]).mean(0)
    return float(rr@r)/(np.linalg.norm(rr)*np.linalg.norm(r))
print("capacity-weighted:  NDEM = (S*f12 + W*f13)*f15*f11*(1-f22) - D*f09")
best=[]
for S in [1,3,5,8,10,12,15,20]:
    for W in [0.4,0.6,0.8,1.0]:
        for D in [0,1,2,4,8,12]:
            g=(S*SOL+W*WND)*TL*IE*(1-XL)-D*DE
            best.append((cos(g),S,W,D))
best.sort(reverse=True)
for c,S,W,D in best[:10]: print(f"   cos={c:.4f}  S={S:5.1f} W={W:4.2f} D={D:4.1f}")
print("\nadditive thermal:  NDEM = W*f13 + S*f12 - Tk*f07 - D*f09")
b2=[]
for S in [0,3,6,9,12,15]:
    for W in [0.4,0.6,0.8]:
        for Tk in [0.1,0.15,0.2,0.25,0.3,0.35]:
            for D in [0,1,2,4,8]:
                b2.append((cos(W*WND+S*SOL-Tk*TP-D*DE),S,W,Tk,D))
b2.sort(reverse=True)
for c,S,W,Tk,D in b2[:10]: print(f"   cos={c:.4f}  S={S:4.1f} W={W:4.2f} Tk={Tk:.2f} D={D:4.1f}")
