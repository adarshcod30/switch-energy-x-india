"""Build nonlinear basis fns orthogonalized against the 19 linear features.
Rank by orthogonal sd = how much *invisible* variance each could carry."""
import numpy as np, pandas as pd, json
RHO=json.load(open('artifacts/RHO_full.json')); cols=sorted(RHO)
X=pd.read_parquet('artifacts/Xv2.parquet').iloc[400000:].reset_index(drop=True)
d={c:X[c].values.astype(np.float64) for c in X.columns}
L=np.column_stack([np.ones(len(X))]+[X[c].values.astype(np.float64) for c in cols])
Q,_=np.linalg.qr(L)                     # orthonormal basis of the linear span
def orth(v):
    v=np.nan_to_num(v,nan=0,posinf=0,neginf=0).astype(np.float64)
    return v-Q@(Q.T@v)
G,T,V=d['feature_01'],d['feature_02'],d['feature_04']
TP,SOC,DE=d['feature_07'],d['feature_08'],d['feature_09']
IE,SOL,WND,GR,TL,XL=d['feature_11'],d['feature_12'],d['feature_13'],d['feature_14'],d['feature_15'],d['feature_22']
C={'f14':GR,'f14*f15':GR*TL,'f13*f15':WND*TL,'usable':d['d_usable'],'f14*f11':GR*IE,
 'f14*(1-f22)':GR*(1-XL),'f13sq':WND**2,'f07sq':TP**2,'f07*f13':TP*WND,'f09*f13':DE*WND,
 'f12*f15':SOL*TL,'f01*f15':G*TL,'f08*f13':SOC*WND,'f02*f13':T*WND,'sqrt_f13':np.sqrt(np.clip(WND,0,None)),
 'log_f13':np.log1p(np.clip(WND,0,None)),'f13cu':WND**3,'f09*f14':DE*GR,'f19*f14':d['feature_19']*GR,
 'f22*f14':XL*GR,'f11*f14':IE*GR,'f15sq':TL**2,'f09sq':DE**2,'expf13':np.expm1(np.clip(WND,0,8)/8)}
rows=[]
for n,v in C.items():
    o=orth(v); rows.append((o.std(),n,o))
rows.sort(reverse=True)
print(f"{'candidate':16s} {'orth_sd':>10s} {'orth_sd/sd':>11s}   (target: a term carrying ~1.25 sd of hidden signal)")
for s,n,o in rows:
    tot=np.nan_to_num(C[n]).std()
    print(f"  {n:14s} {s:10.4f} {s/max(tot,1e-9):11.4f}")
np.save('artifacts/orth_names.npy',np.array([n for _,n,_ in rows]))
np.save('artifacts/orth_vecs.npy',np.column_stack([o for _,_,o in rows]))
