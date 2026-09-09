import numpy as np, pandas as pd, json
RHO=json.load(open('artifacts/RHO_full.json')); cols=sorted(RHO); SY=4.611
X=pd.read_parquet('artifacts/Xv2.parquet').iloc[400000:].reset_index(drop=True)
d={c:X[c].values.astype(np.float64) for c in X.columns}
L=np.column_stack([np.ones(len(X))]+[X[c].values.astype(np.float64) for c in cols])
Q,_=np.linalg.qr(L)
orth=lambda v:(lambda w: w-Q@(Q.T@w))(np.nan_to_num(v,nan=0,posinf=0,neginf=0))
SOC,DE,WND,TP=d['feature_08'],d['feature_09'],d['feature_13'],d['feature_07']
S=np.clip(SOC,0.02,None)
R={'f09/f08':DE/S,'1/f08':1/S,'(1-f08)/f08':(1-SOC)/S,'f09/(f08+.1)':DE/(S+.1),
   'f09*(1-f08)/f08':DE*(1-SOC)/S,'-log(f08)':-np.log(S),'f09*-log(f08)':-DE*np.log(S),
   'f09/f08^2':DE/S**2,'exp(-f08)':np.exp(-SOC),'f09/sqrt(f08)':DE/np.sqrt(S),
   'f13/f09':WND/np.clip(DE,.05,None),'f13*f08':WND*SOC}
print(f"{'term':20s} {'orth_sd':>9s} {'corr w/ f08':>12s} {'corr w/ f09':>12s}")
for n,v in R.items():
    o=orth(v); vv=np.nan_to_num(v)
    print(f"  {n:18s} {o.std():9.4f} {np.corrcoef(vv,SOC)[0,1]:12.4f} {np.corrcoef(vv,DE)[0,1]:12.4f}")
# Does adding a ratio term improve the 19-moment fit AND push sigma_f toward 4.60?
Mx=X[cols].values.astype(np.float64); Mc=Mx-Mx.mean(0)
c_meas=np.array([RHO[c] for c in cols])*SY*Mx.std(0)
BASE={'f13':WND,'f07':TP,'f01':d['feature_01'],'f02':d['feature_02'],'f15':d['feature_15'],
      'f09':DE,'f10':d['feature_10'],'f04':d['feature_04'],'f12':d['feature_12']}
def fit(terms):
    P=np.column_stack([np.nan_to_num(t) for t in terms.values()]); Pc=P-P.mean(0)
    M=((Pc.T@Mc)/(len(Mc)-1)).T
    b,*_=np.linalg.lstsq(M,c_meas,rcond=None)
    f=P@b
    return b,f,1-np.linalg.norm(c_meas-M@b)/np.linalg.norm(c_meas),f.std()
b,f,q,sf=fit(BASE); print(f"\nBASE (9 linear terms): momfit={q:.5f} sigma_f={sf:.3f}  [target sigma_f~4.60]")
best=[]
for n,v in R.items():
    T=dict(BASE); T[n]=v
    b,f,q,sf=fit(T); best.append((q,sf,n,b))
best.sort(reverse=True)
print("adding one ratio/drawdown term:")
for q,sf,n,b in best:
    flag=" <== sigma_f in window" if 4.52<=sf<=4.68 else ""
    print(f"  +{n:18s} momfit={q:.5f} sigma_f={sf:6.3f} coef={b[-1]:+.5g}{flag}")
