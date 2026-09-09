import pandas as pd, numpy as np
tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv')
full=pd.concat([tr,te],ignore_index=True)
F=[f'feature_{i:02d}' for i in range(1,26)]
d={c:full[c].values for c in F}
core=['feature_14','feature_15','feature_11','feature_22','feature_09','feature_08']
m=np.ones(len(full),bool)
for c in core: m&=np.isfinite(d[c])
G=d['feature_14'][m];TL=d['feature_15'][m];IE=d['feature_11'][m];XL=d['feature_22'][m]
DE=d['feature_09'][m];SOC=d['feature_08'][m]
usable=G*TL*IE*(1-XL)
print("n complete:",m.sum())
print(f"usable: mean={usable.mean():.4f} std={usable.std():.4f} skew={pd.Series(usable).skew():.3f} pct<0={100*(usable<0).mean():.2f}%")
print("TARGET SPEC: mean 3.9, std 4.6, range [-20,20], 'deficits not rare'\n")
print("Searching drawdown term T so that (usable - T) matches mean/std/min:")
cands={
 'k*f09':DE,'k*f19':DE*SOC,'k*f09/f08':DE/np.clip(SOC,1e-3,None),
 'k*f09*(1-f08)':DE*(1-SOC),'k*f09/(f08+0.1)':DE/(SOC+0.1),
 'k*(1-f08)/f08':(1-SOC)/np.clip(SOC,1e-3,None),'k*f09^2/f08':DE**2/np.clip(SOC,1e-3,None),
 'k*(-log f08)':-np.log(np.clip(SOC,1e-3,None)),'k*f09*(-log f08)':-DE*np.log(np.clip(SOC,1e-3,None)),
 'k*exp(f09)/f08':np.exp(DE)/np.clip(SOC,1e-3,None),
}
print(f"{'term':22s} {'k(mean-match)':>13s} {'std':>8s} {'min':>9s} {'pct<0':>7s} {'skew':>7s}")
for nm,T in cands.items():
    k=(usable.mean()-3.9)/T.mean()          # k that makes mean exactly 3.9
    y=usable-k*T
    print(f"{nm:22s} {k:13.4f} {y.std():8.4f} {y.min():9.3f} {100*(y<0).mean():6.2f}% {pd.Series(y).skew():7.3f}")

print("\n=== ALTERNATIVE: demand scaled to generation units, then standardized ===")
for P in [1,2,3,5,7,10]:
    y=usable-P*DE
    z=(y-y.mean())/y.std()*4.6+3.9
    print(f"  Load={P}*f09: raw mean={y.mean():7.3f} std={y.std():6.3f} | standardized min={z.min():8.3f} max={z.max():7.3f} pct<0={100*(z<0).mean():5.2f}% skew={pd.Series(z).skew():6.3f}")

print("\n=== What if usable itself is standardized (no drawdown)? ===")
z=(usable-usable.mean())/usable.std()*4.6+3.9
print(f"  standardized usable: min={z.min():.3f} max={z.max():.3f} pct<0={100*(z<0).mean():.2f}% skew={pd.Series(z).skew():.3f}")
print("\n=== log/sqrt transforms of usable (to create left tail)? ===")
for nm,t in [('log1p',np.log1p(np.clip(usable,0,None))),('sqrt',np.sqrt(np.clip(usable,0,None))),('usable^0.5',np.clip(usable,0,None)**.5)]:
    z=(t-t.mean())/t.std()*4.6+3.9
    print(f"  {nm:8s}: min={z.min():8.3f} max={z.max():7.3f} pct<0={100*(z<0).mean():5.2f}% skew={pd.Series(z).skew():6.3f}")
