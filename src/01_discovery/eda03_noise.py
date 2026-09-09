import pandas as pd, numpy as np
tr = pd.read_csv('data/train.csv')
F=[f'feature_{i:02d}' for i in range(1,26)]
d={c:tr[c].values for c in F}
def R2(y,p):
    m=np.isfinite(y)&np.isfinite(p); y,p=y[m],p[m]
    A=np.column_stack([np.ones(len(y)),p]); c,*_=np.linalg.lstsq(A,y,rcond=None)
    r=y-A@c; return 1-r.var()/y.var(), r.std(), c

print("=== TEST: is sqrt(f18) a CLEANER wind speed than f04? ===")
v_raw=d['feature_04']; v_der=np.sqrt(np.clip(d['feature_18'],0,None))
for nm,v in [('f04 (raw sensor)',v_raw),('sqrt(f18) (derived)',v_der)]:
    r2,sd,c=R2(d['feature_13'], v**3); print(f"  f13 ~ v^3 using {nm:22s}: R2={r2:.6f} sd={sd:.4f}")
    r2,sd,c=R2(d['feature_18'] if nm.startswith('f04') else d['feature_04'], v**2 if nm.startswith('f04') else v)
print("  corr(f04, sqrt(f18)) =", np.corrcoef(*[x[np.isfinite(v_raw)&np.isfinite(v_der)] for x in (v_raw,v_der)])[0,1])

print("\n=== TEST: is f17/f01 vs f03 --- which is cleaner? predict f12 ===")
G_raw=d['feature_01']
for nm,G in [('f01 raw',G_raw),('f17/(f03/100)',d['feature_17']/(d['feature_03']/100))]:
    r2,sd,c=R2(d['feature_12'], G); print(f"  f12 ~ G using {nm:18s}: R2={r2:.6f} sd={sd:.4f}")

print("\n=== f13 POWER CURVE (use derived v=sqrt(f18) to reduce x-noise) ===")
m=np.isfinite(v_der)&np.isfinite(d['feature_13'])
vv,ww=v_der[m],d['feature_13'][m]
print("   v      n    mean_f13   p05     p95     f13/v^3")
for lo in np.arange(0,20,1.0):
    s=(vv>=lo)&(vv<lo+1)
    if s.sum()>30:
        print(f"  {lo:4.0f} {s.sum():7d} {ww[s].mean():9.4f} {np.percentile(ww[s],5):7.3f} {np.percentile(ww[s],95):7.3f}  {ww[s].mean()/max(((vv[s])**3).mean(),1e-9):.6f}")

print("\n=== f13: fit rated-power curve  P = Pr*(v^3-vci^3)/(vr^3-vci^3) clipped ===")
best=None
for vci in np.arange(2.0,4.1,0.25):
  for vr in np.arange(9.0,14.1,0.25):
    for vco in [18,20,22,25,100]:
      x=np.clip((vv**3-vci**3)/(vr**3-vci**3),0,1); x[vv>=vco]=0; x[vv<vci]=0
      A=np.column_stack([np.ones(len(x)),x]); c,*_=np.linalg.lstsq(A,ww,rcond=None)
      r=ww-A@c; sd=r.std()
      if best is None or sd<best[0]: best=(sd,vci,vr,vco,c)
print(f"  BEST: resid_sd={best[0]:.5f} v_cutin={best[1]} v_rated={best[2]} v_cutout={best[3]} coef={best[4]}")

print("\n=== f12 SOLAR: search multiplicative forms ===")
C=d['feature_06']; TL=d['feature_15']; IE=d['feature_11']; RH=d['feature_03']
Gc=d['feature_17']/(d['feature_03']/100)   # cleaner G
cands={
 'G/1000':G_raw/1000,
 'G(1-C)/1000':G_raw*(1-C)/1000,
 'G(1-0.75C)/1000':G_raw*(1-0.75*C)/1000,
 'G*TL/1000':G_raw*TL/1000,
 'G(1-C)*TL/1000':G_raw*(1-C)*TL/1000,
 'G(1-C)*TL*IE/1000':G_raw*(1-C)*TL*IE/1000,
 'Gc(1-C)*TL/1000':Gc*(1-C)*TL/1000,
}
for nm,x in cands.items():
    r2,sd,c=R2(d['feature_12'],x); print(f"  f12 ~ {nm:22s}: R2={r2:.6f} sd={sd:.5f} coef={c}")

print("\n=== f22 transmission loss: what drives it? ===")
for nm in F:
    if nm=='feature_22': continue
    r2,sd,c=R2(d['feature_22'], d[nm])
    if r2>0.01: print(f"  f22 ~ {nm}: R2={r2:.5f}")
print("  f22 dist: unique frac", pd.Series(d['feature_22']).nunique()/np.isfinite(d['feature_22']).sum())
q=pd.Series(d['feature_22']).dropna()
print("  f22 quantiles:", np.round(np.percentile(q,[0,1,5,25,50,75,90,95,99,100]),5))

print("\n=== f11 inverter eff: what drives it? ===")
for nm in F:
    if nm=='feature_11': continue
    r2,sd,c=R2(d['feature_11'], d[nm])
    if r2>0.01: print(f"  f11 ~ {nm}: R2={r2:.5f}")
q=pd.Series(d['feature_11']).dropna(); print("  f11 quantiles:", np.round(np.percentile(q,[0,1,5,25,50,75,95,99,100]),5))
