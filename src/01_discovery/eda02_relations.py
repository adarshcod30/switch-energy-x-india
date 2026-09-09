import pandas as pd, numpy as np
tr = pd.read_csv('data/train.csv')
F=[f'feature_{i:02d}' for i in range(1,26)]
d = {c: tr[c].values for c in F}
def ok(*cols):
    m = np.ones(len(tr), bool)
    for c in cols: m &= ~np.isnan(d[c])
    return m
def fit(y_col, X, name, mask=None):
    """OLS of y on design matrix X (dict of name->array). Reports R2 and resid std."""
    m = ~np.isnan(d[y_col])
    for v in X.values(): m &= np.isfinite(v)
    if mask is not None: m &= mask
    y = d[y_col][m]
    A = np.column_stack([np.ones(m.sum())] + [v[m] for v in X.values()])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A@coef; r = y-pred
    r2 = 1-r.var()/y.var()
    print(f"  {name:55s} n={m.sum():7d} R2={r2:.6f} resid_sd={r.std():.5f}  coef=[" +
          ", ".join(f"{c:+.6g}" for c in coef) + "]")
    return coef, r2, r.std()

print("### f10 air density  =  P/(R*T) ?")
rho = d['feature_05']*100/(287.05*(d['feature_02']+273.15))
fit('feature_10', {'rho':rho}, 'f10 ~ P*100/(287.05*(T+273.15))')

print("\n### f07 panel temp = Tamb + k*G ?")
fit('feature_07', {'T':d['feature_02'],'G':d['feature_01']}, 'f07 ~ f02 + f01')
fit('feature_07', {'T':d['feature_02'],'G':d['feature_01'],'v':d['feature_04']}, 'f07 ~ f02+f01+f04')

print("\n### f15 temp-loss factor = 1 + gamma*(Tpanel-25) ?")
fit('feature_15', {'Tp':d['feature_07']}, 'f15 ~ f07')
fit('feature_15', {'Tp':d['feature_07'],'Tp2':d['feature_07']**2}, 'f15 ~ f07 + f07^2')

print("\n### f17 = irradiance * humidity ?")
fit('feature_17', {'GH':d['feature_01']*d['feature_03']/100}, 'f17 ~ f01*f03/100')
fit('feature_17', {'G':d['feature_01'],'GH':d['feature_01']*d['feature_03']/100}, 'f17 ~ f01 + f01*f03/100')

print("\n### f18 = windspeed^2 ?")
fit('feature_18', {'v2':d['feature_04']**2}, 'f18 ~ f04^2')

print("\n### f19 = demand * SOC ?")
fit('feature_19', {'ds':d['feature_09']*d['feature_08']}, 'f19 ~ f09*f08')

print("\n### f14 = f12 + f13 ?")
fit('feature_14', {'s':d['feature_12'],'w':d['feature_13']}, 'f14 ~ f12 + f13')

print("\n### f12 solar generation ~ ?")
G=d['feature_01']; C=d['feature_06']; TL=d['feature_15']; IE=d['feature_11']
fit('feature_12', {'G':G}, 'f12 ~ G')
fit('feature_12', {'G':G,'GC':G*(1-C)}, 'f12 ~ G + G(1-C)')
fit('feature_12', {'GC':G*(1-C)/1000}, 'f12 ~ G(1-C)/1000')
fit('feature_12', {'x':G*(1-C)*TL/1000}, 'f12 ~ G(1-C)*TL/1000')
fit('feature_12', {'x':G*(1-C)*TL*IE/1000}, 'f12 ~ G(1-C)*TL*IE/1000')

print("\n### f13 wind generation ~ v^3 ?")
v=d['feature_04']; rho10=d['feature_10']
fit('feature_13', {'v3':v**3}, 'f13 ~ v^3')
fit('feature_13', {'rv3':rho10*v**3}, 'f13 ~ rho*v^3')
fit('feature_13', {'rv3':0.5*rho10*v**3/1000}, 'f13 ~ 0.5*rho*v^3/1000')
# check saturation
m=ok('feature_04','feature_13')
vv,ww=v[m],d['feature_13'][m]
print("  f13 vs v bins:")
for lo in range(0,20,2):
    s=(vv>=lo)&(vv<lo+2)
    if s.sum()>50: print(f"    v in [{lo},{lo+2}): n={s.sum():6d} mean_f13={ww[s].mean():8.3f} max={ww[s].max():8.3f} mean_v3={(vv[s]**3).mean():9.2f} ratio={ww[s].mean()/max((vv[s]**3).mean(),1e-9):.6f}")

print("\n### f21 grid frequency ~ ?")
fit('feature_21', {'w':d['feature_13']}, 'f21 ~ f13')
fit('feature_21', {'g':d['feature_14']}, 'f21 ~ f14 (gross gen)')
fit('feature_21', {'g':d['feature_14'],'dem':d['feature_09']}, 'f21 ~ f14 + f09')

print("\n### f22 transmission loss ~ ?")
fit('feature_22', {'g':d['feature_14']}, 'f22 ~ f14')
fit('feature_22', {'dem':d['feature_09']}, 'f22 ~ f09')
fit('feature_22', {'dem':d['feature_09'],'d2':d['feature_09']**2}, 'f22 ~ f09 + f09^2')

print("\n### f11 inverter efficiency ~ ?")
fit('feature_11', {'g':d['feature_14']}, 'f11 ~ f14')
fit('feature_11', {'s':d['feature_12']}, 'f11 ~ f12')
fit('feature_11', {'s':d['feature_12'],'s2':d['feature_12']**2}, 'f11 ~ f12 + f12^2')

print("\n### f16 precipitation ~ humidity/cloud ?")
fit('feature_16', {'h':d['feature_03'],'c':d['feature_06']}, 'f16 ~ f03 + f06')

print("\n### distractor f24 drift vs row_id ?")
rid = tr.row_id.values.astype(float)
fit('feature_24', {'r':rid}, 'f24 ~ row_id')
fit('feature_24', {'s':np.sin(2*np.pi*rid/400000),'c':np.cos(2*np.pi*rid/400000)}, 'f24 ~ sin/cos(row_id)')
