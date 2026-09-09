import pandas as pd, numpy as np, sys
imp=pd.read_parquet('artifacts/imputed.parquet'); ntr=400000
G=imp.feature_14.values;TL=imp.feature_15.values;IE=imp.feature_11.values
XL=imp.feature_22.values;DS=imp.feature_19.values;DE=imp.feature_09.values
SOC=imp.feature_08.values;SOL=imp.feature_12.values;WND=imp.feature_13.values
usable=G*TL*IE*(1-XL)
sub=pd.read_csv('data/sample_submission.csv')
def write(name,arr,calib=None):
    a=arr[ntr:].astype(float)
    if calib=='ms': a=(a-a.mean())/a.std()*4.6+3.9
    sub['prediction']=a; sub.to_csv(f'subs/{name}.csv',index=False)
    print(f"{name:28s} mean={a.mean():8.4f} std={a.std():7.4f} min={a.min():8.3f} max={a.max():7.3f}")
    return a
cands={
 's02_usable_m15dem': usable-15.0*DE+10.9,
 's03_a081_m154dem': 0.809*usable-15.4*DE+10.9,
 's05_demand_only'  : -DE,
 's07_wind_only'    : WND,
 's06_solar_only'   : SOL,
 's08_soc_only'     : SOC,
}
for n,a in cands.items():
    write(n,a,'ms' if n in ('s05_demand_only','s07_wind_only','s06_solar_only','s08_soc_only') else None)
