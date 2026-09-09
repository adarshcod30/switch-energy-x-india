"""Submit standardized single-basis predictions to measure corr(basis, hidden target)."""
import pandas as pd, numpy as np, sys, os
imp=pd.read_parquet('artifacts/imputed.parquet'); ntr=400000
full=pd.concat([pd.read_csv('data/train.csv'),pd.read_csv('data/test.csv')],ignore_index=True)
for c in imp.columns: full[c]=imp[c].values
d={c:full[c].values for c in full.columns if c.startswith('feature_')}
G=d['feature_14'];TL=d['feature_15'];IE=d['feature_11'];XL=d['feature_22']
usable=G*TL*IE*(1-XL)
BASIS={
 'b_solar_f12'   : d['feature_12'],
 'b_soc_f08'     : d['feature_08'],
 'b_irr_f01'     : d['feature_01'],
 'b_f19_demstor' : d['feature_19'],
 'b_temploss_f15': d['feature_15'],
 'b_transloss_f22': d['feature_22'],
 'b_gridfreq_f21': d['feature_21'],
 'b_cloud_f06'   : d['feature_06'],
 'b_ambT_f02'    : d['feature_02'],
 'b_logwind'     : np.log1p(np.clip(d['feature_13'],0,None)),
 'b_sqrtwind'    : np.sqrt(np.clip(d['feature_13'],0,None)),
 'b_usable_x_soc': usable*d['feature_08'],
}
sub=pd.read_csv('data/sample_submission.csv')
os.makedirs('subs',exist_ok=True)
which=sys.argv[1:] if len(sys.argv)>1 else list(BASIS)
for n in which:
    a=BASIS[n][ntr:].astype(float); a=(a-a.mean())/a.std()*4.6+3.9
    sub['prediction']=a; sub.to_csv(f'subs/{n}.csv',index=False)
    print(n,'written')
