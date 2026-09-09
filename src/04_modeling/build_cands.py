import numpy as np, pandas as pd, json
J=json.load(open('artifacts/rho.json')); cols=J['cols']; rho=J['rho']; SY=4.611; MY=3.9
X=pd.read_parquet('artifacts/Xfilled.parquet').iloc[400000:].reset_index(drop=True)
d={c:X[c].values.astype(np.float64) for c in X.columns}
G,T,RH,V=d['feature_01'],d['feature_02'],d['feature_03'],d['feature_04']
C,TP,SOC,DE=d['feature_06'],d['feature_07'],d['feature_08'],d['feature_09']
RHOa,IE,SOL,WND,GR,TL=d['feature_10'],d['feature_11'],d['feature_12'],d['feature_13'],d['feature_14'],d['feature_15']
DS,XL=d['feature_19'],d['feature_22']
POOL={'f13':WND,'f12':SOL,'f14':GR,'f07':TP,'f09':DE,'f08':SOC,'f15':TL,'f11':IE,'f22':XL,
 'f01':G,'f02':T,'f03':RH,'f04':V,'f06':C,'f19':DS,'f10':RHOa,
 'f14*f15':GR*TL,'f13*f15':WND*TL,'f12*f15':SOL*TL,'usable':GR*TL*IE*(1-XL),
 'f07sq':TP**2,'f07*f13':TP*WND,'f09*f13':DE*WND}
A=X[cols].values.astype(np.float64); cov_y=np.array([rho[c] for c in cols])*SY*A.std(0)
Ac=A-A.mean(0)
def fitset(terms):
    P=np.column_stack([np.nan_to_num(POOL[t],nan=0,posinf=0,neginf=0) for t in terms])
    Pc=P-P.mean(0); M=((Pc.T@Ac)/(len(Ac)-1)).T
    beta,*_=np.linalg.lstsq(M,cov_y,rcond=None)
    f=P@beta; return beta,f,1-np.linalg.norm(cov_y-M@beta)/np.linalg.norm(cov_y),f.std()
SETS={
 's14_greedy9':['f15','f10','f03','f13*f15','f12*f15','f07','f04','f02','f09*f13'],
 's15_solar9' :['f07','f02','f01','f12','f07*f13'],
 's16_f14f13' :['f13','f07','f02','f01','f14'],
 's17_derated':['f07','f02','f01','f13*f15','f14*f15'],
 's18_greedy7':['f15','f10','f03','f13*f15','f12*f15','f07','f04'],
}
sub=pd.read_csv('data/sample_submission.csv')
for nm,ts in SETS.items():
    b,f,fq,sf=fitset(ts)
    p=f-f.mean()+MY
    print(f"{nm}: momfit={fq:.5f} sigma_f={sf:.3f}  min={p.min():.2f} max={p.max():.2f}")
    print("     "+"  ".join(f"{bb:+.5g}*{t}" for bb,t in zip(b,ts)))
    sub['prediction']=p; sub.to_csv(f'subs/{nm}.csv',index=False)
