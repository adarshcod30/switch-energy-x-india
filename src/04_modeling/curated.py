"""Physics-first candidate families: 'after accounting for X' read as SUBTRACTED losses
with free (non-unit) coefficients -- structurally different from the multiplicative form."""
import numpy as np, pandas as pd, json
SY,MY=4.611,3.9
P=np.load('artifacts/P.npy'); cov_y=np.load('artifacts/cov_y.npy'); Pc=np.load('artifacts/Pc.npy')
X=pd.read_parquet('artifacts/Xv2.parquet').iloc[400000:].reset_index(drop=True)
d={c:X[c].values.astype(np.float64) for c in X.columns}
F={f'f{i:02d}':d[f'feature_{i:02d}'] for i in range(1,26)}
G,T,RH,V=F['f01'],F['f02'],F['f03'],F['f04']
TP,SOC,DE=F['f07'],F['f08'],F['f09']
IE,SOL,WND,GR,TL,PRC,XL,GF=F['f11'],F['f12'],F['f13'],F['f14'],F['f15'],F['f16'],F['f22'],F['f21']
DS=F['f19']; NET=GR-DE
FAM={
 'K1_subtractive_losses':{'f14':GR,'thermloss':(1-TL)*GR,'convloss':(1-IE)*GR,'txloss':XL*GR,'f19':DS,'f09':DE,'f03':RH,'f16':PRC},
 'K2_sub_plus_thermal'  :{'f14':GR,'thermloss':(1-TL)*GR,'convloss':(1-IE)*GR,'txloss':XL*GR,'f07':TP,'f09':DE,'f03':RH,'f16':PRC},
 'K3_regime'            :{'relu_pos':np.clip(NET,0,None),'relu_neg':np.clip(-NET,0,None),'f07':TP,'f02':T,'f01':G,'f03':RH,'f16':PRC,'txloss':XL*GR},
 'K4_regime_derated'    :{'relu_pos':np.clip(NET,0,None)*TL,'relu_neg':np.clip(-NET,0,None),'f07':TP,'f02':T,'f03':RH,'f16':PRC,'convloss':(1-IE)*GR,'txloss':XL*GR},
 'K5_solar_wind_split'  :{'f12':SOL,'f13':WND,'f07':TP,'f02':T,'f09':DE,'f03':RH,'f16':PRC,'txloss':XL*GR},
 'K6_gridfreq_core'     :{'GF50':GF-50,'f07':TP,'f02':T,'f01':G,'f03':RH,'f16':PRC,'convloss':(1-IE)*GR,'txloss':XL*GR},
 'K7_quad_thermal'      :{'f14':GR,'TP2':TP**2,'f02':T,'f03':RH,'f16':PRC,'convloss':(1-IE)*GR,'txloss':XL*GR,'f09':DE},
 'K8_usable_plus'       :{'usable':d['d_usable'],'f07':TP,'f02':T,'f01':G,'f03':RH,'f16':PRC,'f09':DE,'f19':DS},
}
sub=pd.read_csv('data/sample_submission.csv'); out=[]
print(f"{'family':24s} {'rel':>9s} {'sigma_f':>8s}   (noise floor rel=0.00679)")
for nm,terms in FAM.items():
    M=np.column_stack([np.nan_to_num(v,nan=0,posinf=0,neginf=0) for v in terms.values()]); M=M-M.mean(0)
    B=(Pc.T@M)/len(P)
    b,*_=np.linalg.lstsq(B,cov_y,rcond=None)
    g=M@b; rel=np.linalg.norm(cov_y-B@b)/np.linalg.norm(cov_y); sf=g.std()
    ok="OK" if rel<0.012 and 4.4<=sf<=4.85 else ""
    print(f"  {nm:22s} {rel:9.5f} {sf:8.3f}  {ok}")
    print("      "+" ".join(f"{bb:+.4g}*{t}" for bb,t in zip(b,terms)))
    p=g-g.mean()+MY; sub['prediction']=p; sub.to_csv(f'subs/{nm}.csv',index=False)
    out.append({'name':nm,'rel':float(rel),'sd':float(sf),'terms':list(terms),'beta':[float(x) for x in b]})
json.dump(out,open('artifacts/curated.json','w'),indent=1)
