"""Sparse method-of-moments symbolic search, constrained to sigma_f in [4.50,4.65]."""
import numpy as np, pandas as pd, json, itertools, sys
J=json.load(open('artifacts/rho.json')); cols=J['cols']; rho=J['rho']
SY=4.611
X=pd.read_parquet('artifacts/Xfilled.parquet').iloc[400000:].reset_index(drop=True)
d={c:X[c].values.astype(np.float64) for c in X.columns}
G,T,RH,V,P=d['feature_01'],d['feature_02'],d['feature_03'],d['feature_04'],d['feature_05']
C,TP,SOC,DE,RHOa=d['feature_06'],d['feature_07'],d['feature_08'],d['feature_09'],d['feature_10']
IE,SOL,WND,GR,TL=d['feature_11'],d['feature_12'],d['feature_13'],d['feature_14'],d['feature_15']
PRC,IH,V2,DS,XL=d['feature_16'],d['feature_17'],d['feature_18'],d['feature_19'],d['feature_22']
POOL={
 'f13':WND,'f12':SOL,'f14':GR,'f07':TP,'f09':DE,'f08':SOC,'f15':TL,'f11':IE,'f22':XL,
 'f01':G,'f02':T,'f03':RH,'f04':V,'f06':C,'f19':DS,'f05':P,'f10':RHOa,'f16':PRC,'f17':IH,'f18':V2,
 'f14*f15':GR*TL,'f13*f15':WND*TL,'f12*f15':SOL*TL,'f14*f11':GR*IE,'f14*(1-f22)':GR*(1-XL),
 'usable':GR*TL*IE*(1-XL),'f14*f15*f11':GR*TL*IE,
 '(1-f15)*f14':(1-TL)*GR,'f22*f14':XL*GR,'f07*f14':TP*GR,'f07*f13':TP*WND,
 'f07sq':TP**2,'f13sq':WND**2,'sqrt_f13':np.sqrt(np.clip(WND,0,None)),'log_f13':np.log1p(np.clip(WND,0,None)),
 'f04cu':V**3,'f09*f14':DE*GR,'f09*f13':DE*WND,'f01*f15':G*TL,'f01*(1-f06)':G*(1-C),
 'f08*f14':SOC*GR,'(1-f08)*f09':(1-SOC)*DE,'f09sq':DE**2,'f02*f14':T*GR,
}
names=list(POOL); Pall=np.column_stack([np.nan_to_num(POOL[n],nan=0,posinf=0,neginf=0) for n in names])
A=X[cols].values.astype(np.float64); sj=A.std(0)
cov_y=np.array([rho[c] for c in cols])*SY*sj
Ac=A-A.mean(0)
def momcols(idx):
    Pc=Pall[:,idx]-Pall[:,idx].mean(0)
    return (Pc.T@Ac)/(len(Ac)-1)          # (k,16) cov(phi_i, x_j)
def fit(idx):
    M=momcols(idx).T                      # (16,k)
    beta,*_=np.linalg.lstsq(M,cov_y,rcond=None)
    f=Pall[:,idx]@beta
    res=cov_y-M@beta
    return beta,f,1-np.linalg.norm(res)/np.linalg.norm(cov_y),f.std()
# greedy forward selection
cur=[]; best_hist=[]
for step in range(9):
    cands=[]
    for i in range(len(names)):
        if i in cur: continue
        try:
            b,f,fq,sf=fit(cur+[i])
        except Exception: continue
        if not np.isfinite(sf) or sf>12: continue
        pen=abs(min(sf,SY)-SY)/SY
        cands.append((fq-0.5*max(0,(sf-SY)/SY)-0.02*pen,fq,sf,i))
    if not cands: break
    cands.sort(reverse=True); sc,fq,sf,i=cands[0]; cur.append(i)
    b,f,fq,sf=fit(cur)
    print(f"step{step+1}: +{names[i]:14s} momfit={fq:.5f} sigma_f={sf:6.3f}  terms={[names[j] for j in cur]}")
    best_hist.append((list(cur),fq,sf))
print("\n--- exhaustive size-4/5 subsets over top-18 pool terms (sigma_f window 4.45-4.68) ---")
top=[names.index(n) for n in ['f13','f07','f09','f02','f01','f04','f14','f12','f15','f19','f13*f15','f14*f15','f07*f13','usable','f07sq','f13sq','f03','f08']]
res=[]
for k in (4,5):
    for combo in itertools.combinations(top,k):
        try: b,f,fq,sf=fit(list(combo))
        except Exception: continue
        if 4.45<=sf<=4.68 and fq>0.995: res.append((fq,sf,combo,b))
res.sort(reverse=True)
print(f"found {len(res)} qualifying formulas; top 12:")
for fq,sf,combo,b in res[:12]:
    print(f"  momfit={fq:.5f} sigma_f={sf:.3f}  "+" ".join(f"{bb:+.4g}*{names[c]}" for bb,c in zip(b,combo)))
json.dump([[float(fq),float(sf),[names[c] for c in combo],[float(x) for x in b]] for fq,sf,combo,b in res[:40]],
          open('artifacts/search_top.json','w'),indent=1)
