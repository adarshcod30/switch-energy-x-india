"""Large atom pool + 49-moment overdetermined selection (beam search)."""
import numpy as np, pandas as pd, json, itertools, time
SY,MY=4.611,3.9
P=np.load('artifacts/P.npy'); cov_y=np.load('artifacts/cov_y.npy'); Pc=np.load('artifacts/Pc.npy')
names=json.load(open('artifacts/P_names.json'))
X=pd.read_parquet('artifacts/Xv2.parquet').iloc[400000:].reset_index(drop=True)
d={c:X[c].values.astype(np.float64) for c in X.columns}
F={f'f{i:02d}':d[f'feature_{i:02d}'] for i in range(1,26)}
G,T,RH,V,PR=F['f01'],F['f02'],F['f03'],F['f04'],F['f05']
CL,TP,SOC,DE,RHO=F['f06'],F['f07'],F['f08'],F['f09'],F['f10']
IE,SOL,WND,GR,TL=F['f11'],F['f12'],F['f13'],F['f14'],F['f15']
PRC,IH,V2,DS,GF,XL=F['f16'],F['f17'],F['f18'],F['f19'],F['f21'],F['f22']
A={}
for k,v in F.items():
    if k in ('f20','f23','f24','f25'): continue
    A[k]=v
A.update({
 'v3':np.clip(V2,0,None)**1.5,'v3raw':np.clip(V,0,None)**3,'rho_v3':RHO*np.clip(V2,0,None)**1.5,
 'TP2':TP**2,'TP3':TP**3,'T2':T**2,'G2':G**2,'W2':WND**2,'W3':WND**3,'Wsq':np.sqrt(np.clip(WND,0,None)),
 'logW':np.log1p(np.clip(WND,0,None)),'expnegTP':np.exp(-TP/40),
 'usable':d['d_usable'],'GR_TL':GR*TL,'W_TL':WND*TL,'S_TL':SOL*TL,'GR_IE':GR*IE,'GR_XL':GR*(1-XL),
 'GR_TL_IE':GR*TL*IE,'W_TL_IE_XL':WND*TL*IE*(1-XL),
 'TP_W':TP*WND,'T_W':T*WND,'G_W':G*WND,'DE_W':DE*WND,'SOC_W':SOC*WND,'XL_GR':XL*GR,
 'DE_GR':DE*GR,'DS_GR':DS*GR,'DE2':DE**2,'DEoS':DE/np.clip(SOC,.02,None),
 'oneoS':1/np.clip(SOC,.02,None),'neglogS':-np.log(np.clip(SOC,.02,None)),
 'GmC':G*(1-CL),'G_TL':G*TL,'IH_':IH,'relu_DE_GR':np.clip(DE-GR,0,None),
 'relu_GR_DE':np.clip(GR-DE,0,None),'relu_1mTL':np.clip(1-TL,0,None)*GR,
 'derate_loss':(1-TL)*GR,'invloss':(1-IE)*GR,'txloss':XL*GR,
 'TPm25':(TP-25),'TPm25_GR':(TP-25)*GR,'TPm25_W':(TP-25)*WND,
 'net':GR-DE,'net_TL':(GR-DE)*TL,'GF50':GF-50,
})
keys=sorted(A); M=np.column_stack([np.nan_to_num(A[k],nan=0,posinf=0,neginf=0) for k in keys])
M=M-M.mean(0)
CM=(Pc.T@M)/len(P)                     # (49, natoms) cov(atom_j, p_i)
print("atoms:",len(keys)," moment matrix:",CM.shape)
def fit(idx):
    B=CM[:,idx]
    b,*_=np.linalg.lstsq(B,cov_y,rcond=None)
    res=cov_y-B@b
    rel=np.linalg.norm(res)/np.linalg.norm(cov_y)
    g=M[:,idx]@b
    return b,rel,g.std()
# beam search
beam=[([],1.0)]
t0=time.time()
for step in range(8):
    cand={}
    for sel,_ in beam:
        for j in range(len(keys)):
            if j in sel: continue
            key=tuple(sorted(sel+[j]))
            if key in cand: continue
            try: b,rel,sf=fit(list(key))
            except Exception: continue
            if not np.isfinite(rel): continue
            pen=0.0 if 4.40<=sf<=4.80 else abs(sf-4.60)/4.60*0.5
            cand[key]=(rel+pen,rel,sf,b)
    beam=sorted(((list(k),v[0]) for k,v in cand.items()),key=lambda t:t[1])[:24]
    bk=beam[0][0]; sc,rel,sf,b=cand[tuple(sorted(bk))]
    print(f"step{step+1}: rel_resid={rel:.6f} sigma_f={sf:6.3f}  {[keys[j] for j in bk]}  [{time.time()-t0:.0f}s]",flush=True)
print("\nTOP 12 formulas (49-moment fit):")
final=sorted(cand.items(),key=lambda t:t[1][0])[:12]
out=[]
for k,(sc,rel,sf,b) in final:
    terms=[keys[j] for j in k]
    print(f"  rel={rel:.6f} sd={sf:6.3f}  "+"  ".join(f"{bb:+.4g}*{t}" for bb,t in zip(b,terms)))
    out.append({'rel':float(rel),'sd':float(sf),'terms':terms,'beta':[float(x) for x in b]})
json.dump(out,open('artifacts/atoms_top.json','w'),indent=1)
