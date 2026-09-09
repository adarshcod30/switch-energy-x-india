"""Deeper search + LOO validation over the 49 moment equations (offline CV analogue)."""
import numpy as np, pandas as pd, json, itertools, time
SY,MY=4.611,3.9
P=np.load('artifacts/P.npy'); cov_y=np.load('artifacts/cov_y.npy'); Pc=np.load('artifacts/Pc.npy')
X=pd.read_parquet('artifacts/Xv2.parquet').iloc[400000:].reset_index(drop=True)
d={c:X[c].values.astype(np.float64) for c in X.columns}
F={f'f{i:02d}':d[f'feature_{i:02d}'] for i in range(1,26)}
G,T,RH,V,PR=F['f01'],F['f02'],F['f03'],F['f04'],F['f05']
CL,TP,SOC,DE,RHO=F['f06'],F['f07'],F['f08'],F['f09'],F['f10']
IE,SOL,WND,GR,TL=F['f11'],F['f12'],F['f13'],F['f14'],F['f15']
PRC,IH,V2,DS,GF,XL=F['f16'],F['f17'],F['f18'],F['f19'],F['f21'],F['f22']
NET=GR-DE
A={k:v for k,v in F.items() if k not in ('f20','f23','f24','f25')}
A.update({
 'v3':np.clip(V2,0,None)**1.5,'rho_v3':RHO*np.clip(V2,0,None)**1.5,
 'TP2':TP**2,'TP3':TP**3,'T2':T**2,'G2':G**2,'W2':WND**2,'W3':WND**3,
 'Wsq':np.sqrt(np.clip(WND,0,None)),'logW':np.log1p(np.clip(WND,0,None)),
 'usable':d['d_usable'],'GR_TL':GR*TL,'W_TL':WND*TL,'S_TL':SOL*TL,'GR_IE':GR*IE,
 'GR_XL':GR*(1-XL),'GR_TL_IE':GR*TL*IE,'TP_W':TP*WND,'T_W':T*WND,'G_W':G*WND,
 'DE_W':DE*WND,'SOC_W':SOC*WND,'XL_GR':XL*GR,'DE_GR':DE*GR,'DS_GR':DS*GR,'DE2':DE**2,
 'DEoS':DE/np.clip(SOC,.02,None),'oneoS':1/np.clip(SOC,.02,None),'GmC':G*(1-CL),'G_TL':G*TL,
 'derate_loss':(1-TL)*GR,'invloss':(1-IE)*GR,'txloss':XL*GR,'TPm25_GR':(TP-25)*GR,
 'TPm25_W':(TP-25)*WND,'net':NET,'net_TL':NET*TL,'GF50':GF-50,
 # piecewise / regime terms around the surplus-deficit boundary
 'relu_pos':np.clip(NET,0,None),'relu_neg':np.clip(-NET,0,None),
 'relu_pos_TL':np.clip(NET,0,None)*TL,'relu_neg_SOC':np.clip(-NET,0,None)*SOC,
 'ind_def':(NET<0).astype(float),'ind_def_DE':(NET<0).astype(float)*DE,
 'net_usable':d['d_usable']-DE,'relu_us':np.clip(d['d_usable']-DE,0,None),
 'reluneg_us':np.clip(DE-d['d_usable'],0,None),
 'sqrt_pos':np.sqrt(np.clip(NET,0,None)),'net2':NET**2,'net_SOC':NET*SOC,
})
keys=sorted(A); M=np.column_stack([np.nan_to_num(A[k],nan=0,posinf=0,neginf=0) for k in keys])
M=M-M.mean(0); CM=(Pc.T@M)/len(P)
print("atoms:",len(keys),"moments:",CM.shape[0])
def fit(idx,rows=None):
    B=CM[:,idx] if rows is None else CM[np.ix_(rows,idx)]
    tgt=cov_y if rows is None else cov_y[rows]
    b,*_=np.linalg.lstsq(B,tgt,rcond=None); return b
def evalset(idx):
    b=fit(idx); res=cov_y-CM[:,idx]@b
    rel=np.linalg.norm(res)/np.linalg.norm(cov_y); sf=(M[:,idx]@b).std()
    # LOO over the 49 moments
    errs=[]
    n=CM.shape[0]
    for i in range(n):
        r=[j for j in range(n) if j!=i]
        bi=fit(idx,r); errs.append(CM[i,idx]@bi-cov_y[i])
    loo=float(np.sqrt(np.mean(np.square(errs)))/np.abs(cov_y).mean())
    return b,rel,sf,loo
beam=[[]]; t0=time.time(); cand={}
for step in range(8):
    c2={}
    for sel in beam:
        for j in range(len(keys)):
            if j in sel: continue
            key=tuple(sorted(sel+[j]))
            if key in c2: continue
            B=CM[:,list(key)]
            b,*_=np.linalg.lstsq(B,cov_y,rcond=None)
            rel=np.linalg.norm(cov_y-B@b)/np.linalg.norm(cov_y)
            sf=(M[:,list(key)]@b).std()
            if not np.isfinite(rel): continue
            pen=0.0 if 4.45<=sf<=4.78 else abs(sf-4.60)/4.60
            c2[key]=(rel+pen,rel,sf)
    ranked=sorted(c2.items(),key=lambda t:t[1][0])[:24]
    beam=[list(k) for k,_ in ranked]; cand.update(c2)
    k0,(s0,r0,f0)=ranked[0]
    print(f"step{step+1}: rel={r0:.6f} sd={f0:6.3f} {[keys[j] for j in k0]} [{time.time()-t0:.0f}s]",flush=True)
print("\nLOO-validated top candidates:")
res=[]
for k,_ in sorted(cand.items(),key=lambda t:t[1][0])[:20]:
    b,rel,sf,loo=evalset(list(k))
    res.append((loo,rel,sf,[keys[j] for j in k],b))
res.sort()
for loo,rel,sf,ts,b in res[:10]:
    print(f"  LOO={loo:.5f} rel={rel:.6f} sd={sf:6.3f}  "+" ".join(f"{bb:+.4g}*{t}" for bb,t in zip(b,ts)))
json.dump([{'loo':float(l),'rel':float(r),'sd':float(s),'terms':t,'beta':[float(x) for x in bb]} for l,r,s,t,bb in res[:20]],
          open('artifacts/atoms2_top.json','w'),indent=1)
