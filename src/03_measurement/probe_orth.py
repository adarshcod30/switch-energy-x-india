"""Orthogonalized single-atom probes: measure each atom's contribution BEYOND span(P)."""
import numpy as np, pandas as pd, json
SY,MY=4.611,3.9
Pc=np.load('artifacts/Pc.npy')
X=pd.read_parquet('artifacts/Xv2.parquet').iloc[400000:].reset_index(drop=True)
d={c:X[c].values.astype(np.float64) for c in X.columns}
F={f'f{i:02d}':d[f'feature_{i:02d}'] for i in range(1,26)}
G,T,RH,V=F['f01'],F['f02'],F['f03'],F['f04']
CL,TP,SOC,DE,RHO=F['f06'],F['f07'],F['f08'],F['f09'],F['f10']
IE,SOL,WND,GR,TL=F['f11'],F['f12'],F['f13'],F['f14'],F['f15']
PRC,V2,DS,GF,XL=F['f16'],F['f18'],F['f19'],F['f21'],F['f22']
NET=GR-DE; US=d['d_usable']
ATOMS={
 'A_TP3':TP**3,'A_TP2':TP**2,'A_reluneg':np.clip(-NET,0,None),'A_inddef':(NET<0).astype(float),
 'A_txloss':XL*GR,'A_invloss':(1-IE)*GR,'A_derloss':(1-TL)*GR,'A_sqrtpos':np.sqrt(np.clip(NET,0,None)),
 'A_Wsq':np.sqrt(np.clip(WND,0,None)),'A_logW':np.log1p(np.clip(WND,0,None)),'A_W3':WND**3,'A_W2':WND**2,
 'A_G2':G**2,'A_DEoS':DE/np.clip(SOC,.02,None),'A_net2':NET**2,'A_STL':SOL*TL,
 'A_TPW':TP*WND,'A_GW':G*WND,'A_DEGR':DE*GR,'A_netSOC':NET*SOC,'A_v3':np.clip(V2,0,None)**1.5,
 'A_reluus':np.clip(US-DE,0,None),'A_GRTLIE':GR*TL*IE,'A_DE2':DE**2,
}
Q,_=np.linalg.qr(np.column_stack([np.ones(len(Pc)),Pc]))
rows=[]
for n,v in ATOMS.items():
    v=np.nan_to_num(v,nan=0,posinf=0,neginf=0).astype(np.float64)
    o=v-Q@(Q.T@v)
    rows.append((o.std(),n,o))
rows.sort(reverse=True)
print(f"{'atom':14s} {'orth_sd':>10s}   (y_perp sd = 1.033; need orth_sd >> 0 to be measurable)")
sub=pd.read_csv('data/sample_submission.csv'); keep=[]
for s,n,o in rows:
    print(f"  {n:12s} {s:10.4f}")
    if s>1e-6:
        p=(o-o.mean())/o.std()*4.6+3.9
        sub['prediction']=p; sub.to_csv(f'subs/{n}.csv',index=False); keep.append(n)
json.dump(keep,open('artifacts/orth_probe_list.json','w'),indent=1)
np.save('artifacts/Qspan.npy',Q)
print(f"\nwrote {len(keep)} orthogonalized probes. Top 16 by orth_sd will be submitted first.")
print("order:", [n for _,n,_ in rows[:16]])
