"""Rebuild orthogonalized atom probes on Xv3 (phase-aware imputation) + phase probes."""
import numpy as np, pandas as pd, json
Pc=np.load('artifacts/Pc.npy')
X=pd.read_parquet('artifacts/Xv3.parquet').iloc[400000:].reset_index(drop=True)
d={c:X[c].values.astype(np.float64) for c in X.columns}
F={f'f{i:02d}':d[f'feature_{i:02d}'] for i in range(1,26)}
G,T,RH,V=F['f01'],F['f02'],F['f03'],F['f04']
TP,SOC,DE=F['f07'],F['f08'],F['f09']
IE,SOL,WND,GR,TL=F['f11'],F['f12'],F['f13'],F['f14'],F['f15']
V2,DS,XL=F['f18'],F['f19'],F['f22']
NET=GR-DE; US=d['d_usable']
ATOMS={
 'A_TP2':TP**2,'A_logW':np.log1p(np.clip(WND,0,None)),'A_TPW':TP*WND,'A_txloss':XL*GR,
 'A_DEGR':DE*GR,'A_inddef':(NET<0).astype(float),'A_G2':G**2,'A_DE2':DE**2,'A_STL':SOL*TL,
 'A_derloss':(1-TL)*GR,'A_v3':np.clip(V2,0,None)**1.5,'A_W3':WND**3,
 'A_reluneg':np.clip(-NET,0,None),'A_GW':G*WND,'A_GRTLIE':GR*TL*IE,'A_DEoS':DE/np.clip(SOC,.02,None),
 'A_ph08':d['p08_s'],'A_ph08c':d['p08_c'],'A_ph09':d['p09_s'],'A_ph09c':d['p09_c'],
}
Q,_=np.linalg.qr(np.column_stack([np.ones(len(Pc)),Pc]))
sub=pd.read_csv('data/sample_submission.csv'); made=[]
for n,v in ATOMS.items():
    v=np.nan_to_num(v,nan=0,posinf=0,neginf=0).astype(np.float64)
    o=v-Q@(Q.T@v)
    if o.std()<1e-9: print(f"  skip {n} (no orthogonal component)"); continue
    p=(o-o.mean())/o.std()*4.6+3.9
    sub['prediction']=p; sub.to_csv(f'subs/{n}.csv',index=False); made.append(n)
# combine the two phase sin/cos into single probes (2 subs not 4)
for base,(s,c) in {'A_PH08':('A_ph08','A_ph08c'),'A_PH09':('A_ph09','A_ph09c')}.items():
    a=pd.read_csv(f'subs/{s}.csv')['prediction'].values
    b=pd.read_csv(f'subs/{c}.csv')['prediction'].values
    v=(a-a.mean())/a.std()+(b-b.mean())/b.std()
    o=v-Q@(Q.T@v); p=(o-o.mean())/o.std()*4.6+3.9
    sub['prediction']=p; sub.to_csv(f'subs/{base}.csv',index=False)
order=['A_TP2','A_logW','A_TPW','A_txloss','A_DEGR','A_inddef','A_G2','A_DE2','A_STL',
       'A_derloss','A_v3','A_W3','A_reluneg','A_GW','A_PH08','A_PH09']
json.dump(order,open('artifacts/probe_order.json','w'),indent=1)
ref=pd.read_csv('data/sample_submission.csv'); bad=0
for n in order:
    s=pd.read_csv(f'subs/{n}.csv')
    if not(s.shape==(100000,2) and (s.row_id.values==ref.row_id.values).all() and np.isfinite(s.prediction).all()): bad+=1;print("BAD",n)
print(f"rebuilt on Xv3; final probe order ({len(order)}), {bad} problems:")
print("  "+", ".join(order))
