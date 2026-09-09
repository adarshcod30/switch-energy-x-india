"""usage: batch_build.py <best_csv_name> <R0> <tag>   -> writes subs/<tag>_*.csv probes + artifacts/<tag>_meta.json (NO submissions)"""
import numpy as np, pandas as pd, json, subprocess, re, os, sys
best_name, R0, tag = sys.argv[1], float(sys.argv[2]), sys.argv[3]; D=0.40; SY,MY=4.611,3.9
out=subprocess.run(['python3','-m','kaggle','competitions','submissions','-c','switch-energy-x-india','--page-size','200'],capture_output=True,text=True).stdout
S={}
for ln in out.splitlines()[2:]:
    m=re.match(r'\s*(\d+)\s+(\S+?)\.csv\s+\S+ \S+\s+.*?SubmissionStatus\.\w+\s+([\d.]+)',ln)
    if m: S.setdefault(m.group(2),float(m.group(3)))
names=[];P=[];rm=[]
for n,sc in S.items():
    fp=f'subs/{n}.csv'
    if os.path.exists(fp):
        v=pd.read_csv(fp)['prediction'].values.astype(np.float64)
        if np.all(np.isfinite(v)): names.append(n);P.append(v);rm.append(sc)
P=np.column_stack(P); rm=np.array(rm)
np.save('artifacts/P_all.npy',P); np.save('artifacts/covy_all.npy',(SY**2+P.var(0)+(MY-P.mean(0))**2-rm**2)/2); json.dump(names,open('artifacts/names_all.json','w'))
Pc=P-P.mean(0); Q,_=np.linalg.qr(np.column_stack([np.ones(len(P)),Pc]))
base=pd.read_csv('subs/FINAL.csv')['prediction'].values.astype(np.float64)
best=pd.read_csv(f'subs/{best_name}.csv')['prediction'].values.astype(np.float64); corr=best-base
X=pd.read_parquet('artifacts/Xv3.parquet').iloc[400000:].reset_index(drop=True)
d={c:X[c].values.astype(np.float64) for c in X.columns}; f=lambda k:d[f'feature_{k:02d}']
TP,T2,G,DE,V,W,XL,RH,GR=f(7),f(2),f(1),f(9),f(4),f(13),f(22),f(3),f(14)
def spl(x,kn): return np.column_stack([np.ones(len(x)),x]+[np.clip(x-k,0,None)**3 for k in kn])
def q(x,n): return list(np.quantile(x,np.linspace(0.1,0.9,n)))
S07=spl(TP,q(TP,12)); res=corr-S07@np.linalg.lstsq(S07,corr,rcond=None)[0]
fr=lambda A: A@np.linalg.lstsq(A,res,rcond=None)[0]
ten=lambda a,ka,b,kb: fr(np.column_stack([u*v for u in spl(a,q(a,ka)).T for v in spl(b,q(b,kb)).T]))
C={'s07':S07@np.linalg.lstsq(S07,corr,rcond=None)[0],'s01':fr(spl(G,q(G,9))),'s02':fr(spl(T2,q(T2,8))),'s09':fr(spl(DE,q(DE,8))),
   's04':fr(spl(V,q(V,8))),'s03':fr(spl(RH,q(RH,7))),'s22':fr(spl(XL,q(XL,7))),'s13':fr(spl(W,q(W,8))),'sdT':fr(spl(TP-T2,q(TP-T2,8))),
   't0702':ten(TP,4,T2,4),'t0702b':ten(TP,6,T2,5),'t0102':ten(G,3,T2,3),'t0704':ten(TP,3,V,3),'t0122':ten(G,3,XL,3),
   't0709':ten(TP,3,DE,3),'t0913':ten(DE,3,W,3),'t0113':ten(G,3,W,3),'t0213':ten(T2,3,W,3),'t0722':ten(TP,3,XL,3),'t0203':ten(T2,3,RH,3)}
rows=[]
for n,v in C.items():
    v=np.nan_to_num(v); o=v-Q@(Q.T@v); nfr=o.std()/max(v.std(),1e-12)
    A=np.column_stack([np.ones(len(v)),v]); cc,*_=np.linalg.lstsq(A,res,rcond=None); r2=1-(res-A@cc).var()/res.var()
    rows.append((nfr*max(r2,0.005),nfr,r2,n))
rows.sort(reverse=True)
for s,nfr,r2,n in rows[:8]: print(f"  {n:8s} new-frac={nfr:.3f} R2={r2:.3f}")
pick=[n for _,_,_,n in rows[:4]]; Bcur=Q.copy(); vecs=[]
for n in pick:
    v=np.nan_to_num(C[n]).astype(np.float64); o=v-Bcur@(Bcur.T@v); o=o/o.std(); vecs.append(o); Bcur=np.column_stack([Bcur,o/np.linalg.norm(o)])
sub=pd.read_csv('data/sample_submission.csv'); files=[]
for n,o in zip(pick,vecs):
    p=best+D*o; assert np.isfinite(p).all() and len(p)==100000; sub['prediction']=p; fn=f'{tag}_{n}'; sub.to_csv(f'subs/{fn}.csv',index=False); files.append(fn)
np.save(f'artifacts/{tag}_O.npy',np.column_stack(vecs)); json.dump({'R0':R0,'delta':D,'files':files,'best':best_name},open(f'artifacts/{tag}_meta.json','w'),indent=1)
print(f"span={P.shape[1]}  probes written:",files)
