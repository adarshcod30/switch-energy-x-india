import numpy as np, pandas as pd, json, subprocess, re
R0=1.07022; D=0.40
base=pd.read_csv('subs/FINAL.csv')['prediction'].values.astype(np.float64)
O2=np.load('artifacts/GS2.npy'); n2=json.load(open('artifacts/GS2_names.json'))
O3=np.load('artifacts/GS3.npy'); n3=json.load(open('artifacts/GS3_names.json'))
g1=json.load(open('artifacts/gammas.json'))
out=subprocess.run(['python3','-m','kaggle','competitions','submissions','-c','switch-energy-x-india'],capture_output=True,text=True).stdout
S={}
for ln in out.splitlines()[2:]:
    m=re.match(r'\s*(\d+)\s+(\S+?)\.csv\s+\S+ \S+\s+.*?SubmissionStatus\.\w+\s+([\d.]+)',ln)
    if m: S.setdefault(m.group(2),float(m.group(3)))
print(f"{'thermal direction':16s} {'RMSE':>9s} {'gamma':>9s} {'var expl':>10s}")
g3={}
for i,n in enumerate(n3):
    k=f'Y_{n}'
    if k not in S: continue
    g=(D*D-(S[k]**2-R0**2))/(2*D); g3[n]=(g,i)
    print(f"  {n:14s} {S[k]:9.5f} {g:+9.4f} {g*g:10.5f}")
tot1=sum(v*v for v in g1.values()); tot3=sum(g*g for g,_ in g3.values())
print(f"\n  batch1 sum(g^2)={tot1:.5f}   batch2 sum(g^2)={tot3:.5f}   TOTAL={tot1+tot3:.5f}")
print(f"  predicted combined RMSE = {np.sqrt(max(R0**2-tot1-tot3,0.28**2)):.5f}")
sub=pd.read_csv('data/sample_submission.csv')
for shrink,tag in [(1.0,'A'),(0.9,'B')]:
    p=base.copy()
    for n,v in g1.items(): p=p+shrink*v*O2[:,n2.index(n)]
    for n,(g,i) in g3.items(): p=p+shrink*g*O3[:,i]
    sub['prediction']=p; sub.to_csv(f'subs/COMB2_{tag}.csv',index=False)
    print(f"  wrote COMB2_{tag}: sd={p.std():.3f} min={p.min():.2f} max={p.max():.2f}")
json.dump({k:v[0] for k,v in g3.items()},open('artifacts/gammas3.json','w'),indent=1)
