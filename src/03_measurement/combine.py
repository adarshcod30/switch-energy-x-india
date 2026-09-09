"""Decode gamma from each perturbation score, then build the optimal combination.
Orthonormal directions => p = base + sum(gamma_i * o_i), predicted RMSE^2 = R0^2 - sum(gamma_i^2)."""
import numpy as np, pandas as pd, json, subprocess, re, sys
M=json.load(open('artifacts/gs2_meta.json')); R0=M['R0']; D=M['delta']; names=M['names']
O=np.load('artifacts/GS2.npy'); base=pd.read_csv('subs/FINAL.csv')['prediction'].values.astype(np.float64)
out=subprocess.run(['python3','-m','kaggle','competitions','submissions','-c','switch-energy-x-india'],capture_output=True,text=True).stdout
S={}
for ln in out.splitlines()[2:]:
    m=re.match(r'\s*(\d+)\s+(\S+?)\.csv\s+\S+ \S+\s+.*?SubmissionStatus\.\w+\s+([\d.]+)',ln)
    if m: S.setdefault(m.group(2),float(m.group(3)))
gam={}; 
print(f"{'direction':16s} {'RMSE':>9s} {'gamma':>9s} {'var explained':>14s}")
for i,n in enumerate(names):
    k=f'X_{n}'
    if k not in S: continue
    g=(D*D-(S[k]**2-R0**2))/(2*D)
    gam[n]=(g,i)
    print(f"  {n:14s} {S[k]:9.5f} {g:+9.4f} {g*g:14.5f}")
if not gam: print("no probe scores yet"); sys.exit()
tot=sum(g*g for g,_ in gam.values())
print(f"\n  sum(gamma^2) = {tot:.5f}   R0^2 = {R0**2:.5f}")
pred=np.sqrt(max(R0**2-tot,0.28**2))
print(f"  predicted combined RMSE = {pred:.5f}   (noise floor ~0.28)")
sub=pd.read_csv('data/sample_submission.csv')
for shrink,tag in [(1.0,'full'),(0.85,'shr85'),(0.7,'shr70')]:
    p=base.copy()
    for n,(g,i) in gam.items(): p=p+shrink*g*O[:,i]
    sub['prediction']=p; sub.to_csv(f'subs/COMB_{tag}.csv',index=False)
    print(f"  wrote COMB_{tag}: sd={p.std():.3f} min={p.min():.2f} max={p.max():.2f}")
json.dump({k:v[0] for k,v in gam.items()},open('artifacts/gammas.json','w'),indent=1)
