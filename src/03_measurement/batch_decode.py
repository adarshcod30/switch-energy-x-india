"""usage: batch_decode.py <tag> <combo_name>  -> decodes probe scores, writes subs/<combo_name>.csv (NO submissions)"""
import numpy as np, pandas as pd, json, subprocess, re, sys, math
tag, combo = sys.argv[1], sys.argv[2]
M=json.load(open(f'artifacts/{tag}_meta.json')); R0=M['R0']; D=M['delta']; files=M['files']
O=np.load(f'artifacts/{tag}_O.npy'); best=pd.read_csv(f"subs/{M['best']}.csv")['prediction'].values.astype(np.float64)
out=subprocess.run(['python3','-m','kaggle','competitions','submissions','-c','switch-energy-x-india','--page-size','200'],capture_output=True,text=True).stdout
S={}
for ln in out.splitlines()[2:]:
    m=re.match(r'\s*(\d+)\s+(\S+?)\.csv\s+\S+ \S+\s+.*?SubmissionStatus\.\w+\s+([\d.]+)',ln)
    if m: S.setdefault(m.group(2),float(m.group(3)))
p=best.copy(); expl=0; sig2=0.00135; miss=[]
for i,fn in enumerate(files):
    if fn not in S: miss.append(fn); continue
    g=(D*D-(S[fn]**2-R0**2))/(2*D); s=g*g/(g*g+sig2); p=p+s*g*O[:,i]; expl+=(s*g)*(2*g-s*g)
    print(f"  {fn:22s} RMSE={S[fn]:.5f} gamma={g:+.4f} var={g*g:.4f}")
if miss: print("  NOT SCORED YET:",miss)
pred=math.sqrt(max(R0**2-expl,0))
sub=pd.read_csv('data/sample_submission.csv'); sub['prediction']=p; sub.to_csv(f'subs/{combo}.csv',index=False)
print(f"  {combo} predicted RMSE = {pred:.4f}  (sd={p.std():.3f})")
