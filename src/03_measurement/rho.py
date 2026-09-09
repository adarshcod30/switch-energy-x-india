import subprocess, re, pandas as pd, numpy as np
out=subprocess.run(['python3','-m','kaggle','competitions','submissions','-c','switch-energy-x-india'],
                   capture_output=True,text=True).stdout
rows=[]
for ln in out.splitlines()[2:]:
    m=re.match(r'\s*(\d+)\s+(\S+)\s+([\d\-]+ [\d:.]+)\s+(.*?)\s+SubmissionStatus\.(\w+)\s+([\d.]+)?',ln)
    if m and m.group(6): rows.append((m.group(2),float(m.group(6))))
SY,MY=4.6,3.9
print(f"{'submission':28s} {'RMSE':>8s} {'implied rho':>12s}   (calibrated preds: mean=3.9 sd=4.6)")
for f,s in rows:
    rho=1-s*s/(2*SY*SY)
    print(f"  {f:26s} {s:8.5f} {rho:12.4f}")
