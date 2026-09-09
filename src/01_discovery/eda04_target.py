import pandas as pd, numpy as np
tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv')
F=[f'feature_{i:02d}' for i in range(1,26)]
full=pd.concat([tr,te],ignore_index=True)
d={c:full[c].values for c in F}

print("=== MOMENT-MATCH SEARCH for NDEM structure ===")
print("Organizer: mean~3.9, std~4.6, range ~[-20,+20]")
# complete-case rows for the core columns
core=['feature_14','feature_15','feature_11','feature_22','feature_19','feature_09','feature_08']
m=np.ones(len(full),bool)
for c in core: m&=np.isfinite(d[c])
print("complete-case rows for core cols:", m.sum())
G=d['feature_14'][m]; TL=d['feature_15'][m]; IE=d['feature_11'][m]; XL=d['feature_22'][m]
DS=d['feature_19'][m]; DE=d['feature_09'][m]; SOC=d['feature_08'][m]
usable=G*TL*IE*(1-XL)
print(f"  usable = f14*f15*f11*(1-f22):  mean={usable.mean():.4f} std={usable.std():.4f} min={usable.min():.3f} max={usable.max():.3f}")
for nm,term in [('none',0),('1.0*f19',DS),('0.25*f19',0.25*DS),('0.5*f19',0.5*DS),
                ('1.0*f09',DE),('0.5*f09',0.5*DE),('f09*(1-f08)',DE*(1-SOC)),
                ('0.5*f09*(1-f08)',0.5*DE*(1-SOC)),('2*f19',2*DS)]:
    y=usable-term
    print(f"  minus {nm:18s}: mean={np.mean(y):7.4f} std={np.std(y):7.4f} min={np.min(y):8.3f} max={np.max(y):7.3f}")

print("\n=== which multiplicative subset best matches (mean 3.9, std 4.6)? ===")
import itertools
mults={'f15':TL,'f11':IE,'(1-f22)':1-XL}
for r in range(4):
    for combo in itertools.combinations(mults,r):
        y=G.copy()
        for c in combo: y=y*mults[c]
        print(f"  f14*{'*'.join(combo) if combo else '1':22s}: mean={y.mean():7.4f} std={y.std():7.4f} max={y.max():7.3f}")

print("\n=== NEGATIVE TAIL: target min ~ -20. What produces -20? ===")
print("  usable min:", usable.min(), " -> need a large negative contribution")
print("  f09 max:", np.nanmax(d['feature_09']), " f19 max:", np.nanmax(d['feature_19']))
print("  => a -20 floor CANNOT come from demand terms (max ~1.4). Check: does target range come from noise?")
print("  usable percentiles:", np.round(np.percentile(usable,[0,0.1,1,5,50,95,99,99.9,100]),3))
