"""Original first-order reference at 100,200,400 substeps vs ETDRK4."""
import numpy as np
import pandas as pd
from kernel import *
rows=[]
for case in ['test0','test1','test2','rectangles']:
    k=Kernel();u=deterministic(64) if case=='rectangles' else legacy_ic(int(case[-1]))
    ref=run_etd(k,u,[.0035],1.25e-5)[.0035]
    ref2=run_etd(k,u,[.0035],6.25e-6)[.0035]
    for q in [100,200,400]:
        out=u.copy()
        for _ in range(35):out=k.semi(out,DT,q)
        rows.append(dict(case=case,substeps=q,l2_vs_etd=rel(out,ref2),structure_vs_etd=rel(structure(k,out),structure(k,ref2)),
            energy_abs_difference=abs(k.energy(out)-k.energy(ref2)),etd_temporal_check=rel(ref,ref2)))
    print(case,flush=True)
pd.DataFrame(rows).to_csv(DATA/'reference_substeps.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False),flush=True)
