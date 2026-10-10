"""Declared one-step stress sweep plus controlled damping/fallback tests."""
import json
import numpy as np
import pandas as pd
from kernel import *

r=np.load(DATA/'legacy_filter.npz')['R'];rows=[];saved={}
# A priori Cartesian sweep, retaining every result rather than only activations.
for amp in [.08,.3,.6,1.0,1.5]:
    for eps in [.01,.02,.04]:
        for dt in [1e-6,1e-4,1e-3,1e-2]:
            for i in range(6):
                k=Kernel(eps=eps);u=legacy_ic(i,mean=0,amplitude=amp,seed=SEED+500000+i)
                mean=0. if i<3 else 1.
                u+=mean
                with np.errstate(over='ignore',invalid='ignore'):
                    c=k.semi(u,dt,6);candidate=c+np.fft.ifft2(r*np.fft.fft2(c)).real
                    finite=bool(np.isfinite(candidate).all())
                    w,log=k.safeguard(u,candidate)
                rows.append(dict(amplitude=amp,mean=mean,eps=eps,dt=dt,test_id=i,candidate_finite=finite,**log))
                if log['code'] and finite:
                    saved[f'u_{len(rows)-1}']=u;saved[f'candidate_{len(rows)-1}']=candidate;saved[f'returned_{len(rows)-1}']=w
pd.DataFrame(rows).to_csv(DATA/'stress_sweep.csv',index=False)
np.savez_compressed(DATA/'stress_active_fields.npz',**saved)

# Controlled mean-zero candidate with a finite fallback at all 34 trial lengths.
k=Kernel();x=2*np.pi*np.arange(64)/64;d=np.cos(x)[:,None]*np.ones((1,64))
u=np.ones((64,64));candidate=u+1e8*d
w,log=k.safeguard(u,candidate)
assert log['code']==35 and np.array_equal(w,u)
tests=[dict(test='forced_fallback',**log)]
# Starting near unstable zero, an overshoot in the same low-frequency direction
# has a descending segment: backtracking must accept a genuinely moving step.
u=.08*d;candidate=3*d
w,log=k.safeguard(u,candidate)
assert 0<log['code']<35 and rel(w,u)>0 and log['energy_returned']<=log['energy_old']+1e-14
tests.append(dict(test='controlled_overshoot',**log))
pd.DataFrame(tests).to_csv(DATA/'controlled_safeguard_tests.csv',index=False)
print(json.dumps(dict(cases=len(rows),active_finite=sum(r['code']>0 and r['candidate_finite'] for r in rows),
    nonfinite_candidates=sum(not r['candidate_finite'] for r in rows),controlled=tests)),flush=True)
