"""Diagnostic trajectory selected transparently from the complete stress sweep."""
import json
import numpy as np
import pandas as pd
from kernel import *
frame=pd.read_csv(DATA/'stress_sweep.csv')
active=frame[(frame.code>0)&frame.candidate_finite&np.isfinite(frame.energy_candidate)&(frame.amplitude<=.3)].copy()
selection_source='prespecified 360-case sweep'
if active.empty:
    # The declared sweep has no moderate-amplitude activation. A separately
    # retained, explicitly exploratory extension probes much smaller steps;
    # it is an algorithm stress diagnostic, not generalization evidence.
    extension=[];kk=Kernel();rr=np.load(DATA/'legacy_filter.npz')['R']
    for amp in [.08,.3]:
        for dt0 in [1e-7,1e-8]:
            for i0 in [3,4,5]:
                uu=legacy_ic(i0,mean=1.,amplitude=amp,seed=SEED+500000+i0)
                cc=kk.semi(uu,dt0,6);cand=cc+np.fft.ifft2(rr*np.fft.fft2(cc)).real
                ww,ll=kk.safeguard(uu,cand)
                extension.append(dict(amplitude=amp,mean=1.,eps=.02,dt=dt0,test_id=i0,candidate_finite=True,**ll))
    ex=pd.DataFrame(extension);ex.to_csv(DATA/'stress_smallstep_extension.csv',index=False)
    active=ex[(ex.code>0)&np.isfinite(ex.energy_candidate)].copy()
    selection_source='exploratory 12-case small-step extension after no moderate-amplitude activation in the original sweep'
if active.empty:raise RuntimeError('No finite small-step activation; retain negative test and report separately.')
# Prefer the nominal epsilon and the smaller amplitude; then prefer a smaller
# macro-step deviation. Selection is explicitly diagnostic, not a random sample.
active['priority']=abs(np.log(active.eps/.02))+abs(np.log(active.amplitude/.08))+abs(np.log(active.dt/DT))
case=active.sort_values(['priority','test_id']).iloc[0]
k=Kernel(eps=float(case['eps']));dt=float(case['dt']);r=np.load(DATA/'legacy_filter.npz')['R'];i=int(case['test_id'])
u=legacy_ic(i,mean=float(case['mean']),amplitude=float(case.amplitude),seed=SEED+500000+i)
states={s:u.copy() for s in ['safe','unsafe','C6']};v=k.fft(u);v2=v.copy()
rows=[];saved={'initial':u};failed=False
for step in range(1,351):
    old=states['safe'].copy()
    states['safe'],log=k.surrogate(old,r,dt)
    with np.errstate(over='ignore',invalid='ignore'):
        if not failed:states['unsafe'],_=k.surrogate(states['unsafe'],r,dt,safe=False)
        failed=failed or not np.isfinite(states['unsafe']).all()
    states['C6']=k.semi(states['C6'],dt,6)
    sub=max(4,int(np.ceil(dt/2.5e-5)))
    for _ in range(sub):v=k.etd(v,dt/sub)
    for _ in range(2*sub):v2=k.etd(v2,dt/(2*sub))
    ref=k.ifft(v2)
    row=dict(step=step,time=step*dt,**log,reference_temporal_difference=rel(k.ifft(v),ref),unsafe_failed=failed)
    for name,f in states.items():
        with np.errstate(over='ignore',invalid='ignore'):
            row[f'error_{name}']=rel(f,ref) if np.isfinite(f).all() else float('nan')
            row[f'energy_{name}']=k.energy(f) if np.isfinite(f).all() else float('nan')
    rows.append(row)
    if step in [1,35,350]:
        saved[f'reference_{step}']=ref
        for name,f in states.items():saved[f'{name}_{step}']=f.copy()
pd.DataFrame(rows).to_csv(DATA/'stress_rollout.csv',index=False)
np.savez_compressed(DATA/'stress_rollout_fields.npz',**saved)
meta=dict(case=case.drop(labels=['priority']).to_dict(),selection_source=selection_source,selection='minimum stated priority among finite activating cases with amplitude <=.3',
    steps=350,backtracking_events=sum(r['code']>0 for r in rows),fallback_events=sum(r['code']==35 for r in rows),
    postmean_acceptance_violations=sum(r['post_correction_acceptance_violation'] for r in rows),final=rows[-1])
(DATA/'stress_rollout_summary.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2),flush=True)
