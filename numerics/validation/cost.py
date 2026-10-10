"""Same-environment achieved-error / wall-time comparison with repetition.

All methods use the legacy spatial operator and identical initial states.
Five initial states and five repeats per method; warmup each configuration.
Surrogate includes its intrinsic checks. Other methods need no safeguards;
all are assessed with common output diagnostics outside the timed loop.
"""
import time,json,platform,os,sys
import numpy as np
import pandas as pd
from kernel import *

k=Kernel();fit=np.load(DATA/'legacy_filter.npz');r=fit['R'];a=fit['A_pidmd'];T=.0035
def timed_surrogate(u):
    # Original online state machine; campaign-only logging and the extra
    # post-correction diagnostic energy evaluation are outside this timing.
    c=k.semi(u,DT,6)
    v=c+np.fft.ifft2(r*np.fft.fft2(c)).real
    mean=float(u.mean());v+=mean-float(v.mean());e0=k.energy(u)
    if k.energy(v)<=e0+1e-14:out=v
    else:
        out=u.copy()
        for j in range(1,35):
            w=u+2.**(-j)*(v-u);w+=mean-float(w.mean())
            if k.energy(w)<=e0+1e-14:out=w;break
    out+=mean-float(out.mean())
    return out
configs=[('C'+str(q),'semi',q) for q in [1,6,12,24,100]]
configs += [('residual','residual',0),('piDMD','piDMD',0)]
configs += [(f'ETD_h{h:g}','etd',h) for h in [4e-4,2e-4,1e-4]]
configs += [(f'adaptive_{meth}_tol{tol:g}','adaptive_'+meth,tol) for meth in ['semi','etd'] for tol in [1e-2,1e-3,1e-4]]

def run(u,kind,value):
    info={}
    if kind.startswith('adaptive_'):
        out,info=adaptive(k,u,T,value,method=kind.split('_')[1]);info.pop('history');return out,info
    if kind=='etd':
        v=k.fft(u);steps=int(np.ceil(T/value-1e-10))
        for j in range(steps):v=k.etd(v,value if j<steps-1 else T-value*(steps-1))
        return k.ifft(v),dict(steps=steps)
    out=u.copy()
    for step in range(35):
        if kind=='semi':out=k.semi(out,DT,value)
        elif kind=='residual':out=timed_surrogate(out)
        else:out=np.fft.ifft2(a*np.fft.fft2(out)).real
    return out,info

rows=[]
for i in range(5):
    u=legacy_ic(i)
    ref=run_etd(k,u,[T],1.25e-5)[T]
    for name,kind,value in configs:run(u,kind,value)
    # Rotating configuration order reduces a systematic warm-cache/order bias.
    for rep in range(5):
        order=configs[rep:]+configs[:rep]
        for name,kind,value in order:
            t=time.perf_counter();out,info=run(u,kind,value);seconds=time.perf_counter()-t
            rows.append(dict(test_id=i,repeat=rep,method=name,seconds=seconds,l2=rel(out,ref),structure_error=rel(structure(k,out),structure(k,ref)),**info))
    print(json.dumps(dict(stage='cost',test_id=i)),flush=True)
    pd.DataFrame(rows).to_csv(DATA/'cost_raw.csv',index=False)
frame=pd.DataFrame(rows)
summary=frame.groupby('method').agg(seconds_mean=('seconds','mean'),seconds_sd=('seconds','std'),seconds_min=('seconds','min'),
    seconds_max=('seconds','max'),l2_mean=('l2','mean'),structure_mean=('structure_error','mean'))
summary.to_csv(DATA/'cost_summary.csv');print(summary.to_string(),flush=True)
(DATA/'cost_protocol.json').write_text(json.dumps(dict(hardware='Intel Core i7-4770K @ 3.50GHz',logical_processors=os.cpu_count(),
    physical_memory_gib=15.93956,python=sys.version,numpy=np.__version__,platform=platform.platform(),
    precision='float64/complex128',fft='NumPy pocketfft',other_campaign_jobs_running=False,
    thread_policy='No BLAS matrix products or explicit FFT worker threads; operating-system scheduling, clock and other desktop applications not controlled',
    trajectories=5,repeats=5,warmup='one complete rollout per configuration per initial state',
    statistic='sample SD across 25 measured times; errors average the five distinct fields',
    fixed_ETD_coefficients='cached after warmup; cache can evict under changing adaptive steps, such work is timed',
    adaptive='step doubling, accepted two-half-step field, p=1 or 4, atol=rtol*1e-3, RMS scaled norm, safety=.9, factor range [.2,5], initial h=1e-4,max h=5e-4',
    reference='same-grid ETDRK4 h=1.25e-5; spatial error excluded from accuracy-cost interpretation'),indent=2))
