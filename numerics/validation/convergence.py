"""Checkpointed reference convergence; configurations fixed before execution.

Shared continuous initial data on all grids, Galerkin cubic with 2N padding.
Fourier interpolation from smaller to larger grid measures full-field error,
including unresolved high modes. Temporal refinement is separate.
"""
import argparse,json,time
import numpy as np
from kernel import *
from kernel_fast import Kernel

TIMES=[.0035,.035,.07]
def initial(name,n):
    return deterministic(n) if name=='rectangles' else continuous_random(n,int(name[-1]))
def solve(name,n,h,kind='galerkin'):
    path=DATA/f'convergence_{name}_{kind}_N{n}_h{h:.8g}.npz'
    if path.exists():return path
    k=Kernel(n,kind=kind);u=initial(name,n)
    t=time.perf_counter();fields=run_etd(k,u,TIMES,h)
    elapsed=time.perf_counter()-t
    np.savez_compressed(path,**{f'u{j}':fields[t] for j,t in enumerate([0.]+TIMES)},times=[0.]+TIMES,
        n=n,h=h,kind=kind,case=name,seconds=elapsed,
        energies=[k.energy(fields[t]) for t in [0.]+TIMES],
        means=[fields[t].mean() for t in [0.]+TIMES])
    print(json.dumps(dict(stage='convergence',case=name,n=n,h=h,kind=kind,seconds=elapsed)),flush=True)
    return path

def summarize():
    rows=[]
    for name in ['random0','random1','random2','rectangles']:
        for n in [64,128,256,512,768,1024]:
            files=sorted(DATA.glob(f'convergence_{name}_galerkin_N{n}_h*.npz'))
            if not files:continue
            runs=sorted([np.load(f) for f in files],key=lambda a:float(a['h']))
            best=runs[0]
            for run in runs[1:]:
                for j,t in enumerate(TIMES,1):
                    rows.append(dict(comparison='temporal',case=name,n=n,h=float(run['h']),fine_h=float(best['h']),time=t,
                        field_error=rel(run[f'u{j}'],best[f'u{j}']),energy_abs_difference=abs(float(run['energies'][j]-best['energies'][j]))))
            for nf in [128,256,512,768,1024]:
                if nf<=n:continue
                finefiles=list(DATA.glob(f'convergence_{name}_galerkin_N{nf}_h*.npz'))
                if not finefiles:continue
                fine=min([np.load(f) for f in finefiles],key=lambda a:float(a['h']))
                for j,t in enumerate([0.]+TIMES):
                    mapped=Kernel(nf).ifft(resize_hat(Kernel(n).fft(best[f'u{j}']),nf))
                    rows.append(dict(comparison='spatial',case=name,n=n,fine_n=nf,h=float(best['h']),fine_h=float(fine['h']),time=t,
                        field_error=rel(mapped,fine[f'u{j}']),energy_abs_difference=abs(float(best['energies'][j]-fine['energies'][j]))))
            legacyfile=DATA/f'convergence_{name}_legacy_N{n}_h{float(best["h"]):.8g}.npz'
            if legacyfile.exists():
                legacy=np.load(legacyfile)
                for j,t in enumerate(TIMES,1):
                    rows.append(dict(comparison='aliasing_and_nyquist',case=name,n=n,h=float(best['h']),time=t,
                        field_error=rel(legacy[f'u{j}'],best[f'u{j}'])))
    import pandas as pd
    pd.DataFrame(rows).to_csv(DATA/'convergence_comparisons.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',default='pilot');p.add_argument('--case',default='random0');p.add_argument('--n',type=int,default=512);p.add_argument('--h',type=float,default=5e-5);a=p.parse_args()
    if a.stage=='custom':solve(a.case,a.n,a.h)
    elif a.stage=='pilot':
        for name in ['random0','rectangles']:
            for n in [64,128,256]:
                for h in [1e-4,5e-5]:solve(name,n,h)
    elif a.stage=='fine':
        for name in ['random0','rectangles']:
            for h in [1e-4,5e-5]:solve(name,512,h)
    elif a.stage=='ensemble':
        for name in ['random1','random2']:
            for n in [64,128,256,512]:solve(name,n,5e-5)
    elif a.stage=='alias':
        for name in ['random0','rectangles']:
            for n in [64,128,256]:solve(name,n,5e-5,'legacy')
    summarize()
