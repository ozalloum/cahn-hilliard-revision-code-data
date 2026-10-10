"""Separate dealiased formulation: frozen legacy transfer versus new fits.

New training uses 60 independent continuous random fields, 16 states each,
ETDRK4 teachers h=2.5e-5 and the same six-substep coarse propagator. The fixed
cutoff 28 is retained to distinguish spatial changes from mode-count tuning.
Both numerator and denominator use NumPy unnormalized FFTs, lambda=1e-12.
"""
import argparse,time,json
import numpy as np
import pandas as pd
from kernel import *
from kernel_fast import Kernel

def fit(n):
    path=DATA/f'galerkin_filter_N{n}.npz'
    if path.exists():return
    k=Kernel(n,kind='galerkin');num=np.zeros((n,n),complex);den=np.zeros((n,n));dn=num.copy();dd=den.copy()
    refsecs=coarsesecs=regsecs=0.;start=time.perf_counter()
    for i in range(60):
        mean=np.random.default_rng(20261006+i).uniform(-.02,.02)
        u=continuous_random(n,1000+i,mean=mean)
        for j in range(16):
            t=time.perf_counter();v=k.fft(u)
            for sub in range(4):v=k.etd(v,DT/4)
            target=k.ifft(v);refsecs+=time.perf_counter()-t
            t=time.perf_counter();coarse=k.semi(u,DT,6);coarsesecs+=time.perf_counter()-t
            t=time.perf_counter();c=np.fft.fft2(coarse);rr=np.fft.fft2(target-coarse);uh=np.fft.fft2(u);yh=np.fft.fft2(target)
            num+=rr*np.conj(c);den+=abs(c)**2;dn+=yh*np.conj(uh);dd+=abs(uh)**2
            regsecs+=time.perf_counter()-t;u=target
        if (i+1)%10==0:print(json.dumps(dict(stage='galerkin_fit',n=n,trajectories=i+1)),flush=True)
    r=num/(den+1e-12);r[0,0]=0;r[k.k2>28**2]=0
    a=np.divide(dn,dd,out=np.zeros_like(dn),where=dd>0);a[0,0]=1;a*=k.mask
    np.savez_compressed(path,R=r,A_pidmd=a,numerator=num,denominator=den,training_trajectories=60,states_per_trajectory=16)
    (DATA/f'galerkin_fit_N{n}.json').write_text(json.dumps(dict(n=n,reference_seconds=refsecs,coarse_seconds=coarsesecs,regression_seconds=regsecs,
        total_seconds=time.perf_counter()-start,concurrent_campaign_jobs_possible=True,timing_scope='campaign wall time, not an isolated timing benchmark',kind='2N-padded Galerkin with zero Nyquist planes',teacher='ETDRK4 h=2.5e-5',
        training_seeds='continuous_random indices 1000..1059; mean RNG 20261006+i',cutoff=28,regularizer=1e-12,
        hermitian_error=hermitian_error(r)),indent=2))

def evaluate(n,case,steps):
    path=DATA/f'galerkin_rollout_N{n}_{case}_{steps}.npz';csv=path.with_suffix('.csv')
    if path.exists() and csv.exists():return
    k=Kernel(n,kind='galerkin');r=np.load(DATA/f'galerkin_filter_N{n}.npz')['R'];frozen=resize_hat(np.load(DATA/'legacy_filter.npz')['R'],n)
    u=deterministic(n) if case=='rectangles' else continuous_random(n,int(case[6:]))
    u=k.ifft(k.fft(u)*k.mask)
    states={s:u.copy() for s in ['C6','C12','frozen','refitted','ETD']};v=k.fft(u)
    saved={'initial':u};rows=[];bt={s:0 for s in ['frozen','refitted']};fall=bt.copy();maxmass={s:0. for s in states};viol={s:0 for s in states};energies={s:k.energy(u) for s in states}
    for j in range(1,steps+1):
        for q in [6,12]:states[f'C{q}']=k.semi(states[f'C{q}'],DT,q)
        for name,rr in [('frozen',frozen),('refitted',r)]:
            states[name],log=k.surrogate(states[name],rr);bt[name]+=log['code']>0;fall[name]+=log['code']==35
        for sub in range(4):v=k.etd(v,DT/4)
        states['ETD']=k.ifft(v)
        for name,w in states.items():
            en=k.energy(w);viol[name]+=en>energies[name]+1e-12;energies[name]=en;maxmass[name]=max(maxmass[name],abs(float(w.mean()-u.mean())))
        if j in [35,350,700]:
            for name,w in states.items():
                rows.append(dict(n=n,case=case,steps=j,method=name,l2_same_grid=rel(w,states['ETD']),
                    structure_same_grid=rel(structure(k,w),structure(k,states['ETD'])),energy=energies[name],energy_violations=viol[name],max_mass_error=maxmass[name],
                    backtracking=bt.get(name,0),fallback=fall.get(name,0)))
                saved[f'{name}_{j}']=w.copy()
    np.savez_compressed(path,**saved);pd.DataFrame(rows).to_csv(csv,index=False)
    print(json.dumps(dict(stage='galerkin_evaluation',n=n,case=case,steps=steps)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,required=True);a=p.parse_args()
    fit(a.n)
    for i in range(10):evaluate(a.n,f'random{i}',700 if i<3 else 35)
    evaluate(a.n,'rectangles',700)
