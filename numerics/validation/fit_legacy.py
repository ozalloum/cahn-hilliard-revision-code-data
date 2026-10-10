"""Regenerate original 600-trajectory filter and shared piDMD statistics."""
import time,json,platform,hashlib,sys,os
from pathlib import Path
import numpy as np
from kernel import Kernel,SEED,DT,DATA,legacy_ic,hermitian_error,rel,structure

k=Kernel();ntraj=600
path=DATA/'legacy_fit_statistics.npz'
numer=np.zeros((ntraj,64,64),complex);denom=np.zeros((ntraj,64,64))
dmdnum=numer.copy();dmdden=denom.copy();targetpower=denom.copy()
elapsed=dict(reference=0.,coarse=0.,regression=0.,total=0.)
done=0
if path.exists():
    z=np.load(path);done=int(z['done'])
    for name in ('numer','denom','dmdnum','dmdden','targetpower'):globals()[name][:]=z[name]
    elapsed=json.loads(str(z['elapsed']))
t0=time.perf_counter()
for i in range(done,ntraj):
    phi=legacy_ic(i,train=True)
    for step in range(16):
        t=time.perf_counter();target=k.semi(phi,DT,100,original_power=True);elapsed['reference']+=time.perf_counter()-t
        t=time.perf_counter();coarse=k.semi(phi,DT,6,original_power=True);elapsed['coarse']+=time.perf_counter()-t
        t=time.perf_counter()
        c=np.fft.fft2(coarse);r=np.fft.fft2(target-coarse);u=np.fft.fft2(phi);y=np.fft.fft2(target)
        numer[i]+=r*np.conj(c);denom[i]+=abs(c)**2
        dmdnum[i]+=y*np.conj(u);dmdden[i]+=abs(u)**2;targetpower[i]+=abs(r)**2
        elapsed['regression']+=time.perf_counter()-t;phi=target
    if (i+1)%25==0 or i==ntraj-1:
        elapsed['total']+=time.perf_counter()-t0;t0=time.perf_counter()
        np.savez(path,numer=numer,denom=denom,dmdnum=dmdnum,dmdden=dmdden,targetpower=targetpower,done=i+1,elapsed=json.dumps(elapsed))
        print(json.dumps(dict(stage='legacy_fit',trajectories=i+1,times=elapsed)),flush=True)
# Reproduce the actual floating-point cutoff in the submitted code. Its
# fftfreq(d=2*pi/N)*2*pi places the four axial |k|=28 modes just above 28.
wave=np.fft.fftfreq(64,d=2*np.pi/64)*2*np.pi
kx,ky=np.meshgrid(wave,wave,indexing='ij');mask=np.sqrt(kx*kx+ky*ky)<=28.
r=numer.sum(axis=0)/(denom.sum(axis=0)+1e-12);r[0,0]=0;r[~mask]=0
a=np.divide(dmdnum.sum(axis=0),dmdden.sum(axis=0),out=np.zeros((64,64),complex),where=dmdden.sum(axis=0)>0);a[0,0]=1
np.savez(DATA/'legacy_filter.npz',R=r,A_pidmd=a,mask=mask,fit_trajectories=600,states_per_trajectory=16,regularizer=1e-12,cutoff=28)
meta=dict(trajectories=600,pairs=9600,times=elapsed,filter_hermitian_max=hermitian_error(r),pidmd_hermitian_max=hermitian_error(a),
          retained_nonzero_modes=int(mask.sum()-1),axial_boundary_modes_excluded_by_original_floating_point_cutoff=True,
          numerator_denominator_trajectory_sufficient_statistics_saved=True,python=sys.version,numpy=np.__version__,platform=platform.platform(),logical_processors=os.cpu_count(),
          source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),kernel_hash=hashlib.sha256((Path(__file__).parent/'kernel.py').read_bytes()).hexdigest())
(DATA/'legacy_fit_metadata.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta),flush=True)
