"""Regression check for conjugate-symmetry preservation over a long rollout."""
import json,time
import numpy as np
from kernel import *
from kernel_fast import Kernel as Fast
rows=[]
for name,n,h,cls in [('fast128',128,5e-5,Fast),('fast128_half',128,2.5e-5,Fast),('full128',128,5e-5,Kernel)]:
    k=cls(n,kind='galerkin');v=k.fft(deterministic(n))*k.mask
    start=time.perf_counter();energies=[k.energy(k.ifft(v))];maximag=0.;viol=0
    for j in range(round(.07/h)):
        v=k.etd(v,h)
        if (j+1)%100==0:
            en=k.energy(k.ifft(v));viol+=en>energies[-1]+1e-12;energies.append(en)
            maximag=max(maximag,float(abs(np.fft.ifft2(v,norm='forward').imag).max()))
    u=k.ifft(v);np.savez_compressed(DATA/f'verified_{name}.npz',field=u)
    rows.append(dict(name=name,energy=energies[-1],energy_increases=viol,max_imag=maximag,max_field=float(abs(u).max()),seconds=time.perf_counter()-start))
    print(rows[-1],flush=True)
a=np.load(DATA/'verified_fast128.npz')['field'];b=np.load(DATA/'verified_fast128_half.npz')['field'];c=np.load(DATA/'verified_full128.npz')['field']
result=dict(runs=rows,time_refinement=rel(a,b),complex_real_agreement=rel(a,c))
assert rows[0]['energy_increases']==0 and rows[1]['energy_increases']==0
assert result['time_refinement']<1e-6 and result['complex_real_agreement']<1e-10
(DATA/'long_reference_verification.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
