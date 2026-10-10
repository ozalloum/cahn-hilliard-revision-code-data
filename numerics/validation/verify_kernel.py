"""Independent identities and refinement checks for the new numerical kernels."""
import sys,json
import numpy as np
from kernel import *
sys.path.insert(0,str(HERE.parent))
from revision_baselines import env

results={}
k=Kernel();u=legacy_ic(0,train=True)
mean=np.random.default_rng(SEED).uniform(-.02,.02)
original=env['initial_condition'](SEED+10000,mean=mean)
results['legacy_ic_max']=float(abs(u-original).max())
results['legacy_reference_max']=float(abs(k.semi(u,DT,100,True)-env['reference_step'](original,DT,.02,1.)).max())
results['legacy_energy_diff']=k.energy(u)-env['free_energy'](u,.02)
assert results['legacy_reference_max']<1e-13

g=Kernel(16,kind='galerkin');x=2*np.pi*np.arange(16)/16
u=np.cos(5*x)[:,None]*np.ones((1,16));v=g.fft(u)
expected=.75*u
results['dealiased_cubic_projection_max']=float(abs(g.ifft(g.cubic(v))-expected).max())
assert results['dealiased_cubic_projection_max']<1e-13
results['padding_roundtrip_max']=float(abs(resize_hat(resize_hat(v,32),16)-v).max())
assert results['padding_roundtrip_max']<1e-14
# Variational derivative of padded quadrature equals projected cubic - u - eps^2 lap u.
rng=np.random.default_rng(17)
v=g.fft(rng.normal(size=(16,16))*.1)*g.mask
d=g.fft(rng.normal(size=(16,16)))*g.mask
u=g.ifft(v);du=g.ifft(d)
gradient=g.ifft(g.eps**2*g.k2*v+g.cubic(v)-v)
delta=1e-5
fd=(g.energy(u+delta*du)-g.energy(u-delta*du))/(2*delta)
results['energy_gradient_identity_error']=abs(fd-float(np.mean(gradient*du)))
assert results['energy_gradient_identity_error']<1e-9
class Linear(Kernel):
    def nl(self,v):return np.zeros_like(v)
lin=Linear(16);v=lin.fft(u)
results['etd_linear_exact_max']=float(abs(lin.etd(v,.001)-np.exp(.001*lin.lin)*v).max())
assert results['etd_linear_exact_max']<1e-15
# Nonlinear fourth-order refinement: independent h/2 sequence on a small grid.
g=Kernel(16,eps=.1,kind='galerkin');u=.15*np.cos(3*x)[:,None]+.1*np.cos(2*x)[None,:]
hs=[.004,.002,.001,.0005,.00025]
fields=[run_etd(g,u,[.02],h)[.02] for h in hs]
errors=[rel(a,fields[-1]) for a in fields[:-1]]
results['etdrk4_h']=hs;results['etdrk4_errors']=errors
results['etdrk4_observed_orders']=[float(np.log2(a/b)) for a,b in zip(errors[:-1],errors[1:])]
assert min(results['etdrk4_observed_orders'])>3.5
results['etd_mass_error']=abs(float(fields[-1].mean()-u.mean()))
assert results['etd_mass_error']<1e-14
(DATA/'kernel_verification.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
