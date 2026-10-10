"""Real-FFT optimization of the same padded Galerkin equations.

Validated against the full-complex implementation in kernel.py. Only the
Galerkin cubic and energy quadrature are optimized; legacy methods unchanged.
"""
from kernel import *
from kernel import Kernel as FullKernel

class Kernel(FullKernel):
    def semi(self,u,dt,q=1,original_power=False):
        if self.kind=='legacy':return super().semi(u,dt,q,original_power)
        h=dt/q;v=self.fft(u)*self.mask
        a=1+h*self.mob*2*self.k2;b=h*self.mob*self.k2
        den=1+h*self.mob*(self.eps**2*self.k2**2+2*self.k2)
        for _ in range(q):v=(a*v-b*(self.cubic(v)-v))/den
        return self.ifft(v*self.mask)
    def padded_physical(self,v):
        n=self.n;m=n//2
        p=np.zeros((2*n,n+1),complex)
        p[:m,:m]=v[:m,:m]
        p[-m+1:,:m]=v[-m+1:,:m]
        return np.fft.irfft2(p,s=(2*n,2*n),norm='forward')
    def cubic(self,v):
        if self.kind=='legacy':return super().cubic(v)
        n=self.n;m=n//2
        u=self.padded_physical(v*self.mask)
        z=np.fft.rfft2(u*u*u,norm='forward')
        out=np.zeros((n,n),complex)
        out[:m,:m]=z[:m,:m];out[-m+1:,:m]=z[-m+1:,:m]
        idx=(-np.arange(n))%n
        out[:, -np.arange(1,m)]=np.conj(out[idx[:,None],np.arange(1,m)[None,:]])
        return out
    def energy(self,u):
        if self.kind=='legacy':return super().energy(u)
        v=self.fft(u)*self.mask;w=self.padded_physical(v)
        return float(.5*self.eps**2*np.sum(self.k2*abs(v)**2)+.25*np.mean((w*w-1)**2))

if __name__=='__main__':
    import time,json
    rows=[]
    for n in [16,64,128,256,512]:
        f=FullKernel(n,kind='galerkin');g=Kernel(n,kind='galerkin')
        u=np.random.default_rng(8).normal(size=(n,n))*.2;v=f.fft(u)*f.mask;u=f.ifft(v)
        t=time.perf_counter();a=f.cubic(v);tf=time.perf_counter()-t
        t=time.perf_counter();b=g.cubic(v);tg=time.perf_counter()-t
        err=float(abs(a-b).max());en=abs(f.energy(u)-g.energy(u))
        assert err<1e-13 and en<1e-12
        semierr=rel(f.semi(u,1e-4,6),g.semi(u,1e-4,6));assert semierr<1e-12
        rows.append(dict(n=n,cubic_max_difference=err,energy_difference=en,semi_relative_difference=semierr,full_seconds=tf,real_seconds=tg))
    f=FullKernel(64,kind='galerkin');g=Kernel(64,kind='galerkin');u=continuous_random(64)
    a=run_etd(f,u,[.0035],1e-4)[.0035];b=run_etd(g,u,[.0035],1e-4)[.0035]
    assert rel(a,b)<1e-12
    result=dict(cubic_tests=rows,short_rollout_relative_difference=rel(a,b))
    (DATA/'real_fft_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
