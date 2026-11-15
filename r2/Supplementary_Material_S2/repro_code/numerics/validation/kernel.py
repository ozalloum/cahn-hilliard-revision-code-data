"""Versioned Cahn-Hilliard validation kernels (float64/complex128, NumPy FFT).

legacy: original unpadded collocation, full Laplacian, real-derivative energy.
galerkin: zero Nyquist rows/columns; cubic evaluated on a 2N padded grid.
ETDRK4: Cox-Matthews stages with Kassam-Trefethen contour coefficients.
"""
from pathlib import Path
from functools import lru_cache
import os
import numpy as np

SEED = 20260618
DT = 1e-4
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get('CH_VALIDATION_RESULTS',str(HERE/'results'))).resolve()
DATA.mkdir(parents=True,exist_ok=True)

def rel(a,b):
    return float(np.sqrt(np.sum(np.abs(a-b)**2)/max(np.sum(np.abs(b)**2),1e-300)))

def resize_hat(v,n):
    """Fourier coefficient injection/restriction; normalized-forward convention.
    Inputs for cross-grid comparisons have zero Nyquist planes; no splitting.
    """
    old=v.shape[-1]
    if old==n: return v.copy()
    a=np.fft.fftshift(v,axes=(-2,-1))
    if n>old:
        out=np.zeros(v.shape[:-2]+(n,n),complex)
        start=n//2-old//2
        out[...,start:start+old,start:start+old]=a
    else:
        start=old//2-n//2
        out=a[...,start:start+n,start:start+n].copy()
    out=np.fft.ifftshift(out,axes=(-2,-1))
    if n<old:
        out[...,n//2,:]=0;out[...,:,n//2]=0
    return out

def hermitian_error(v):
    n=v.shape[-1];idx=(-np.arange(n))%n
    return float(np.max(np.abs(v-np.conj(v[...,idx[:,None],idx[None,:]]))))

class Kernel:
    def __init__(self,n=64,eps=.02,mob=1.,kind='legacy'):
        self.n=n;self.eps=eps;self.mob=mob;self.kind=kind
        k=np.fft.fftfreq(n)*n
        self.kx,self.ky=np.meshgrid(k,k,indexing='ij')
        self.k2=self.kx**2+self.ky**2
        self.mask=np.ones((n,n),bool)
        if kind=='galerkin': self.mask[n//2,:]=False;self.mask[:,n//2]=False
        self.lin=mob*(self.k2-eps**2*self.k2**2)
    def fft(self,u):return np.fft.fft2(u,norm='forward')
    def ifft(self,v):return np.fft.ifft2(v,norm='forward').real
    def cubic(self,v):
        if self.kind=='legacy':
            u=self.ifft(v)
            return self.fft(u*u*u)
        w=self.ifft(resize_hat(v*self.mask,2*self.n))
        return resize_hat(self.fft(w*w*w),self.n)*self.mask
    def nl(self,v):return -self.mob*self.k2*self.cubic(v)
    def energy(self,u):
        v=self.fft(u)
        if self.kind=='legacy':
            dx=self.ifft(1j*self.kx*v);dy=self.ifft(1j*self.ky*v)
            return float(np.mean(.5*self.eps**2*(dx*dx+dy*dy)+.25*(u*u-1)**2))
        # Padded quadrature integrates the degree-four potential without alias.
        w=self.ifft(resize_hat(v*self.mask,2*self.n))
        return float(.5*self.eps**2*np.sum(self.k2*np.abs(v*self.mask)**2)+.25*np.mean((w*w-1)**2))
    def semi(self,u,dt,q=1,original_power=False):
        out=u.copy();h=dt/q
        stab=h*self.mob*2*self.k2
        a=1+stab
        b=h*self.mob*self.k2
        den=1+h*self.mob*(self.eps**2*self.k2**2+2*self.k2)
        for _ in range(q):
            if self.kind=='legacy':
                # Preserve original unnormalized FFT for archive regeneration.
                v=np.fft.fft2(out)
                non=np.fft.fft2((out**3 if original_power else out*out*out)-out)
                v=(v-b*non+stab*v)/den
                out=np.fft.ifft2(v).real
            else:
                v=self.fft(out)*self.mask
                v=(a*v-b*(self.cubic(v)-v))/den
                out=self.ifft(v*self.mask)
        return out
    @lru_cache(maxsize=24)
    def coefficients(self,h):
        z=h*self.lin
        e=np.exp(z);e2=np.exp(z/2)
        q=np.zeros_like(z);f1=q.copy();f2=q.copy();f3=q.copy()
        # Sequential contour accumulation bounds peak memory at large N.
        for j in range(32):
            w=z+np.exp(1j*np.pi*(j+.5)/32)
            ew=np.exp(w)
            q+=np.real(np.expm1(w/2)/w)/32
            f1+=np.real((-4-w+ew*(4-3*w+w*w))/(w*w*w))/32
            f2+=np.real((2+w+ew*(-2+w))/(w*w*w))/32
            f3+=np.real((-4-3*w-w*w+ew*(4-w))/(w*w*w))/32
        return e,e2,h*q,h*f1,h*f2,h*f3
    def etd(self,v,h):
        e,e2,q,f1,f2,f3=self.coefficients(float(h))
        nv=self.nl(v);a=e2*v+q*nv
        na=self.nl(a);b=e2*v+q*na
        nb=self.nl(b);c=e2*a+q*(2*nb-nv)
        out=(e*v+f1*nv+2*f2*(na+nb)+f3*self.nl(c))*self.mask
        # The state represents a real field. Roundoff in conjugate partners
        # must not evolve as an unphysical imaginary solution of the unstable
        # linear operator (the cubic is evaluated on real fields).
        idx=(-np.arange(self.n))%self.n
        return .5*(out+np.conj(out[...,idx[:,None],idx[None,:]]))
    def safeguard(self,old,candidate):
        mean=float(old.mean());c=candidate+(mean-float(candidate.mean()))
        e0=self.energy(old);ec=self.energy(c);code=0;accepted=c
        if not np.isfinite(ec) or ec>e0+1e-14:
            code=35;accepted=old.copy();d=c-old
            for j in range(1,35):
                w=old+2.**(-j)*d;w+=mean-float(w.mean())
                ew=self.energy(w)
                if np.isfinite(ew) and ew<=e0+1e-14:
                    code=j;accepted=w;break
        accepted+=mean-float(accepted.mean())
        er=self.energy(accepted)
        return accepted,dict(code=code,alpha=0. if code==35 else 2.**(-code),
            energy_old=e0,energy_candidate=ec,energy_returned=er,
            post_correction_acceptance_violation=bool(er>e0+1e-14),
            mass_error=abs(float(accepted.mean())-mean),
            relative_change=rel(accepted,c))
    def surrogate(self,u,r,dt=DT,safe=True):
        c=self.semi(u,dt,6)
        z=np.fft.ifft2(r*np.fft.fft2(c))
        v=c+z.real
        if not safe:return v,dict(code=0,max_imag=float(abs(z.imag).max()))
        w,log=self.safeguard(u,v);log['max_imag']=float(abs(z.imag).max())
        return w,log

def legacy_ic(i,train=False,mean=None,amplitude=.08,sigma=10.,seed=None):
    k=Kernel()
    if seed is None:seed=SEED+(10000 if train else 20000)+i
    if mean is None:mean=np.random.default_rng(SEED+(0 if train else 200)+i).uniform(-.02,.02)
    noise=np.random.default_rng(seed).normal(size=(64,64))
    z=np.fft.ifft2(np.fft.fft2(noise)*np.exp(-.5*k.k2/sigma**2)).real
    return amplitude*(z-z.mean())/z.std()+mean

def continuous_random(n,i=0,amplitude=.08,mean=0.,sigma=10.):
    # One fixed trigonometric polynomial on every grid: original 64-point
    # random realization with both Nyquist planes removed, then normalized once.
    k=Kernel();u=legacy_ic(i,mean=0.,amplitude=1.,sigma=sigma,seed=SEED+400000+i)
    v=k.fft(u);v[32,:]=0;v[:,32]=0;v[0,0]=0
    v*=amplitude/np.sqrt(np.sum(np.abs(v)**2));v[0,0]=mean
    return Kernel(n).ifft(resize_hat(v,n))

def deterministic(n):
    # Analytic, smooth and periodic two-rectangle geometry; no endpoint jumps.
    x=np.arange(n)*2*np.pi/n;x,y=np.meshgrid(x,x,indexing='ij')
    def window(z,c,a):return .5*(1+np.tanh((np.cos(z-c)-np.cos(a))/.08))
    a=window(x,np.pi/2,.55)*window(y,np.pi,.8)
    b=window(x,3*np.pi/2,.55)*window(y,np.pi,.8)
    return -0.8+1.6*(a+b)

def run_etd(k,u,times,h):
    v=k.fft(u)*k.mask;t=0.;out={0.:k.ifft(v)}
    for target in times:
        count=int(np.ceil((target-t)/h-1e-10));step=(target-t)/count
        for _ in range(count):v=k.etd(v,step)
        if not np.isfinite(v).all():raise FloatingPointError('ETDRK4 nonfinite')
        out[target]=k.ifft(v);t=target
    return out

def adaptive(k,u,T,rtol,method='etd',atol=None,h0=1e-4,max_h=5e-4):
    if atol is None:atol=rtol*1e-3
    t=0.;h=h0;accepted=0;rejected=0;history=[];u=u.copy()
    order=4 if method=='etd' else 1
    while t<T-1e-15:
        h=min(h,T-t,max_h)
        if h<1e-12:raise RuntimeError('adaptive step underflow')
        if method=='etd':
            v=k.fft(u);coarse=k.ifft(k.etd(v,h));fine=k.ifft(k.etd(k.etd(v,h/2),h/2))
        else:
            coarse=k.semi(u,h);fine=k.semi(u,h,2)
        scale=atol+rtol*np.maximum(np.abs(u),np.abs(fine))
        err=float(np.sqrt(np.mean(((fine-coarse)/((2**order-1)*scale))**2)))
        if np.isfinite(err) and err<=1:
            u=fine;t+=h;accepted+=1;history.append((t,h,err))
        else:rejected+=1
        factor=5. if err==0 else float(np.clip(.9*err**(-1/(order+1)),.2,5)) if np.isfinite(err) else .2
        h*=factor
        if accepted+rejected>1000000:raise RuntimeError('excess attempts')
    return u,dict(accepted=accepted,rejected=rejected,min_h=min(a[1] for a in history),max_h=max(a[1] for a in history),rtol=rtol,atol=atol,history=history)

def structure(k,u):
    v=abs(k.fft(u-u.mean()))**2
    if k.kind=='legacy':
        # Match archived bin membership at floating-point radial boundaries.
        wave=np.fft.fftfreq(k.n,d=2*np.pi/k.n)*2*np.pi
        r=np.sqrt(wave[:,None]**2+wave[None,:]**2)
    else:r=np.sqrt(k.k2)
    edges=np.linspace(0,r.max(),33)
    s=np.array([np.mean(v[(r>=a)&(r<b)]) if np.any((r>=a)&(r<b)) else 0. for a,b in zip(edges[:-1],edges[1:])])
    return s/max(s.sum(),1e-300)
