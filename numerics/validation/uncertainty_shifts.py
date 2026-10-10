"""Whole-trajectory fit bootstrap and prespecified sensitivity/shift ensembles."""
import json,time
import numpy as np
import pandas as pd
from kernel import *

k=Kernel();fitfile=np.load(DATA/'legacy_filter.npz');r=fitfile['R'];legacy_mask=fitfile['mask']
def rollout(k,u,r,steps=35,safe=True):
    bt=fall=0
    for _ in range(steps):
        u,log=k.surrogate(u,r,safe=safe);bt+=int(log['code']>0);fall+=int(log['code']==35)
    return u,bt,fall

def bootstrap():
    stats=np.load(DATA/'legacy_fit_statistics.npz');num=stats['numer'];den=stats['denom'];power=stats['targetpower']
    totaln=num.sum(0);totald=den.sum(0);p=power.sum(0)
    sse=p-2*np.real(np.conj(r)*totaln)+abs(r)**2*totald
    retained=(k.k2>0)&legacy_mask
    diagnostic=dict(retained_mode_count=int(retained.sum()),training_residual_relative_squared_loss=float(sse[retained].sum()/p[retained].sum()),
        denominator_min=float(totald[retained].min()),denominator_max=float(totald[retained].max()),
        regularizer=1e-12,training_units=600,states_per_unit=16,
        bootstrap_units='entire trajectories, with replacement; 600 draws per fit',bootstrap_replicates=200,bootstrap_seed=20261005,
        evaluation='first ten documented test trajectories, fixed during bootstrap; conditional on those ten states',
        intervals='percentile intervals from 200 bootstrap fits; Monte Carlo quantile precision limited')
    (DATA/'regression_diagnostics.json').write_text(json.dumps(diagnostic,indent=2))
    u=np.stack([legacy_ic(i) for i in range(10)])
    # Batch numerical updates are independent on the last two axes. Mean and
    # safeguard intentionally absent here: this bootstrap estimates filter
    # uncertainty for the unprojected map; nominal safeguard is inactive.
    refs=np.stack([np.load(DATA/f'legacy_rollout_{i:03d}_35.npz')['ETD_half_35'] for i in range(10)])
    rng=np.random.default_rng(20261005);rows=[];filters=[]
    for b in range(200):
        count=np.bincount(rng.integers(0,600,600),minlength=600)
        rr=(num*count[:,None,None]).sum(0)/((den*count[:,None,None]).sum(0)+1e-12)
        rr[0,0]=0;rr[~legacy_mask]=0
        out=u.copy()
        for step in range(35):
            c=k.semi(out,DT,6);out=c+np.fft.ifft2(rr*np.fft.fft2(c)).real
        errors=np.sqrt(np.sum((out-refs)**2,axis=(-2,-1))/np.sum(refs**2,axis=(-2,-1)))
        rows.append(dict(replicate=b,mean_field_error=float(errors.mean()),sd_field_error=float(errors.std(ddof=1)),filter_relative_difference=rel(rr,r)))
        filters.append(rr)
        if (b+1)%25==0:print(json.dumps(dict(stage='training_bootstrap',replicates=b+1)),flush=True)
    pd.DataFrame(rows).to_csv(DATA/'training_bootstrap.csv',index=False)
    np.savez_compressed(DATA/'bootstrap_filters.npz',R=np.stack(filters))
    diagnostic['conditional_mean_error_percentile95']=np.percentile([x['mean_field_error'] for x in rows],[2.5,97.5]).tolist()
    (DATA/'regression_diagnostics.json').write_text(json.dumps(diagnostic,indent=2))

def sensitivities():
    stats=np.load(DATA/'legacy_fit_statistics.npz');num=stats['numer'].sum(0);den=stats['denom'].sum(0)
    rows=[]
    for i in range(20):
        u=legacy_ic(i,mean=float(np.random.default_rng(SEED+610000+i).uniform(-.02,.02)),seed=SEED+600000+i)
        ref=run_etd(k,u,[.0035],2.5e-5)[.0035]
        for cutoff in [16,24,28,31]:
            for lam in [1e-12,1e-6,1.,1000.]:
                wave=np.fft.fftfreq(64,d=2*np.pi/64)*2*np.pi
                radius=np.sqrt(wave[:,None]**2+wave[None,:]**2)
                rr=num/(den+lam);rr[0,0]=0;rr[radius>cutoff]=0
                out,bt,fall=rollout(k,u,rr)
                rows.append(dict(validation_id=i,cutoff=cutoff,regularizer=lam,l2=rel(out,ref),structure_error=rel(structure(k,out),structure(k,ref)),backtracking=bt,fallback=fall))
    pd.DataFrame(rows).to_csv(DATA/'hyperparameter_sensitivity.csv',index=False)
    print('Sensitivity complete',flush=True)

def shifts():
    cases=[('nominal',.02,1.,.08,10.,0.,False),('eps_low',.01,1.,.08,10.,0.,False),('eps_high',.03,1.,.08,10.,0.,False),
        ('mob_low_fixedT',.02,.5,.08,10.,0.,False),('mob_high_fixedT',.02,2.,.08,10.,0.,False),
        ('mob_low_matchedMT',.02,.5,.08,10.,0.,True),('mob_high_matchedMT',.02,2.,.08,10.,0.,True),
        ('amplitude_low',.02,1.,.04,10.,0.,False),('amplitude_high',.02,1.,.16,10.,0.,False),('amplitude_stress',.02,1.,.3,10.,0.,False),
        ('smoothing_low',.02,1.,.08,5.,0.,False),('smoothing_high',.02,1.,.08,15.,0.,False),
        ('mean_positive',.02,1.,.08,10.,.2,False),('mean_negative',.02,1.,.08,10.,-.2,False),('joint',.015,1.5,.16,15.,.2,False)]
    rows=[]
    for name,eps,mob,amp,sigma,mean,matched in cases:
        kk=Kernel(eps=eps,mob=mob)
        # Matched MT and matched M*macro_dt give the exact time-rescaling
        # control. This is a control identity, not independent generalization.
        T=.0035/mob if matched else .0035
        last=0.;steps=[T/35]*35
        for i in range(10):
            u=legacy_ic(i,mean=mean,amplitude=amp,sigma=sigma,seed=SEED+700000+i)
            ref=run_etd(kk,u,[T],min(2.5e-5,2.5e-5/mob))[T]
            ref2=run_etd(kk,u,[T],min(1.25e-5,1.25e-5/mob))[T]
            states={s:u.copy() for s in ['C6','C12','residual']};bt=fall=0
            for h in steps:
                for q in [6,12]:states[f'C{q}']=kk.semi(states[f'C{q}'],h,q)
                states['residual'],log=kk.surrogate(states['residual'],r,h);bt+=int(log['code']>0);fall+=int(log['code']==35)
            for name2,out in states.items():
                rows.append(dict(case=name,test_id=i,method=name2,eps=eps,mobility=mob,amplitude=amp,sigma=sigma,mean=mean,T=T,
                    effective_time=mob*T,last_partial_step=last if last>1e-12 else 0.,l2=rel(out,ref2),structure_error=rel(structure(kk,out),structure(kk,ref2)),
                    reference_temporal_difference=rel(ref,ref2),backtracking=bt if name2=='residual' else 0,fallback=fall if name2=='residual' else 0))
        print(json.dumps(dict(stage='shift_ensemble',case=name)),flush=True)
    pd.DataFrame(rows).to_csv(DATA/'shift_ensemble.csv',index=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--stage',default='all');a=p.parse_args()
    for name,fn in [('bootstrap',bootstrap),('sensitivity',sensitivities),('shifts',shifts)]:
        if a.stage in [name,'all']:fn()
