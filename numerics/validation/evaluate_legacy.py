"""Matched legacy-grid rollouts, independent temporal reference, full fields."""
import argparse,json,time
import numpy as np
import pandas as pd
from kernel import *

def one(i,steps):
    path=DATA/f'legacy_rollout_{i:03d}_{steps}.npz'
    csv=path.with_suffix('.csv')
    if path.exists() and csv.exists():return
    fit=np.load(DATA/'legacy_filter.npz');r=fit['R'];a=fit['A_pidmd'];k=Kernel()
    u=legacy_ic(i);names=['C1','C6','C12','C100','residual','unconstrained','piDMD','ETD','ETD_half']
    fields={s:u.copy() for s in names};v={s:k.fft(u) for s in ['ETD','ETD_half']}
    energy={s:k.energy(u) for s in names};mass={s:0. for s in names};viol={s:0 for s in names}
    logs=[];snap={'initial':u};rows=[];history=[];bt=fall=0;maximag=0.;maxpost=0
    t0=time.perf_counter()
    for step in range(1,steps+1):
        for q in [1,6,12,100]:fields[f'C{q}']=k.semi(fields[f'C{q}'],DT,q)
        fields['residual'],log=k.surrogate(fields['residual'],r)
        fields['unconstrained'],_=k.surrogate(fields['unconstrained'],r,safe=False)
        fields['piDMD']=np.fft.ifft2(a*np.fft.fft2(fields['piDMD'])).real
        for name,sub in [('ETD',2),('ETD_half',4)]:
            for j in range(sub):v[name]=k.etd(v[name],DT/sub)
            fields[name]=k.ifft(v[name])
        bt+=int(log['code']>0);fall+=int(log['code']==35);maxpost+=log['post_correction_acceptance_violation']
        maximag=max(maximag,log['max_imag'])
        for name in names:
            en=k.energy(fields[name]);viol[name]+=int(en>energy[name]+1e-12);energy[name]=en
            mass[name]=max(mass[name],abs(float(fields[name].mean()-u.mean())))
        if i==0 or log['code']:
            logs.append(dict(test_id=i,step=step,**log))
        if i==0:
            history.append(dict(step=step,time=step*DT,**{f'error_{s}':rel(fields[s],fields['ETD_half']) for s in names},
                **{f'energy_{s}':energy[s] for s in names if s!='piDMD'}))
        if step in [35,350,700]:
            for name in names:
                f=fields[name];ref=fields['ETD_half'];snap[f'{name}_{step}']=f.copy()
                rows.append(dict(test_id=i,steps=step,method=name,
                    l2_vs_C100=rel(f,fields['C100']),structure_vs_C100=rel(structure(k,f),structure(k,fields['C100'])),
                    l2_vs_ETD=rel(f,ref),structure_vs_ETD=rel(structure(k,f),structure(k,ref)),
                    max_mass_error=mass[name],energy_violations=viol[name],backtracking_events=bt if name=='residual' else 0,
                    fallback_events=fall if name=='residual' else 0,max_imaginary_correction=maximag if name=='residual' else 0.,
                    postmean_acceptance_violations=maxpost if name=='residual' else 0,
                    safe_unsafe_field_difference=rel(fields['residual'],fields['unconstrained']) if name=='residual' else 0.))
    np.savez_compressed(path,**snap)
    pd.DataFrame(rows).to_csv(csv,index=False)
    if logs:pd.DataFrame(logs).to_csv(path.with_name(path.stem+'_safeguard.csv'),index=False)
    if history:pd.DataFrame(history).to_csv(path.with_name(path.stem+'_history.csv'),index=False)
    print(json.dumps(dict(stage='legacy_evaluation',test_id=i,steps=steps,seconds=time.perf_counter()-t0,
        residual_l2=rows[4]['l2_vs_C100'],backtracking=bt)),flush=True)

def summarize():
    paths=[f for f in DATA.glob('legacy_rollout_*.csv') if not any(s in f.stem for s in ['safeguard','history'])]
    if not paths:return
    allrows=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    # A 700-step run already includes the short and middle checkpoints.
    allrows=allrows.drop_duplicates(['test_id','steps','method'],keep='last')
    allrows.to_csv(DATA/'legacy_all_metrics.csv',index=False)
    summary=allrows.groupby(['steps','method']).agg(n=('test_id','size'),l2_mean=('l2_vs_ETD','mean'),l2_sd=('l2_vs_ETD','std'),
        legacy_l2_mean=('l2_vs_C100','mean'),structure_mean=('structure_vs_ETD','mean'),structure_sd=('structure_vs_ETD','std'),
        mass_max=('max_mass_error','max'),energy_violations=('energy_violations','sum'),backtracking=('backtracking_events','sum'),fallback=('fallback_events','sum'))
    summary.to_csv(DATA/'legacy_summary.csv');print(summary.to_string(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',default='short');a=p.parse_args()
    if a.stage=='short':
        for i in range(100):one(i,35)
    elif a.stage=='long':
        for i in range(10):one(i,700)
    summarize()
