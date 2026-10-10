"""Derive report tables and cross-check all claimed new numbers from raw files."""
import json,hashlib
import numpy as np
import pandas as pd
from kernel import *
summary={}
def write(name,frame):
    frame.to_csv(DATA/name,index=False)
    return frame.to_dict(orient='records')

if (DATA/'legacy_all_metrics.csv').exists():
    f=pd.read_csv(DATA/'legacy_all_metrics.csv')
    archive=pd.read_csv(HERE.parent/'submitted'/'rollout_metrics.csv',comment='#')
    short=f[(f.steps==35)&(f.method=='residual')].sort_values('test_id')
    if len(short)==100:
        summary['archive_reproduction']=dict(trajectories=100,
            max_absolute_field_metric_difference=float(abs(short.l2_vs_C100.to_numpy()-archive.final_l2_learned.to_numpy()).max()),
            max_absolute_structure_metric_difference=float(abs(short.structure_vs_C100.to_numpy()-archive.structure_error_learned.to_numpy()).max()),
            regenerated_mean_field_error=float(short.l2_vs_C100.mean()))
    rng=np.random.default_rng(20261004);rows=[]
    for steps,ff in f.groupby('steps'):
        pivot=ff.pivot(index='test_id',columns='method',values='l2_vs_ETD')
        for baseline in ['C1','C6','C12','piDMD']:
            d=(pivot[baseline]-pivot['residual']).to_numpy();n=len(d)
            means=d[rng.integers(0,n,(20000,n))].mean(axis=1)
            ci=np.percentile(means,[2.5,97.5])
            rows.append(dict(steps=int(steps),baseline=baseline,n=n,paired_mean_difference=float(d.mean()),CI95_lower=float(ci[0]),CI95_upper=float(ci[1])))
    summary['paired_test_intervals']=write('paired_test_intervals.csv',pd.DataFrame(rows))
    summary['legacy_diagnostics']=dict(postmean_acceptance_violations=int(f.postmean_acceptance_violations.max()),
        max_imaginary_correction=float(f.max_imaginary_correction.max()),max_safe_unsafe_field_difference=float(f.safe_unsafe_field_difference.max()),
        max_etd_refinement_difference=float(f[f.method=='ETD'].l2_vs_ETD.max()))

if (DATA/'stress_sweep.csv').exists():
    f=pd.read_csv(DATA/'stress_sweep.csv')
    finite=f[f.candidate_finite & np.isfinite(f.energy_candidate)]
    summary['stress_sweep']=dict(total_cases=len(f),finite_energy_candidates=len(finite),active_finite_cases=int((finite.code>0).sum()),
        fallback_finite_cases=int((finite.code==35).sum()),nonfinite_field_candidates=int((~f.candidate_finite).sum()),
        nonfinite_energy_candidates=int((~np.isfinite(f.energy_candidate)).sum()),
        finite_fields_with_nonfinite_energy=int((f.candidate_finite & ~np.isfinite(f.energy_candidate)).sum()),
        postmean_acceptance_violations=int(f.post_correction_acceptance_violation.sum()))

for name in ['shift_ensemble','hyperparameter_sensitivity']:
    p=DATA/f'{name}.csv'
    if p.exists():
        f=pd.read_csv(p);groups=['case','method'] if name=='shift_ensemble' else ['cutoff','regularizer']
        g=f.groupby(groups).agg(n=('l2','size'),l2_mean=('l2','mean'),l2_sd=('l2','std'),structure_mean=('structure_error','mean'),
            backtracking=('backtracking','sum'),fallback=('fallback','sum')).reset_index()
        summary[name]=write(f'{name}_summary.csv',g)
        if name=='shift_ensemble':summary['shift_reference_temporal_max']=float(f.reference_temporal_difference.max())

paths=list(DATA.glob('galerkin_rollout_*.csv'))
if paths:
    f=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    f.to_csv(DATA/'galerkin_all_metrics.csv',index=False)
    f['ensemble']=np.where(f.case=='rectangles','rectangles','random')
    g=f.groupby(['n','steps','ensemble','method']).agg(cases=('case','size'),l2_mean=('l2_same_grid','mean'),l2_sd=('l2_same_grid','std'),
        structure_mean=('structure_same_grid','mean'),backtracking=('backtracking','sum'),fallback=('fallback','sum'),
        mass_max=('max_mass_error','max'),energy_violations=('energy_violations','sum')).reset_index()
    summary['galerkin']=write('galerkin_summary.csv',g)
    cross=[]
    for p in DATA.glob('galerkin_rollout_*.npz'):
        import re
        match=re.match(r'galerkin_rollout_N(\d+)_(.+)_(\d+)',p.stem);n=int(match[1]);case=match[2];z=np.load(p)
        refs=list(DATA.glob(f'convergence_{case}_galerkin_N512_h*.npz'))
        if not refs:continue
        ref=min([np.load(q) for q in refs],key=lambda a:float(a['h']))
        for step,j in [(35,1),(350,2),(700,3)]:
            for method in ['C6','C12','frozen','refitted','ETD']:
                key=f'{method}_{step}'
                if key not in z:continue
                field=Kernel(512).ifft(resize_hat(Kernel(n).fft(z[key]),512))
                cross.append(dict(n=n,case=case,steps=step,method=method,l2_vs_N512=rel(field,ref[f'u{j}'])))
    if cross:summary['galerkin_crossgrid']=write('galerkin_vs_N512.csv',pd.DataFrame(cross))

for name in ['legacy_fit_metadata','regression_diagnostics','stress_rollout_summary','environment']:
    p=DATA/f'{name}.json'
    if p.exists():summary[name]=json.loads(p.read_text())
for name in ['legacy_summary','cost_summary','reference_substeps']:
    p=DATA/f'{name}.csv'
    if p.exists():summary[name]=pd.read_csv(p).to_dict(orient='records')
def json_finite(value):
    if isinstance(value,dict):return {key:json_finite(item) for key,item in value.items()}
    if isinstance(value,list):return [json_finite(item) for item in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value
(DATA/'evidence_summary.json').write_text(json.dumps(json_finite(summary),indent=2,allow_nan=False))
print(json.dumps({key:value for key,value in summary.items() if key in ['archive_reproduction','legacy_diagnostics','stress_sweep','shift_reference_temporal_max']},indent=2))
print('Saved evidence_summary.json with',len(summary),'categories',flush=True)
