"""Evaluate matched uncorrected baselines using the submitted solver functions.

The archived reproduction script is parsed, not run as a program. Only numerical
constants and the named numerical functions below are loaded; output generation,
training, packaging and browser-download functions are excluded.
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT/'submitted'/'reproduce_cahn_hilliard.py'
DEST = ROOT/'revision_results'
DEST.mkdir(exist_ok=True)
names = {'initial_condition','free_energy','semi_implicit_step','reference_step',
         'coarse_operator_step','relative_l2','radial_structure_factor','structure_error'}
original = ast.parse(SRC.read_text(encoding='utf-8'))
nodes = []
for n in original.body:
    if isinstance(n,(ast.Assign,ast.AnnAssign)) and n.lineno < 60:
        nodes.append(n)
    elif isinstance(n,ast.FunctionDef) and n.name in names:
        nodes.append(n)
env = {'np':np,'pd':pd}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(SRC),'exec'),env)

def one(test_id):
    start = time.perf_counter()
    seed=env['SEED']; dt=env['MACRO_DT']; eps=env['EPS_TRAIN']; mob=env['MOBILITY_TRAIN']
    mean=np.random.default_rng(seed+200+test_id).uniform(-.02,.02)
    phi0=env['initial_condition'](seed+20000+test_id,mean=mean)
    fields={n:phi0.copy() for n in [1,6,12,100]}
    energy={n:env['free_energy'](phi0,eps) for n in fields}
    viol={n:0 for n in fields}
    mass={n:0.0 for n in fields}
    hist=[]
    for step in range(1,36):
        for n in fields:
            fields[n]=env['coarse_operator_step'](fields[n],dt,eps,mob,half_steps=n)
            en=env['free_energy'](fields[n],eps)
            viol[n]+=int(en-energy[n]>1e-12)
            energy[n]=en
            mass[n]=max(mass[n],abs(float(fields[n].mean())-float(phi0.mean())))
        hist.append(dict(test_id=test_id,step=step,time=step*dt,
                         **{f'error_{n}':env['relative_l2'](fields[n],fields[100]) for n in [1,6,12]}))
    row={'test_id':test_id,'wall_seconds':time.perf_counter()-start}
    for n in [1,6,12]:
        row[f'l2_{n}']=env['relative_l2'](fields[n],fields[100])
        row[f'structure_{n}']=env['structure_error'](fields[n],fields[100])
        row[f'energy_violations_{n}']=viol[n]
        row[f'max_mass_error_{n}']=mass[n]
    if test_id==0:
        np.savez(DEST/'representative_fields.npz',initial=phi0,**{f'final_{n}':a for n,a in fields.items()})
        pd.DataFrame(hist).to_csv(DEST/'representative_baseline_history.csv',index=False)
        sf=env['radial_structure_factor'](fields[100]).rename(columns={'structure_factor':'reference'})
        for n in [1,6,12]: sf[f'coarse_{n}']=env['radial_structure_factor'](fields[n]).structure_factor
        sf.to_csv(DEST/'representative_structure.csv',index=False)
    return row

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--count',type=int,default=100);ap.add_argument('--resume',action='store_true'); args=ap.parse_args()
    rows=[]
    dest=DEST/'matched_baselines.csv'
    if args.resume and dest.exists(): rows=pd.read_csv(dest).to_dict('records')
    completed={int(r['test_id']) for r in rows}
    for i in range(args.count):
        if i in completed: continue
        r=one(i);rows.append(r);pd.DataFrame(rows).to_csv(dest,index=False)
        print(json.dumps(r),flush=True)
    old=pd.read_csv(ROOT/'submitted'/'rollout_metrics.csv',comment='#')
    new=pd.DataFrame(rows).merge(old,on='test_id',validate='one_to_one')
    check=float(np.max(np.abs(new.l2_1-new.final_l2_coarse)))
    assert check<1e-10, f'Archived one-step errors did not reproduce: {check}'
    summary={'n':len(new),'maximum_archived_coarse_error_discrepancy':check,
             'methods':{},'comparison_provenance':'Recomputed uncorrected baselines and 100-substep reference; learned errors from archived CSVs; no new filter fit.'}
    for n in [1,6,12]:
        summary['methods'][str(n)]={k:float(new[f'{k}_{n}'].mean()) for k in ['l2','structure']}
        summary['methods'][str(n)]['l2_std']=float(new[f'l2_{n}'].std(ddof=1)) if len(new)>1 else None
        summary['methods'][str(n)]['structure_std']=float(new[f'structure_{n}'].std(ddof=1)) if len(new)>1 else None
        summary['methods'][str(n)]['energy_violations']=int(new[f'energy_violations_{n}'].sum())
    if len(new)>1:
        rng=np.random.default_rng(20261004)
        idx=rng.integers(0,len(new),size=(20000,len(new)))
        diff=(new.l2_6-new.final_l2_learned).to_numpy()
        summary['six_vs_learned']={'mean_difference':float(diff.mean()),'ci95':np.quantile(diff[idx].mean(axis=1),[.025,.975]).tolist(),
                                  'relative_error_reduction_percent':float(100*(new.l2_6.mean()-new.final_l2_learned.mean())/new.l2_6.mean()),
                                  'structure_error_reduction_percent':float(100*(new.structure_6.mean()-new.structure_error_learned.mean())/new.structure_6.mean())}
    (DEST/'matched_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
