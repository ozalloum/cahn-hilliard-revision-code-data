"""Sequential numerical reproduction. No manuscript/PDF authoring is performed.

Example: python run_campaign.py --stage all --output-dir results_reproduced
The default output directory is the sibling 'results' directory. Existing
completed fitted/rollout/convergence files are reused; use a new output path
for a complete independent rerun. Timing runs are sequential and isolated
from other campaign stages. Requirements: Python 3.12, NumPy, pandas.
"""
import argparse,subprocess,sys,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['verify','legacy','convergence','finer','uncertainty','stress','cost','summary','all'],default='verify');p.add_argument('--output-dir');a=p.parse_args()
here=Path(__file__).resolve().parent
env=os.environ.copy()
if a.output_dir:env['CH_VALIDATION_RESULTS']=str(Path(a.output_dir).resolve())
commands={
'verify':[('verify_kernel.py',),('kernel_fast.py',),('verify_long_reference.py',)],
'legacy':[('fit_legacy.py',),('evaluate_legacy.py','--stage','short'),('evaluate_legacy.py','--stage','long')],
'convergence':[('convergence.py','--stage','pilot'),('convergence.py','--stage','fine'),('convergence.py','--stage','ensemble'),('convergence.py','--stage','alias'),('convergence.py','--stage','custom','--case','random0','--n','768','--h','5e-5')],
'finer':[('resolved_models.py','--n','128'),('resolved_models.py','--n','256')],
'uncertainty':[('uncertainty_shifts.py',)],
'stress':[('stress.py',),('stress_rollout.py',)],
'cost':[('reference_substeps.py',),('cost.py',)],
'summary':[('convergence.py','--stage','summary'),('evaluate_legacy.py','--stage','summary'),('analyze.py',)]}
for stage,cmds in commands.items():
    if a.stage not in ['all',stage]:continue
    for cmd in cmds:
        print('Running',stage,':',' '.join(cmd),flush=True)
        subprocess.run([sys.executable,'-u',str(here/cmd[0]),*cmd[1:]],env=env,check=True)
