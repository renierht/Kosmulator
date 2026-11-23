"""Analysis-only plotting override; no sampler/core files or source chains changed."""
import sys,runpy
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
if '--mode' not in sys.argv or sys.argv[sys.argv.index('--mode')+1]!='analyse':
    raise SystemExit('This plotting wrapper requires --mode analyse')
import Plots.Plots as plotting
original=plotting.MCSamples
def automatic_kde(*args,**kwargs):
    settings=dict(kwargs.get('settings') or {})
    settings['smooth_scale_2D']=-1.0
    kwargs['settings']=settings
    return original(*args,**kwargs)
plotting.MCSamples=automatic_kde
print('Postprocessing-only override: GetDist automatic 2D bandwidth; unchanged samples, priors and 1D settings.',flush=True)
runpy.run_path(str(root/'reproducibility/SAIP2026_IDE/scripts/run_analysis.py'),run_name='__main__')
