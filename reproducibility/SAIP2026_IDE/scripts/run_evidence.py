#!/usr/bin/env python3
"""Optional evidence analysis of completed chains; no CLASS or sampler execution."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,csv,json,sys,re
from pathlib import Path
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root))
from postprocessing.evidence import analyse,compare,file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='New directory; existing reports are never overwritten')
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if a.output.exists():p.error('Output already exists; choose a new directory')
    for spec in cfg['runs']:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',spec['label']):p.error('Unsafe output label')
        for name,path in spec['audit']['dataset_files'].items():
            if file_hash(path)!=spec['audit']['dataset_hashes'][name]:p.error('Dataset file hash mismatch: '+name)
        for path,digest in spec['audit']['provenance_files'].items():
            if file_hash(path)!=digest:p.error('Provenance file hash mismatch: '+path)
    # Validate matching comparison conventions before any expensive calculation.
    compare([dict(label=s['label'],audit=s['audit'],logZ_nominal=None,logZ_sensitivity_envelope=None) for s in cfg['runs']],cfg['reference'])
    a.output.mkdir(parents=True)
    (a.output/'input_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    results=[]
    for spec in cfg['runs']:
        print('Analysing '+spec['label'],flush=True)
        r=analyse(spec);results.append(r)
        (a.output/(spec['label']+'.json')).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
        print(spec['label'],r['logZ_nominal'],r['logZ_sensitivity_envelope'],flush=True)
    compare(results,cfg['reference'])
    (a.output/'evidence.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    keys=['label','logZ_nominal','delta_logZ','status']
    with (a.output/'evidence.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in results)
    lines=['# Approximate Bayesian evidence','',
           'Positive ΔlnZ favours the model over the reference. Values are in saved likelihood units; absolute normalizations are not asserted.',
           'All results are provisional. Sensitivity ranges are NOT confidence/credible intervals. No independent ensembles were available.','',
           '| Model | lnZ | ΔlnZ | sensitivity range for ΔlnZ |','|---|---:|---:|---|']
    for r in results:
        number=lambda v: 'unavailable' if v is None else f'{v:.3f}'
        env=r['delta_logZ_sensitivity_envelope']
        interval='unavailable' if env is None else f'[{env[0]:.3f}, {env[1]:.3f}]'
        lines.append(f"| {r['label']} | {number(r['logZ_nominal'])} | {number(r['delta_logZ'])} | {interval} |")
    for r in results:
        lines.extend(['', '## '+r['label'],'']+['- '+v for v in r['warnings']])
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Completed:',a.output,flush=True)

if __name__=='__main__':main()
