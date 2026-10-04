#!/usr/bin/env python3
"""Reproduce existing paper chains through Kosmulator.main (serial, load-only).

Defaults: all regimes, polished criteria and 6000 derived-r_d samples.
DIC convention remains posterior-mean pending agreement. No fresh sampling.
"""
import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import traceback
from zipfile import ZipFile, ZIP_DEFLATED

class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,text):
        for s in self.streams:s.write(text);s.flush()
        return len(text)
    def flush(self):
        for s in self.streams:s.flush()

@contextmanager
def forbid_class_rebuild(cr):
    """Guard existing CLASS binaries while retaining the frozen implementation."""
    original_ensure, original_run = cr.ensure_class_ready, cr.run_model
    def ensure(model, **kwargs):
        kwargs.update(force=False, no_rebuild=True)
        return original_ensure(model, **kwargs)
    def blocked(*args, **kwargs):
        raise RuntimeError("CLASS rebuilding is prohibited in paper reproduction")
    cr.ensure_class_ready, cr.run_model = ensure, blocked
    try: yield
    finally: cr.ensure_class_ready, cr.run_model = original_ensure, original_run

def require(ok,message):
    if not ok:raise RuntimeError(message)

def chain_paths(config,root,regime):
    paths={}
    for name in ('lcdm',regime):
        case=config['cases'][name];rel=Path(case['chain'])
        require(not rel.is_absolute() and '..' not in rel.parts,'Unsafe archive path')
        path=(root/rel).resolve();require(path.is_relative_to(root.resolve()),'Path escapes archive root')
        require(path.is_file(),'Missing chain: '+str(path));paths[case['model']]=path
    return paths

def configure(kc,udm,K,config,regime):
    kc.model_names=['LCDM_v','NonLinear_IDE_2'];kc.true_model=config['reference_model']
    kc.observations=config['observations'];kc.nwalkers=config['walkers'];kc.burn=config['burn']
    # Load-only steering value, not provenance of the original MCMC run.
    kc.nsteps=max(x['completed_steps'] for x in config['cases'].values())
    kc.prior_limits=dict(kc.prior_limits,**{k:tuple(v) for k,v in config['prior_limits'].items()})
    for k,v in config['class_settings'].items():setattr(K,k,v)
    for k,v in zip(('ALLOW_NEGATIVE_ENERGIES','ALLOW_BIG_RIP','ALLOW_DOOM_FACTOR_INSTABILITIES'),config['regime_switches'][regime]):setattr(udm,k,v)

def check_chain(path,case,config,emcee,np):
    backend=emcee.backends.HDFBackend(str(path),read_only=True)
    require(backend.iteration==case['completed_steps'],'Wrong completed chain length')
    require(tuple(backend.shape)==(config['walkers'],len(case['parameters'])),'Wrong walker/parameter shape')
    require((backend.iteration-config['burn'])*config['walkers']==case['retained_samples'],'Wrong retention recipe')
    ll=backend.get_blobs(discard=config['burn'],thin=1,flat=True)
    require(ll is not None and len(ll)==case['retained_samples'] and np.all(np.isfinite(ll)),'Missing/invalid saved likelihood blobs')

def check_row(row,case,reference,polish):
    require(row['N']==1670 and row['k']==len(case['parameters']),'Wrong N/k')
    require(row['Parameter_names']==case['parameters'],'Wrong parameter order')
    require(row['Retained_samples']==case['retained_samples'] and row['Excluded_samples']==0,'Wrong retention')
    require(row['Likelihood_source'].startswith('saved likelihood blobs'),'Wrong likelihood source')
    require(abs(row['DIC']-reference['DIC'])<=1e-4,'Mean-DIC regression failed')
    if polish:
        require(abs(row['Chi_squared']-reference['Chi_squared'])<=5e-4,'Polished minimum regression failed')
        require(row['Optimizer_runs'] and all(x['success'] for x in row['Optimizer_runs']),'Optimizer diagnostics require review')
    else:require(not row['Optimizer_runs'],'Unexpected optimisation')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--chain-root',type=Path,required=True)
    parser.add_argument('--regime',choices=('all','iw','plus_iw','siw'),default='all')
    parser.add_argument('--derived-rd-samples',type=int)
    parser.add_argument('--no-polish',action='store_true')
    parser.add_argument('--output',type=Path,help='New, nonexistent output directory')
    parser.add_argument('--copy-to',type=Path,help='Existing directory receiving the results ZIP')
    args=parser.parse_args();root=args.repo.resolve();project=root/'reproducibility/SAIP2026_IDE'
    config=json.loads((project/'configs/paper.json').read_text())
    reference=json.loads((project/'reference_results/model_comparison.json').read_text())
    regimes=('iw','plus_iw','siw') if args.regime=='all' else (args.regime,)
    count=args.derived_rd_samples if args.derived_rd_samples is not None else config['derived_rd_samples']
    require(count>0,'Positive derived-r_d sample count required')
    require(config['burn']==1000 and config['thin']==1,'Unsupported paper retention')
    paths={r:chain_paths(config,args.chain_root,r) for r in regimes}
    if args.copy_to:require(args.copy_to.is_dir(),'Copy destination must exist')
    if args.output:
        output=args.output.resolve();require(not output.exists(),'Refusing existing output directory');output.mkdir(parents=True)
    else:output=Path(tempfile.mkdtemp(prefix='Kosmulator_paper_workflow_'))
    states={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for group in paths.values() for p in group.values()}
    source_names=('Kosmulator.py','User_defined_modules.py','Kosmulator_main/MCMC_setup.py','Kosmulator_main/utils.py','Kosmulator_main/Model_comparison.py','Kosmulator_main/Post_processing.py','Kosmulator_main/Class_run.py','Kosmulator_main/rd_helpers.py','Plots/Plots.py','Plots/Plot_functions.py')
    hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in source_names}
    report={'regimes':list(regimes),'derived_rd_samples':count,'polish':not args.no_polish,'engineering_output':args.no_polish or count<6000,'dic_convention':'posterior-mean; final consensus pending','source_sha256':hashes,'config':config,'python':sys.version,'platform':platform.platform(),'failures':[]}
    report['packages']={}
    for name in ('numpy','scipy','matplotlib','emcee','h5py','getdist'):
        try:report['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:report['packages'][name]=None
    os.environ['MPLBACKEND']='Agg';os.environ['KOSM_NO_CLASS_REBUILD']='1';sys.path.insert(0,str(root));os.chdir(root)
    print('RESULTS FOLDER:',output,flush=True);all_results={}
    with (output/'workflow.log').open('w') as log,redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
        try:
            import emcee
            import numpy as np
            import matplotlib.pyplot as plt
            import Kosmulator as kc
            import User_defined_modules as udm
            from Kosmulator_main import constants as K,Model_comparison as mc,MCMC_setup as setup,Class_run as cr
            require(kc.run_mcmc is setup.main,'Unexpected setup fallback')
            for module in (kc,udm,K,mc,setup,cr):require(Path(module.__file__).resolve().is_relative_to(root),'Unexpected module path')
            labels={'iw':'iwCDM','plus_iw':'+iwCDM','siw':'SiwCDM'}
            for regime in regimes:
                for name in ('lcdm',regime):
                    case=config['cases'][name];check_chain(paths[regime][case['model']],case,config,emcee,np)
                configure(kc,udm,K,config,regime);destination=output/regime
                suffix='SAIP2026_'+regime+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                settings={'save_root':str(destination/'plots'),'statistical_save_root':str(destination/'tables'),'derived_rd_samples':count,'derived_rd_seed':config['derived_rd_seed']}
                sys.argv=[str(root/'Kosmulator.py'),'--force_emcee','--num_cores','1','--plot_table','--output_suffix',suffix]
                plt.rcParams['text.usetex']=False
                with forbid_class_rebuild(cr):
                    kc.main(workflow_options={'existing_chain_paths':{m:str(p) for m,p in paths[regime].items()},'postprocessing_options':dict(config['postprocessing'],polish=not args.no_polish),'plot_settings_overrides':settings})
                stats=json.loads((destination/'tables/model_comparison.json').read_text())
                for name,label in (('lcdm','LCDM'),(regime,labels[regime])):
                    case=config['cases'][name];group=stats[case['model']];row=next(iter(group.values()))
                    check_row(row,case,next(iter(reference[label].values())),not args.no_polish);all_results[label]=group
                    figures=list((destination/'plots'/suffix/case['model']).rglob('*.png'))
                    require(any('corner' in p.name for p in figures) and any(p.name=='bestfit.png' for p in figures),'Missing normal figures')
                plt.close('all');print('NORMAL PAPER WORKFLOW PASS:',regime,flush=True)
            mc.compare_models(all_results,'LCDM');mc.export_comparison(all_results,output/'combined_tables')
        except Exception:traceback.print_exc();report['failures'].append('Workflow failed; see workflow.log')
        finally:
            report['source_files_unchanged']=all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
            report['chain_metadata_unchanged']=all([Path(p).stat().st_size,Path(p).stat().st_mtime_ns]==s for p,s in states.items())
            for name in ('source_files_unchanged','chain_metadata_unchanged'):
                if not report[name]:report['failures'].append(name+' failed')
            report['passed']=not report['failures'];(output/'workflow_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'README.txt').write_text('Current mean-DIC; coauthor consensus pending.\nEngineering output: '+str(report['engineering_output'])+' (small r_d subset or no polishing).\nUses normal main/setup/loader/plotting functions; load-only mode cannot sample or resume.\nChain size/mtime checks are not cryptographic hashes.\n')
    archive=Path(str(output)+'.zip')
    with ZipFile(archive,'x',ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(output))
    print('RESULTS ZIP:',archive)
    if args.copy_to:
        target=args.copy_to/archive.name;require(not target.exists(),'Refusing ZIP overwrite');shutil.copy2(archive,target);print('DOWNLOADS ZIP:',target)
    print('PAPER WORKFLOW:', 'PASS' if report['passed'] else 'REQUIRES REVIEW');return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
