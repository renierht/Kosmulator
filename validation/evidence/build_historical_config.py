"""Explicit audit/config for the legacy Pantheon+SH0ES + DESI DR2 runs."""
from pathlib import Path
import sys,json,ast,difflib
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from postprocessing.evidence import file_hash
base=root.parent;warm=base/'Kosmulator_warm_dm';follow=base/'Kosmulator_followup'
files=['PantheonP.dat','PantheonP.cov','DESI_DR2_synced.txt','DESI_DR2_covtot.txt']
hashes={name:file_hash(warm/'Observations'/name) for name in files}
assert all(file_hash(follow/'Observations'/name)==hashes[name] for name in files)
diffs=[]
for filename in ['Kosmulator_main/Kosmulator_MCMC.py','Kosmulator_main/rd_helpers.py','Kosmulator_main/Statistical_packages.py']:
 source=[(r/filename).read_text() for r in [warm,follow]]
 for line in difflib.unified_diff(source[0].splitlines(),source[1].splitlines(),fromfile=str(warm/filename),tofile=str(follow/filename)):
  diffs.append(line)
(Path(__file__).parent/'legacy_source_differences.diff').write_text('\n'.join(diffs)+'\n')
records=[('LCDM',warm,'Model9_unrestricted_20261014_iw','LCDM_v'),('Model9',warm,'Model9_fresh_20261016_iw','SignSwitch_IDE_1'),('Model10_original',follow,'Model10_unrestricted_v1_iw','SignSwitch_IDE_2'),('Model10_wide',follow,'Model10_wide_priors_v2_iw','SignSwitch_IDE_2')]
runs=[]
for label,repo,run,model in records:
 folder=repo/'MCMC_Chains'/run;p=next((folder/model).rglob('*.h5'))
 paper=repo/'reproducibility/SAIP2026_IDE/configs/paper.json';settings=json.loads(paper.read_text())
 prior_file=folder/'prior_run_settings.json'
 if prior_file.exists():settings.update(json.loads(prior_file.read_text()))
 names=json.loads(Path(str(p)+'.metadata.json').read_text())['parameters']
 priors=settings.get('priors',settings['prior_limits'])
 provenance=[paper,repo/'Kosmulator_main/Kosmulator_MCMC.py',repo/'Kosmulator_main/rd_helpers.py',repo/'Kosmulator_main/Statistical_packages.py',repo/'User_defined_modules.py']
 if prior_file.exists():provenance.append(prior_file)
 runs.append(dict(label=label,model=model,chain=str(p),parameters=names,burn=1000,priors={k:priors[k] for k in names},
  prior_policy='normalized_uniform_box_zero_likelihood_outside_domain',audit=dict(
   dataset_files={name:str(repo/'Observations'/name) for name in files},dataset_hashes=hashes,
   likelihood_convention='legacy PantheonPS + DESI_DR2: saved log-likelihood blobs; common omitted constants; no new likelihood reevaluation',
   class_settings=settings['class_settings'],
   zero_domain_convention='full rectangular prior, global background and CLASS numerical rejection retained as zero likelihood; no renormalization on solver success',
   provenance_files={str(p):file_hash(p) for p in provenance},
   provenance_quality='Legacy reconstruction: no contemporaneous dataset hashes. Existing files match across both checkouts; historical likelihood-path differences audited as DES-Dovekie/CPL additions only.',
   prior_source=str(prior_file if prior_file.exists() else paper),
   note='Includes BBN-table numerical domain. Background-only IDE. No independent ensemble. Model10 prior runs differ in both delta and Omega_m bounds.')))
config={'reference':'LCDM','runs':runs}
(root/'reproducibility/SAIP2026_IDE/configs/evidence_signswitch_legacy.json').write_text(json.dumps(config,indent=2)+'\n')
print('Audited configuration written for', [s['label'] for s in runs])
