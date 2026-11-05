"""Posterior reference summaries and guarded, approximate Wilks comparisons.

Neither output is Bayesian evidence. No chains or likelihoods are modified.
"""
import json
import csv
from pathlib import Path
import numpy as np
from scipy.stats import chi2, norm

REFERENCES = dict(delta=0., delta_dm=0., delta_de=0., w=-1., w_dm=0., w0=-1., wa=0.)
# Other models require an explicit nesting audit before enabling Wilks output.
NESTED = {'w0waCDM': {'w0': -1., 'wa': 0.},
          'SignSwitch_IDE_1': {'w': -1., 'delta': 0., 'w_dm': 0.},
          'SignSwitch_IDE_2': {'w': -1., 'delta': 0., 'w_dm': 0.}}

def likelihood_ratio(improvement, df):
    if not np.isfinite(improvement) or improvement < 0 or df <= 0 or int(df) != df:
        return None, None
    p = float(chi2.sf(improvement, df))
    return p, float(norm.isf(p/2)) if p > 0 else None

def posterior_checks(samples, names, bounds, fixed=None, chain=None, prior_probe=None,
                     restricted=False, converged=None):
    a = np.asarray(samples, float)
    out = {'label': 'Posterior reference checks; not detection significance or evidence',
           'converged': converged, 'parameters': {}}
    if a.ndim != 2 or a.shape[1] != len(names) or not np.isfinite(a).all():
        out['status'] = 'unavailable: invalid posterior samples'; return out
    for name, reference in REFERENCES.items():
        if name not in names:
            if name in (fixed or {}):
                out['parameters'][name] = {'status': 'fixed', 'value': float(fixed[name]), 'reference': reference}
            continue
        j = names.index(name); x = a[:, j]
        q = np.quantile(x, [.0015, .025, .16, .5, .84, .975, .9985])
        row = dict(reference=reference, median=float(q[3]), interval68=[float(q[2]),float(q[4])],
                   interval95=[float(q[1]),float(q[5])], interval99_7=[float(q[0]),float(q[6])],
                   probability_below=float(np.mean(x < reference)),
                   probability_above=float(np.mean(x > reference)),
                   probability_equal=float(np.mean(x == reference)), sigma=None,
                   status='provisional: no independent-ensemble confirmation', batch_mcse={})
        lo, hi = bounds[name]
        permitted = lo < reference < hi
        if permitted and prior_probe is not None:
            center = np.median(a,axis=0)
            for value in (reference, reference-1e-5*(hi-lo), reference+1e-5*(hi-lo)):
                probe = center.copy(); probe[j] = value
                permitted = permitted and bool(prior_probe(probe))
        tail = min(row['probability_below'],row['probability_above'])
        row['favored_direction'] = 'below' if row['probability_below'] > row['probability_above'] else 'above'
        if restricted or not permitted:
            row['status'] = 'prior/domain restricted: sigma suppressed'
        elif tail == 0 or row['probability_equal'] > 0:
            row['status'] = 'unresolved tail or atom at reference: sigma suppressed'
        elif min(np.sum(x<reference),np.sum(x>reference)) < 20:
            row['status'] = 'sparse tail: sigma suppressed'
        else:
            row['sigma'] = float(norm.isf(tail))
        if chain is not None:
            events = np.asarray(chain)[:,:,j] < reference
            for block in (100,200,500):
                series = events.mean(axis=1); n = len(series)//block
                if n >= 4:
                    means=series[:n*block].reshape(n,block).mean(axis=1)
                    row['batch_mcse'][str(block)] = dict(blocks=n, se=float(means.std(ddof=1)/np.sqrt(n)))
        if not row['batch_mcse']:
            row['uncertainty_note'] = 'Correlated tail uncertainty unavailable; flattened draws are not independent.'
        if converged is not True:
            row['convergence_note'] = 'Convergence unconfirmed: treat summaries as provisional.'
        out['parameters'][name] = row
    return out

def posterior_for_group(samples, cfg, index, source):
    from Kosmulator_main import Kosmulator_MCMC as km
    import h5py
    chain=None; converged=None
    paths=[source] if isinstance(source,(str,Path)) else (source or [])
    for path in paths:
        if not Path(path).is_file(): continue
        with h5py.File(path,'r') as f:
            if 'mcmc' in f:
                n=int(f['mcmc'].attrs['iteration']); raw=f['mcmc/chain'][int(cfg.get('burn',0)):n]
            elif 'samples' in f and f['samples'].ndim==3:
                raw=f['samples'][int(cfg.get('burn',0)):]
            else: continue
            if raw.reshape(-1,raw.shape[-1]).shape==np.shape(samples) and np.array_equal(raw.reshape(-1,raw.shape[-1]),samples):
                chain=raw;converged=bool(f.attrs['converged']) if 'converged' in f.attrs else None;break
    import User_defined_modules as udm
    flags=cfg.get('_postprocessing_switches',{k:getattr(udm,k) for k in ('ALLOW_NEGATIVE_ENERGIES','ALLOW_BIG_RIP','ALLOW_DOOM_FACTOR_INSTABILITIES')})
    restricted=any(not v for v in flags.values()) and 'IDE' in cfg.get('model_name','')
    return posterior_checks(samples,list(cfg['parameters'][index]),cfg['prior_limits'][index],
                            cfg.get('fixed_parameters',{}),chain,
                            lambda t: np.isfinite(km.log_prior_all(t[None,:],cfg,index)).all(),restricted,converged)

def validate_nesting(model, reference, row, ref, cfg, rcfg, data, index, ri):
    """Check actual prior/likelihood nesting, local background rank and fit quality.

    Passing numerical checks does not establish the Wilks sampling distribution.
    """
    from Kosmulator_main import Kosmulator_MCMC as km
    import User_defined_modules as udm
    if reference != 'LCDM_v' or model not in NESTED:
        return {'status':'unavailable: model pair not audited'}
    names=cfg['parameters'][index]; rn=rcfg['parameters'][ri]; null=NESTED[model]
    extra=[p for p in names if p not in rn]
    if not extra or set(extra)-set(null) or set(rn)-set(names):
        return {'status':'unavailable: incompatible parameter sets'}
    if row['Data_counts'] != ref['Data_counts']:
        return {'status':'unavailable: different datasets'}
    for c,r,nm in [(cfg,row,names),(rcfg,ref,rn)]:
        if not r.get('Optimizer_runs') or not all(t.get('success',False) for t in r['Optimizer_runs']):
            return {'status':'unavailable: optimization not fully successful'}
        for p,v in r['MLE_parameters'].items():
            lo,hi=c['prior_limits'][index if c is cfg else ri][p]
            if min(v-lo,hi-v) <= 1e-6*(hi-lo): return {'status':'unavailable: optimum at prior boundary'}
    for p in rn:
        if tuple(cfg['prior_limits'][index][p]) != tuple(rcfg['prior_limits'][ri][p]):
            return {'status':'unavailable: common priors differ'}
    if sorted(cfg['observations'][index]) != sorted(rcfg['observations'][ri]):
        return {'status':'unavailable: observation groups differ'}
    fixed=cfg.get('fixed_parameters',{})
    for p,v in rcfg.get('fixed_parameters',{}).items():
        if p not in null and fixed.get(p) != v:
            return {'status':'unavailable: common fixed parameters differ'}
    for p,v in null.items():
        if p in names:
            lo,hi=cfg['prior_limits'][index][p]
            if not lo<v<hi: return {'status':'unavailable: null at/outside prior boundary'}
        elif fixed.get(p,v)!=v: return {'status':'unavailable: fixed parameter excludes null'}
    flags=row.get('Physicality_switches',{})
    if 'IDE' in model and any(not v for v in flags.values()):
        return {'status':'unavailable: restricted IDE regime requires calibration'}
    old={k:getattr(udm,k) for k in flags}
    try:
        for k,v in flags.items():setattr(udm,k,v)
        def evaluate(c,m,idx,p):
            ns=c['parameters'][idx];t=np.array([[p[n] for n in ns]])
            prior=np.asarray(km.log_prior_all(t,c,idx))
            if not np.isfinite(prior).all():return np.nan
            if not np.allclose(prior,0.,atol=1e-12):
                raise ValueError('Nonuniform prior: MAP/ML equivalence not established')
            return float(np.asarray(km.log_likelihood_all(t,data,c,udm.Get_model_function(m),m,c['observations'][idx],c['observation_types'][idx],idx)).ravel()[0])
        differences=[];base=dict(ref['MLE_parameters']);base.update(fixed);base.update(null)
        for offset in (0.,-.002,.002):
            p=base.copy();p['Omega_m']+=offset
            l0=evaluate(rcfg,reference,ri,p);l1=evaluate(cfg,model,index,p)
            if not np.isfinite([l0,l1]).all() or max(abs(l0),abs(l1))>1e90:
                return {'status':'unavailable: null likelihood invalid'}
            differences.append(abs(2*(l1-l0)))
        if max(differences)>1e-3:
            return {'status':'unavailable: LCDM likelihood mismatch','null_delta_chi2':differences}
        # Verify both sides of each extra parameter are permitted, and locally
        # distinct effects on the background exist (necessary, not sufficient).
        derivatives=[];z=np.linspace(.02,2.5,40)
        for name in extra:
            h=1e-4;plus=base.copy();minus=base.copy();plus[name]+=h;minus[name]-=h
            near=np.array([evaluate(cfg,model,index,plus),evaluate(cfg,model,index,minus)])
            if not np.isfinite(near).all() or (np.abs(near)>1e90).any():
                return {'status':'unavailable: null neighborhood invalid'}
            f=udm.Get_model_function(model)
            derivatives.append((np.asarray(f(z,plus))-np.asarray(f(z,minus)))/(2*h))
        jac=np.array(derivatives).T
        if not np.isfinite(jac).all() or np.linalg.matrix_rank(jac)!=len(extra):
            return {'status':'unavailable: local background identifiability check failed'}
        return {'status':'approximate Wilks: numerical nesting checked; asymptotic calibration unverified',
                'df':len(extra),'null_delta_chi2':differences,'local_background_rank':int(np.linalg.matrix_rank(jac))}
    except (ValueError, FloatingPointError, ArithmeticError) as exc:
        return {'status':'unavailable: null check failed: '+str(exc)}
    finally:
        for k,v in old.items():setattr(udm,k,v)

def attach_lrt(results,reference):
    for model,groups in results.items():
        for key,row in groups.items():
            row.update(LRT_sigma=None,LRT_p=None,LRT_df=None)
            check=row.get('Nesting_check',{})
            row['LRT_status']='reference model' if model==reference else check.get('status','unavailable: nesting/reference not checked')
            if model != reference and not np.isfinite(row.get('dChi_squared',np.nan)):
                row['LRT_status']='unavailable: matching reference absent'
            if check.get('df') and np.isfinite(row.get('dChi_squared',np.nan)):
                p,z=likelihood_ratio(-row['dChi_squared'],check['df'])
                row.update(LRT_p=p,LRT_sigma=z,LRT_df=check['df'])
                if p is None:row['LRT_status']='unavailable: alternative fit worse or invalid'
    return results

def export_reports(results,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    posterior={m:{k:r.get('Posterior_preferences',{'status':'unavailable: posterior not supplied'}) for k,r in groups.items()} for m,groups in results.items()}
    (folder/'posterior_preferences.json').write_text(json.dumps(posterior,indent=2,allow_nan=False)+'\n')
    rows=[]
    for m,groups in results.items():
        for key,r in groups.items():
            rows.append(dict(Model=m,Observation=key,Reference=r.get('Reference_model'),
                             **{k:r.get(k) for k in ('LRT_df','LRT_p','LRT_sigma','LRT_status')}))
    with (folder/'likelihood_ratio_significance.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['Model']);w.writeheader();w.writerows(rows)
    lines=['# Posterior reference checks','', 'Sigma is a Gaussian-equivalent posterior tail, not a detection significance or Bayes factor. Intervals are equal-tailed. Correlated batch uncertainty and convergence flags are in the JSON. All results remain conditional on explored modes and numerical likelihood domains.','']
    for m,groups in posterior.items():
        for key,r in groups.items():
            lines.extend([f'## {m}: {key}','','| Parameter | Reference | P(below) | P(above) | Tail sigma | Status |','|---|---:|---:|---:|---:|---|'])
            for name,row in r.get('parameters',{}).items():
                lines.append('| '+' | '.join(str(v) for v in [name,row['reference'],row.get('probability_below','—'),row.get('probability_above','—'),row.get('sigma','—'),row['status']])+' |')
            lines.append('')
    (folder/'posterior_preferences.md').write_text('\n'.join(lines)+'\n')
