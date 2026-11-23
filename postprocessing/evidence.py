"""Equal-weight kNN evidence (Heavens et al. 2017; MCEvidence-compatible).

Only normalized rectangular uniform priors are supported. Saved likelihood
normalization is retained, so logZ is in those units. Bayes factors require
identical data and compatible likelihood constants. No sampler imports.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial import cKDTree
from scipy.special import gammaln, logsumexp

METHOD = 'heavens-2017-auto-knn-equal-weight-v1'


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def estimate(samples, log_likelihood, bounds, ks=(1, 2, 3)):
    """MAP scale estimator used by upstream MCEvidence, in log space.

    logZ = log[N/(Nk+1)] + log sum_i(V_di L_i) + log|J| - log V_prior.
    V_di is the kth-neighbour ball volume in whitened coordinates.
    Exact duplicates are rejected, never jittered or silently removed.
    """
    x = np.asarray(samples, dtype=float)
    ll = np.asarray(log_likelihood, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if x.ndim != 2 or ll.shape != (len(x),):
        raise ValueError('Samples and likelihoods must align exactly')
    n, d = x.shape
    if bounds.shape != (d, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError('Finite increasing bounds required for every sampled parameter')
    if not ks or any(int(k) != k or k < 1 for k in ks) or n < max(100, max(ks)+2, d+2):
        raise ValueError('Insufficient samples or invalid neighbour counts')
    if not np.isfinite(x).all() or not np.isfinite(ll).all() or np.any(np.abs(ll) > 1e90):
        raise ValueError('Invalid retained state or numerical failure sentinel; do not silently drop it')
    if np.any(x < bounds[:, 0]) or np.any(x > bounds[:, 1]):
        raise ValueError('Samples outside declared prior bounds')
    if np.unique(x, axis=0).shape[0] != n:
        raise ValueError('Duplicate states: increase temporal thinning; no deduplication or jitter allowed')
    cov = np.atleast_2d(np.cov(x, rowvar=False))
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Singular covariance; include sampled dimensions only') from exc
    y = solve_triangular(chol, (x-x.mean(axis=0)).T, lower=True).T
    radii = cKDTree(y).query(y, k=[int(k)+1 for k in ks], workers=1)[0]
    if np.any(radii <= 0) or not np.isfinite(radii).all():
        raise ValueError('Degenerate neighbour distances')
    log_jac = float(np.log(np.diag(chol)).sum())
    log_prior_volume = float(np.log(bounds[:, 1]-bounds[:, 0]).sum())
    log_ball = d/2*np.log(np.pi)-gammaln(1+d/2)
    values = {str(k): float(np.log(n)-np.log(n*k+1)+logsumexp(ll+log_ball+d*np.log(radii[:, j]))
                           +log_jac-log_prior_volume) for j, k in enumerate(ks)}
    # Balls crossing the rectangular support can bias kNN volumes upwards.
    # A theta_j boundary in whitened space has normal of norm sqrt(cov_jj).
    wall_distance = np.minimum(x-bounds[:, 0], bounds[:, 1]-x)/np.sqrt(np.diag(cov))
    boundary = {str(k): float(np.mean(np.any(wall_distance < radii[:, j, None], axis=1)))
                for j, k in enumerate(ks)}
    return dict(method=METHOD, n=n, ndim=d, logZ=values,
                log_prior_volume=log_prior_volume, log_whitening_jacobian=log_jac,
                neighbour_balls_crossing_box_fraction=boundary)


def read_completed_chain(spec):
    """Read only committed iterations, all walkers; require likelihood blobs.

    Legacy chains need an explicit audited config; this function never guesses
    a prior from sample extrema or treats log posterior as log likelihood.
    """
    import h5py
    p = Path(spec['chain'])
    meta = json.loads(Path(str(p)+'.metadata.json').read_text())
    conv = json.loads(Path(spec.get('convergence', str(p)+'.convergence.json')).read_text())
    status = conv.get('status', {})
    if not status.get('converged', status.get('ok', False)):
        raise ValueError('A passing completion/convergence record is required')
    if spec.get('prior_policy') != 'normalized_uniform_box_zero_likelihood_outside_domain':
        raise ValueError('Only explicit normalized-box priors are supported; coupled/conditioned priors require a separate implementation')
    names = list(meta['parameters'])
    if names != spec['parameters'] or set(names) != set(spec['priors']):
        raise ValueError('Prior and parameter identities must match metadata; fixed/derived dimensions are forbidden')
    if meta['model'] != spec['model']:
        raise ValueError('Model identity mismatch')
    if any(name in meta.get('fixed_parameters', {}) for name in names):
        raise ValueError('Fixed parameter included among sampled dimensions')
    if meta['model'] == 'w0waCDM':
        raise ValueError('CPL has a coupled prior; box-only implementation intentionally refuses it')
    burn = int(spec['burn'])
    if burn != int(conv['burn']):
        raise ValueError('Burn must match the supplied convergence assessment')
    with h5py.File(p, 'r') as f:
        g = f['mcmc']; n = int(g.attrs['iteration'])
        completed = int(status.get('stopped_at') or status.get('it') or spec.get('completed_steps', -1))
        if n != completed:
            raise ValueError('Chain length differs from completion record: possibly live or stale')
        if not 0 <= burn < n or 'blobs' not in g:
            raise ValueError('Invalid burn or missing saved likelihood blobs')
        x = np.asarray(g['chain'][burn:n]); ll = np.asarray(g['blobs'][burn:n])
        if ll.dtype.names:
            if ll.dtype.names != ('log_likelihood',):
                raise ValueError('Ambiguous structured likelihood blobs')
            ll = ll['log_likelihood']
        if ll.shape != x.shape[:2] or x.shape[2] != len(names):
            raise ValueError('Likelihood/sample dimensions do not align')
        if not np.isfinite(x).all() or not np.isfinite(ll).all() or np.any(np.abs(ll)>1e90):
            raise ValueError('Invalid retained samples or likelihood failure sentinels')
        bounds = np.asarray([spec['priors'][k] for k in names], dtype=float)
        if bounds.shape != (len(names),2) or np.any(x<bounds[:,0]) or np.any(x>bounds[:,1]):
            raise ValueError('Stored chain violates declared bounds')
        if 'log_prob' in g:
            posterior = np.asarray(g['log_prob'][burn:n])
            if posterior.shape != ll.shape or not np.allclose(posterior, ll, rtol=0, atol=1e-8):
                raise ValueError('Saved posterior differs from likelihood: nonconstant prior needs a dedicated normalized-prior implementation')
    return x, ll, bounds, meta, conv, n


def temporal_subset(x, ll, stride, seed, fraction=1., block='all', cap=50000):
    """Thin each walker in time with independently randomized offsets.

    Walkers from the same ensemble are NOT declared independent chains.
    Fractions use sampling without replacement (ordinary bootstrap is invalid
    for a nearest-neighbour estimator because it creates duplicates).
    """
    if stride < 1 or not 0 < fraction <= 1 or cap < 100:
        raise ValueError('Invalid thinning/subsample settings')
    rng = np.random.default_rng(seed)
    if block == 'first': x, ll = x[:len(x)//2], ll[:len(ll)//2]
    elif block == 'second': x, ll = x[len(x)//2:], ll[len(ll)//2:]
    elif block != 'all': raise ValueError('Unknown temporal block')
    if len(x) < 2*stride:
        raise ValueError('Too few temporal intervals for requested thinning')
    xs, ls = [], []
    for walker in range(x.shape[1]):
        offset = int(rng.integers(stride))
        xs.append(x[offset::stride, walker]); ls.append(ll[offset::stride, walker])
    xs, ls = np.concatenate(xs), np.concatenate(ls)
    count = min(cap, int(len(xs)*fraction))
    indices = rng.choice(len(xs), count, replace=False)
    return xs[indices], ls[indices]


def analyse(spec, seeds=(101,202,303), factors=(1,2,5), ks=(1,2,3)):
    x,ll,bounds,meta,conv,n = read_completed_chain(spec)
    tau = float(conv['status']['tau_max'])
    if not np.isfinite(tau) or tau <= 0: raise ValueError('Invalid saved autocorrelation time')
    trials=[]
    residual_correlation={}
    centered=x-x.mean(axis=0,keepdims=True)
    for factor in factors:
        lag=max(1,int(np.ceil(factor*tau)))
        if lag < len(x):
            left,right=centered[:-lag],centered[lag:]
            den=np.sqrt(np.sum(left*left,axis=(0,1))*np.sum(right*right,axis=(0,1)))
            residual_correlation[str(factor)]=(np.sum(left*right,axis=(0,1))/den).tolist()
    del centered
    settings=[(factor, fraction, 'all') for factor in factors for fraction in (.5,1.)]
    settings += [(2,1.,'first'),(2,1.,'second')]
    for factor,fraction,block in settings:
        for seed in seeds:
            stride = max(1,int(np.ceil(factor*tau)))
            row=dict(tau_factor=factor,stride=stride,fraction=fraction,block=block,seed=seed)
            try:
                a,b=temporal_subset(x,ll,stride,seed,fraction,block)
                row.update(estimate(a,b,bounds,ks))
                row['valid']=True
            except ValueError as exc:
                row.update(valid=False,error=str(exc))
            trials.append(row)
    primary=[r for r in trials if r['valid'] and r['tau_factor']==2 and r['fraction']==1 and r['block']=='all']
    warnings=['Approximate single-ensemble evidence; independent-ensemble confirmation unavailable.',
              'Numerical/physical invalid regions retain zero likelihood under the full rectangular prior.',
              'Saved likelihood units: omitted model-independent constants cancel only within the audited comparison group.',
              'Replicate spread is a sensitivity diagnostic, not a calibrated evidence uncertainty.']
    if len(primary)!=len(seeds):warnings.append('Some primary trials failed; do not quote a definitive Bayes factor.')
    nominal=float(np.median([r['logZ']['1'] for r in primary])) if primary else None
    valid=[r for r in trials if r['valid']]
    envelope=[min(r['logZ'][str(k)] for r in valid for k in ks),max(r['logZ'][str(k)] for r in valid for k in ks)] if valid else None
    if envelope and envelope[1]-envelope[0]>.3:warnings.append('Evidence changes by >0.3 log units across sensitivity checks.')
    if any(r['neighbour_balls_crossing_box_fraction']['1']>.01 for r in valid):warnings.append('Prior-boundary neighbour-volume bias is possible; >1% of k=1 balls cross the box.')
    if any(not r['valid'] for r in trials):warnings.append('Failed trials are retained in the report, not silently averaged away.')
    if any(abs(v)>.1 for v in residual_correlation.get('2',[])):
        warnings.append('Residual within-walker correlation exceeds 0.1 at the primary thinning stride.')
    source_files=[Path(spec['chain']),Path(str(spec['chain'])+'.metadata.json'),Path(spec.get('convergence',str(spec['chain'])+'.convergence.json'))]
    return dict(label=spec['label'],model=spec['model'],source=spec['chain'],source_hashes={str(p):file_hash(p) for p in source_files},
                parameters=meta['parameters'],priors=spec['priors'],prior_policy=spec['prior_policy'],
                burn=spec['burn'],steps=n,walkers=x.shape[1],retained_samples=int(ll.size),tau_max=tau,
                logZ_nominal=nominal,logZ_sensitivity_envelope=envelope,trials=trials,warnings=warnings,
                within_walker_lag_correlation=residual_correlation,
                status='provisional',primary_definition='median k=1 over seeds, stride=ceil(2*tau_max), all retained time',
                audit=spec.get('audit',{}))


def compare(results, reference):
    """Refuse unaudited/different data or likelihood normalizations."""
    by_name={r['label']:r for r in results}
    if len(by_name)!=len(results) or reference not in by_name:raise ValueError('Invalid comparison labels/reference')
    ref=by_name[reference]
    keys=('dataset_hashes','likelihood_convention','class_settings','zero_domain_convention')
    for r in results:
        for k in keys:
            if not ref['audit'].get(k) or r['audit'].get(k)!=ref['audit'][k]:
                raise ValueError('Evidence comparison provenance mismatch: '+k)
        z,z0=r['logZ_nominal'],ref['logZ_nominal']
        r['reference']=reference
        r['delta_logZ']=z-z0 if z is not None and z0 is not None else None
        env,env0=r['logZ_sensitivity_envelope'],ref['logZ_sensitivity_envelope']
        r['delta_logZ_sensitivity_envelope']=([env[0]-env0[1],env[1]-env0[0]] if env and env0 else None)
        if r is ref:r['delta_logZ_sensitivity_envelope']=[0.,0.]
    return results
