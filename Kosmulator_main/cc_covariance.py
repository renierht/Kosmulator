"""Fixed CC covariance from published total-error correlation coefficients."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def attach_cc_correlations(data, path):
    """Require a valid pair list; ambiguous redshift matches fail closed."""
    z = np.asarray(data['redshift'], dtype=float)
    sigma = np.asarray(data['type_data_error'], dtype=float)
    if z.ndim != 1 or sigma.shape != z.shape or not np.isfinite(z).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError('Invalid CC redshifts or errors')
    pairs = np.loadtxt(path, ndmin=2)
    if pairs.shape[1] != 3 or not np.isfinite(pairs).all():
        raise ValueError('CC correlation file must have finite z_i, z_j, rho rows')
    covariance = np.diag(sigma**2)
    seen = set()
    for zi, zj, rho in pairs:
        indices = []
        for zz in (zi,zj):
            matches = np.flatnonzero(np.isclose(z,zz,atol=1e-6,rtol=0))
            if len(matches) != 1:
                raise ValueError(f'CC correlation z={zz} matches {len(matches)} rows')
            indices.append(int(matches[0]))
        i,j = indices
        key = tuple(sorted(indices))
        if i == j or key in seen or abs(rho) >= 1:
            raise ValueError('Invalid or duplicate CC correlation pair')
        seen.add(key)
        covariance[i,j] = covariance[j,i] = rho*sigma[i]*sigma[j]
    factor = cho_factor(covariance,lower=True)
    precision = cho_solve(factor,np.eye(len(z)))
    data['cov'] = covariance
    data['inv_cov'] = (precision+precision.T)/2
    return data
