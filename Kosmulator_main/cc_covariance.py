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


# Moresco 2020 Table 3, percent biases; upstream 34091d1.
CC_MORESCO2020_TABLE3 = ((0.075, 0.47, 7.4, 15.86, 9.91), (0.125, 0.47, 7.4, 14.23, 6.98), (0.175, 0.47, 7.4, 13.34, 5.4), (0.225, 0.47, 7.4, 13.21, 5.4), (0.275, 0.47, 7.4, 13.29, 5.4), (0.325, 0.47, 7.4, 12.2, 5.4), (0.375, 0.47, 7.4, 12.99, 5.4), (0.425, 0.47, 7.4, 10.29, 6.2), (0.475, 0.46, 7.39, 8.91, 5.86), (0.525, 0.23, 7.4, 9.99, 6.51), (0.575, 0.28, 6.87, 10.09, 6.12), (0.625, 0.47, 6.65, 11.17, 6.21), (0.675, 0.47, 6.57, 11.12, 5.71), (0.725, 0.47, 5.9, 10.81, 5.16), (0.775, 0.45, 6.03, 10.75, 5.05), (0.825, 0.47, 6.1, 10.75, 5.05), (0.875, 0.47, 5.89, 9.08, 2.79), (0.925, 0.44, 5.8, 8.62, 3.7), (0.975, 0.4, 5.94, 7.32, 3.65), (1.025, 0.27, 6.07, 5.84, 3.37), (1.075, 0.2, 6.08, 6.02, 3.49), (1.125, 0.2, 6.07, 4.72, 2.33), (1.175, 0.19, 6.09, 4.31, 2.33), (1.225, 0.19, 6.09, 3.9, 2.33), (1.275, 0.19, 6.09, 3.9, 2.33), (1.325, 0.2, 6.09, 3.91, 2.34), (1.375, 0.19, 6.09, 3.9, 2.34), (1.425, 0.19, 6.09, 3.9, 2.33), (1.475, 0.2, 6.09, 3.91, 2.34))
CC_D4000_Z = (0.179, 0.199, 0.352, 0.3802, 0.4004, 0.4247, 0.4497, 0.4783, 0.593, 0.68, 0.781, 0.875, 1.037, 1.363, 1.965)


def cc_sps_covariance(z, measured_h, mode, *, include_library=True, extrapolation='error'):
    """Fixed-data Eq. 9 covariance on the 15 designated Moresco D4000 rows.

    Linear interpolation of Table 3; endpoint extension needs explicit 'hold'.
    Other studies receive no additional term, avoiding assumed applicability.
    """
    if mode not in ('full','ooo') or extrapolation not in ('error','hold'):
        raise ValueError('Require mode full/ooo and extrapolation error/hold')
    if not isinstance(include_library, (bool,np.bool_)):
        raise ValueError('include_library must be boolean')
    z, h = np.asarray(z,dtype=float), np.asarray(measured_h,dtype=float)
    if z.ndim != 1 or h.shape != z.shape or not np.isfinite(z).all() or not np.isfinite(h).all() or np.any(h <= 0):
        raise ValueError('Invalid CC redshifts or measured H')
    on = np.zeros(len(z),dtype=bool)
    for zz in CC_D4000_Z:
        hits=np.flatnonzero(np.isclose(z,zz,rtol=0,atol=1e-6))
        if len(hits)!=1:
            raise ValueError(f'SPS redshift {zz} matches {len(hits)} rows; expected one')
        on[hits[0]]=True
    table=np.asarray(CC_MORESCO2020_TABLE3)
    outside=on & ((z<table[0,0]) | (z>table[-1,0]))
    if outside.any() and extrapolation=='error':
        raise ValueError(f'CC SPS table does not cover redshifts {z[outside].tolist()}; explicitly choose hold to extend endpoints')
    columns=[1,3 if mode=='full' else 4]
    if include_library: columns.append(2)
    covariance=np.zeros((len(z),len(z)))
    for col in columns:
        v=np.where(on,np.interp(z,table[:,0],table[:,col])*h/100,0.)
        covariance+=np.outer(v,v)
    return covariance


def attach_cc_systematics(data, mode, *, include_library=True, extrapolation='error'):
    """Add an optional fixed covariance once, preserving the correlation base."""
    if mode is None:
        return data
    if 'cc_sps_policy' in data:
        raise ValueError('CC SPS covariance already attached')
    extra=cc_sps_covariance(data['redshift'],data['type_data'],mode,
                            include_library=include_library,extrapolation=extrapolation)
    base=np.asarray(data['cov']) if 'cov' in data else np.diag(np.asarray(data['type_data_error'])**2)
    covariance=base+extra
    factor=cho_factor(covariance,lower=True)
    precision=cho_solve(factor,np.eye(len(covariance)))
    data['cov']=covariance
    data['inv_cov']=(precision+precision.T)/2
    data['cc_sps_policy']=dict(mode=mode,include_library=include_library,extrapolation=extrapolation)
    return data
