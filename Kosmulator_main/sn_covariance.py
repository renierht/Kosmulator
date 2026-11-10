"""Explicit covariance loading for DESY5 (systematics) and Union3 (total)."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def load_sn_covariance(path, n, sigma_stat=None):
    """Load an N-prefixed covariance stream; never infer covariance vs precision.

    Add statistical variances only when supplied. Fail on missing, malformed,
    nonfinite or non-positive-definite total covariance, without pseudoinversion.
    """
    raw = np.loadtxt(path).reshape(-1)
    if raw.size != 1 + n*n or raw[0] != n:
        raise ValueError(f"{path}: covariance dimensions do not match {n} SN rows")
    cov = raw[1:].reshape(n,n).copy()
    if not np.isfinite(cov).all():
        raise ValueError(f"{path}: nonfinite covariance")
    if not np.allclose(cov, cov.T, rtol=1e-7, atol=1e-12):
        raise ValueError(f"{path}: asymmetric covariance")
    cov = (cov + cov.T) / 2
    if sigma_stat is not None:
        sigma_stat = np.asarray(sigma_stat, dtype=float)
        if sigma_stat.shape != (n,) or not np.isfinite(sigma_stat).all() or np.any(sigma_stat <= 0):
            raise ValueError("Statistical SN errors must be finite and positive")
        cov[np.diag_indices(n)] += sigma_stat**2
    factor = cho_factor(cov, lower=True, check_finite=True)
    precision = cho_solve(factor, np.eye(n), check_finite=False)
    return cov, (precision + precision.T) / 2
