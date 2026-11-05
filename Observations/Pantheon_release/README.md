# Pantheon (2018): release files for the systematic covariance

Two files of the Pantheon release (Scolnic et al. 2018, ApJ 859, 101, arXiv:1710.00845), unchanged,
from github.com/dscolnic/Pantheon:

| File | md5 | Content |
|---|---|---|
| `lcparam_full_long.txt` | 1e1af8f17d62a5aabfac1040874dde40 | light-curve parameters of the 1048 SNe (zcmb, mb, dmb, ...) |
| `sys_full_long.txt` | fa8cd7843126b81eb158300d6b8cb09f | systematic covariance of mb, 1048 x 1048 (first line: 1048, then one value per line) |

Kosmulator's `Pantheon.dat` holds the same SNe sorted by redshift, with columns zcmb, mb + 19.3 and dmb.
With `PANTHEON_SYS_COV = "full"` (constants.py, default None) `Config.pantheon_sys_covariance` matches
every row of `Pantheon.dat` to one row of `lcparam_full_long.txt` (zcmb, mb + 19.3 and dmb) and reorders
the systematic covariance to `Pantheon.dat`'s order; the likelihood then uses C = diag(dmb^2) + C_sys, as
the release's CosmoMC data set file does (`full_long.dataset`: `mag_covmat_file = sys_full_long.txt`,
`pecz = 0`, `intrinsicdisp = 0`). Without it the errors are dmb only, i.e. the paper's statistical-only
analysis.

Redshifts: `Pantheon.dat` has the zcmb of `lcparam_full_long.txt`, in which zhel is set equal to zcmb.
The release also has `lcparam_full_long_zhel.txt` (the file Cobaya's Pantheon likelihood reads), with
the heliocentric redshifts and slightly different zcmb for 802 of the 1048 SNe (rms 3.5e-4, at most
1.7e-3; mb, dmb and the systematic covariance are the same). With that file Omega_m (flat LCDM, Pantheon
alone) is higher by about 0.001, 0.04 sigma (Kosmulator audit, 19 Oct 2026).

Flat LCDM, Pantheon alone (Scolnic et al. 2018, Table 8 and Sect. 6.1): Omega_m = 0.284 +- 0.012
(statistical only) and 0.298 +- 0.022 (statistical and systematic).
