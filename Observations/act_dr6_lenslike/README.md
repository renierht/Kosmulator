# ACT DR6 + Planck PR4 lensing (CMB_lensing_ACTPlanck)

`v1.2/` is the ACT DR6 lensing likelihood data release v1.2 (`ACT_dr6_likelihood_v1.2.tgz`, 2023-12-22;
ACT's own README is `v1.2/README`), from
https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/data/ , used through the
`act_dr6_lenslike` package (`pip install act_dr6_lenslike`).

In git: the band powers, covariances, binning matrices and fiducial spectra (5 MB). Not in git: the twelve
correction matrices in `v1.2/like_corrs/` (`N1*.txt`, `*.npy`; 225 - 384 MB each, 2.8 GB), which are above
GitHub's 100 MB file limit. They are needed only when the group also has primary CMB (the likelihood
corrections); lensing without primary CMB uses the CMB-marginalised covariance and works without them. To
add them, unpack the .tgz above into `v1.2/`.

`.cache/` (not in git) holds Kosmulator's binned form of the corrections (2.9 MB), written by the first run
that needs them.

References: Madhavacheril et al. (arXiv:2304.05203); Qu et al. (arXiv:2304.05202); Carron, Mirmelstein
& Lewis, "CMB lensing from Planck PR4 maps" (arXiv:2206.07773).
