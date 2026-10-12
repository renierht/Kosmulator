# ACT DR6 + Planck PR4 lensing (CMB_lensing_ACTPlanck)

`v1.2/` is the ACT DR6 lensing likelihood data release v1.2 (`ACT_dr6_likelihood_v1.2.tgz`, 2023-12-22;
ACT's own README is `v1.2/README`), from
https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/data/ , used through the
`act_dr6_lenslike` package (`pip install act_dr6_lenslike`).

In git: the band powers, covariances, binning matrices and fiducial spectra (5 MB). Not in git: the twelve
correction matrices in `v1.2/like_corrs/` (`N1*.txt`, `*.npy`; 225 - 384 MB each, 2.8 GB), which are above
GitHub's 100 MB file limit. Lensing without primary CMB uses the CMB-marginalised covariance and does not
need them. With primary CMB in the group the likelihood corrections need them only in binned form, which
is in git:

`actplanck_baseline_corrections_v1.2.npz` (2.9 MB): the corrections of `actplanck_baseline` (nsims 792 / 400,
spectra to l = 2998) binned once (`Kosmulator_main/ground_cmb.py`, `_act_lens_compact`), with the settings
and the act_dr6_lenslike version (1.2.1) it was made with, which Kosmulator checks before using it. It gives
the package's own reference chi^2 (21.46) and equals the form made from the 2.8 GB (identical arrays). To
remake it after a new ACT release or a change of the binned form, unpack the .tgz above into `v1.2/` and run

```bash
python -c "from Kosmulator_main import ground_cmb as G; print(G.write_act_lens_shipped())"
```

`.cache/` (not in git) holds the same binned form, made from the 2.8 GB on machines that have them.

References: Madhavacheril et al. (arXiv:2304.05203); Qu et al. (arXiv:2304.05202); Carron, Mirmelstein
& Lewis, "CMB lensing from Planck PR4 maps" (arXiv:2206.07773).
