# SPT-3G MUSE lensing (CMB_lensing_SPT)

`90_150_220.h5` is the data file of the SPT-3G 2019 - 2020 delensed EE + optimal phi phi (MUSE)
likelihood, unchanged: `src/muse3glike/dat/90_150_220.h5` of `muse_3g_like_march_2025.zip`
(muse3glike 1.0.0), from https://pole.uchicago.edu/public/data/ge25/
(md5 8839ad236d98022ba54d4de8a0ead995).

Kosmulator uses the 16 phi phi band powers only (`Kosmulator_main/ground_cmb.py`, `MusePhiPhi`,
a numpy port of the package's likelihood with `components=["ϕϕ"]`); the delensed EE band powers
and the systematics parameters are marginalised, as in the package.

Reference: F. Ge et al. (SPT-3G), "Cosmology From CMB Lensing and Delensed EE Power Spectra Using
2019-2020 SPT-3G Polarization Data", Phys. Rev. D 111, 083534 (2025), arXiv:2411.06000.
