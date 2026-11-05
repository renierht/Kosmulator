# JLA supernova sample (official likelihood inputs)

Joint Light-curve Analysis of SDSS-II and SNLS, 740 SNe Ia:
Betoule et al. 2014, A&A 568, A22, doi:10.1051/0004-6361/201423413,
arXiv:1401.4064.
Release page: https://supernovae.in2p3.fr/sdss_snls_jla/ReadMe.html

The files here are the CosmoMC/Cobaya distribution of the JLA likelihood
inputs, copied unchanged from https://github.com/CobayaSampler/sn_data
(directory `JLA/`, repository commit 61d96434cafc2770928322c38e5a750e686368ae).

| File | Content |
| --- | --- |
| jla_lcparams.txt | per SN: zcmb, zhel, dz, m_B (mb, dmb), X_1 (x1, dx1), C (color, dcolor), host log10(M*/Msun) (3rdvar, d3rdvar), cov_m_s, cov_m_c, cov_s_c, set |
| jla_v0_covmatrix.dat | C_eta block for m_B |
| jla_va_covmatrix.dat | C_eta block for X_1 |
| jla_vb_covmatrix.dat | C_eta block for C |
| jla_v0a_covmatrix.dat | m_B - X_1 cross block |
| jla_v0b_covmatrix.dat | m_B - C cross block |
| jla_vab_covmatrix.dat | X_1 - C cross block |

Each matrix file holds N = 740 on the first line, then the N x N entries.

## Likelihood used by Kosmulator (observation tag "JLA")

mu = m_B - (M_B - alpha X_1 + beta C), with M_B -> M_B + Delta_M for host
log10(M*/Msun) > 10 (Betoule et al. 2014, Eqs. 4-5).

C(alpha, beta) = V0 + alpha^2 Va + beta^2 Vb + 2 alpha V0a - 2 beta V0b
- 2 alpha beta Vab + diag(dmb^2 + alpha^2 dx1^2 + beta^2 dcolor^2
+ 2 alpha cov_m_s - 2 beta cov_m_c - 2 alpha beta cov_s_c),
as in Cobaya's `sn.jla` (from CosmoMC). The coherent dispersion, lensing and
peculiar-velocity terms are already included in these files.

D_L = (1 + zhel) D_M(zcmb). alpha_JLA and beta_JLA are sampled (default priors
0.01 - 2 and 0.9 - 4.6, as Cobaya). The two absolute magnitudes (M_B and
M_B + Delta_M) enter linearly and are fitted analytically for every
parameter point, so H_0 is not constrained by JLA alone.

Check: flat LCDM best fit chi^2 = 682.7 for 740 SNe (the release notes quote
-2 ln L = 682.9 for the complete likelihood), Omega_m = 0.294.

## Checksums (sha256)

```
bae75653debe24058fef0fc5acaca844e9d069224c1a23d4c6a93fd60822d6ad  jla_lcparams.txt
269df269fb7685a1859b75903b0eb809d408cfea4a28eb8ee8a7f78ec2eb7c30  jla_v0_covmatrix.dat
d16f220c3c9e382c2662b73e862a32f0c5994bb42098d9129e8c92f04b2c7ff8  jla_v0a_covmatrix.dat
67c7dca567483283192cff1b86131aea4e82b1cf76a796f95052a1dce191ddb7  jla_v0b_covmatrix.dat
4e729ab4e8d802f4ab93050cd1e133c0fa047eff0116165888e6e29548bd3e45  jla_va_covmatrix.dat
63ceab98cec74d042015f656ba5904a0e99f5f25aeb140b2d3a88ae404e04778  jla_vab_covmatrix.dat
7f47e2d8d7cbc34267495fbc2aab413f0d69cc49fe7dec90295e5a64d2474d39  jla_vb_covmatrix.dat
```

## JLA_legacy

`Observations/JLA_legacy.dat` (tag "JLA_legacy") is the file Kosmulator used
as "JLA" before October 2026: 359 SNe whose redshifts are JLA zcmb values but
whose distance moduli and errors do not follow from the JLA light-curve
parameters. Its origin is not documented; it is kept only so that older runs
can be reproduced.
