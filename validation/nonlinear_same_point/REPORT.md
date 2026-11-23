# Unrestricted nonlinear IDE same-point check

Q=3H delta rho_dm^2/(rho_dm+rho_de). Parameters: Omega_m=0.296, H0=73.44, w=-0.913, w_dm=0, delta=-0.006, M_abs=-19.243; Omega_b=0.048. Same paper CLASS settings and unrestricted switches. Parameter vector is illustrative and is not asserted to be a best fit.

| Quantity | Preserved paper checkout | Current checkout |
|---|---:|---:|
| CLASS r_d (Mpc) | 136.103939979 | 136.103940318 |
| chi2 PantheonPS | 1452.424535 | 1451.740117 |
| chi2 DESI_DR2 | 9.013843 | 9.184763 |
| chi2 CC | 31.010891 | 40.997029 |

E(z) relative difference at seven BAO redshifts: 2.22e-16. CLASS r_d relative difference: 2.49e-09.
SN+BAO total chi2 change at this point: -0.513498.

Independent adaptive quadrature: old sparse-grid maximum relative distance error 0.005996; current 0.000002639 at this point. Raw Pantheon and DESI inputs are hashed in the two output JSON files. CC input files/covariance changed, so its chi2 difference does not isolate the effect of adding CC.

Single illustrative parameter point, not an optimum or posterior comparison.
CC files/covariance differ between implementations; CC column is not the effect of adding CC to a fit.
No LCDM point or refit included, so no delta IC attribution possible.
CLASS cached binaries only; no rebuild; no sampler or chain modifications.
