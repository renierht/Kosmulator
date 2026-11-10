# CC integration audit — 2026-11-10

Scope: six modified rows and three correlations from upstream 34091d1 checked
against primary papers. Other rows retain upstream source attributions; this is
not a new verification of the entire compilation or its sample independence.

| Redshift | Change | Primary source |
|---|---|---|
| 0.4497 | Correct 0.44497 typo | [Moresco 2016 Table 3](https://arxiv.org/html/1601.01701) |
| 0.46 | sigma 12.89 to 12.33 | [Loubser 2025 equations 4–6](https://arxiv.org/html/2511.02730v1) |
| 0.67 | sigma 23.03 to 17.82 | Same source |
| 0.83 | sigma 25.15 to 18.13 | Same source |
| 0.47 | sigma 23 to 49.6 | [Ratsimbazafy 2017](https://arxiv.org/abs/1702.00418) |
| 0.50 | sigma 41.2 to 34.68 | [Loubser et al. 2025](https://arxiv.org/abs/2506.03836) |

Errors combine the separately reported statistical and systematic components in
quadrature, rounded as upstream. Loubser section 3.3 gives correlations between
measurements (0.46,0.67)=0.932, (0.46,0.83)=0.830, (0.67,0.83)=0.776. Following
upstream, interpret these as total-error correlation coefficients and construct
C_ij=rho_ij sigma_i sigma_j. No additional correlation coefficients are inferred.

The loader requires CC_corr.txt, matches each referenced redshift uniquely,
rejects invalid/duplicate pairs and non-positive-definite covariance, and solves
by Cholesky. The CC likelihood uses the full precision matrix. Generic run
manifests hash both CC.dat and CC_corr.txt. Old source or dataset hashes reject
resume. Existing saved chains must remain with their original checkout; fresh
chains are required for this changed likelihood. The paper launcher has no CC
selection, so its current dataset combinations are unaffected by this change.

Validation: 28 tests and 83 subtests pass, including actual 36-row data, independent
linear-solve likelihood agreement, invalid pair rejection and changed-data resume
rejection. validation/audit_cc_update.py records a fixed-parameter comparison for
LCDM, CPL and IDE 9/10 in cc_update_audit.json. Minimum covariance eigenvalue is
12.0166. At LCDM Omega_m=.3,H0=70, old chi-square is 16.3037, revised diagonal
chi-square 16.8115, and correlated chi-square 24.8380. These are not minima or
posterior comparisons. Fixed covariance normalization constants are omitted as
in the existing chi-square convention; do not compare absolute IC values across
different dataset/covariance versions.

## Remaining limitations

Moresco 2020 optional SPS-model covariance has NOT been enabled or integrated.
The 36-row compilation therefore does not yet represent that extra systematic
budget. Sample overlap and cross-study systematic correlations are not established
by this audit. Those checks remain necessary before recommending a final CC
combination for publication. The existing z=.12 Wang row is retained, not newly
added or independently verified here. No production runs or IDE physics changed.
