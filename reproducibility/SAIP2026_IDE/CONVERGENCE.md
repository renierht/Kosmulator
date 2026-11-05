# Convergence diagnostics for the SAIP 2026 IDE analysis

This report documents diagnostics of the four final CLASS-based chains used in the SAIP analysis. It complements the brief manuscript statement and the [reproduction guide](REPRODUCE.md). It describes the original chains; independently generated chains require their own convergence assessment.

## Chains and retained samples

All four chains use 120 emcee walkers. The first 1000 steps per walker are discarded, with no thinning (`thin=1`). Retained sample counts are flattened counts across the walkers, not effective sample sizes.

| Model/regime | Completed steps per walker | Retained steps per walker | Retained samples |
|---|---:|---:|---:|
| ΛCDM | 4700 | 3700 | 444000 |
| iwCDM | 8500 | 7500 | 900000 |
| +iwCDM | 7300 | 6300 | 756000 |
| SiwCDM | 8200 | 7200 | 864000 |

Paths relative to the original `MCMC_Chains/` directory:

| Model/regime | Chain file |
|---|---|
| ΛCDM | `RD_CLASS_DIAG_5K/LCDM_v/DESI_DR2_PantheonP_SH0ES/DESI_DR2+PantheonP_SH0ES.h5` |
| iwCDM | `RD_CLASS_SEEDED_2K/NonLinear_IDE_2/DESI_DR2_PantheonP_SH0ES/DESI_DR2+PantheonP_SH0ES_FINAL_8500.h5` |
| +iwCDM | `RD_CLASS_PLUS_IW_8K/NonLinear_IDE_2/DESI_DR2_PantheonP_SH0ES/DESI_DR2+PantheonP_SH0ES.h5` |
| SiwCDM | `RD_CLASS_SIW_8K/NonLinear_IDE_2/DESI_DR2_PantheonP_SH0ES/DESI_DR2+PantheonP_SH0ES.h5` |

The original chains are not included in GitHub. These paths identify the analysed files; they are not required filenames for independent runs.

## Numerical diagnostics

| Model/regime | Maximum autocorrelation time τ_max (steps) | N_post / τ_max | Approximate minimum ESS | Mean acceptance fraction |
|---|---:|---:|---:|---:|
| ΛCDM | 36.938 | 100.17 | 12020 | 0.6441 |
| iwCDM | 53.778 | 139.46 | 16735 | 0.5504 |
| +iwCDM | 70.794 | 88.99 | 10679 | 0.5142 |
| SiwCDM | 72.698 | 99.04 | 11885 | 0.5138 |

Autocorrelation times were estimated from the unflattened post-burn chains using `emcee.autocorr.integrated_time` with `c=5` and `tol=50`. Here τ_max is the largest estimate among sampled parameters, and N_post is the retained number of steps per walker. None of the full retained chains triggered the estimator's length warning.

The approximate pooled ESS for each parameter is `120 * N_post / tau`; the table gives the minimum across sampled parameters. This is an ensemble approximation and does not assume that the interacting walkers are independent chains. A conventional between-chain Gelman–Rubin R-hat was therefore not calculated from the walkers of a single ensemble.

Acceptance fractions use the whole stored backend, including burn-in. The per-walker ranges were 0.6262–0.6645 (ΛCDM), 0.5389–0.5649 (iwCDM), 0.4937–0.5373 (+iwCDM), and 0.4962–0.5317 (SiwCDM). Acceptance is a supplementary diagnostic, not a convergence test by itself.

## Stability and visual checks

Trace plots were inspected for every sampled parameter, showing 16 evenly selected walkers together with the ensemble median and 16th–84th percentile band. They showed no obvious sustained post-burn drift.

Across the sampled parameters and regimes:

- Extending the analysed retained-chain prefix from 90% to 100% changed estimated autocorrelation times by at most 2.83%.
- First-half and second-half posterior medians differed by at most 2.34% of the full-chain 68% interval width.
- Changing burn-in from the adopted 1000 steps to 500 or 1500 steps changed medians by at most 0.28% of that interval width.
- Consecutive-quarter median comparisons showed small offsets without a sustained common trend.

The diagnostics were read-only; SHA256 comparisons confirmed that the four chain files remained unchanged.

## Why the completed lengths differ

Earlier run monitoring used approximately 100 autocorrelation times and stable autocorrelation estimates as a guiding target. The unrestricted iwCDM chain was extended after initial diagnostics indicated insufficient retained length; its final fixed-length continuation reached 8500 steps.

The table reports actual completed lengths and fresh final-chain diagnostics, not a common predetermined iteration count. The configured fractional autocorrelation-change tolerance was 0.01, but the available records do not establish that this automatic stopping criterion was satisfied at termination for every chain. The estimator setting `tol=50` above is distinct from that sampler setting.

The fresh retained-length ratios span 89–139 autocorrelation times. In particular, +iwCDM does not exceed 100τ under these final estimates, so 100τ should be described as a guiding target rather than a verified threshold met by all chains.

## Interpretation and scope

Together, the autocorrelation, trace and posterior-stability checks support adequate mixing and stable summaries for the sampled parameters within the explored regions. They do not prove that all possible posterior modes have been visited or independently establish the Monte Carlo precision of every derived quantity, including the derived sound horizon.

The report records the completed final-chain assessment; it does not reconstruct every original initialisation, resume operation or stopping decision. For independent runs, inspect convergence and stability rather than reproducing these chain lengths mechanically.
