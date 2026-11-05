# Four-model comparison

DESI DR2 + Pantheon+ with SH0ES; N=1670; fixed burn=1000; native polished likelihood maxima and posterior-mean DIC. All differences are model minus LCDM.

| Model | chi2 | AIC | AICc | DIC | BIC |
|---|---:|---:|---:|---:|---:|
| LCDM | 1472.715 | 1478.715 | 1478.730 | 1478.703 | 1494.977 |
| CPL | 1464.089 | 1474.089 | 1474.125 | 1474.023 | 1501.192 |
| IDE Model 9 | 1461.434 | 1471.434 | 1471.470 | 1471.414 | 1498.537 |
| IDE Model 10 | 1462.911 | 1472.911 | 1472.947 | 1471.755 | 1500.014 |

| Model | Delta AIC | Delta AICc | Delta DIC | Delta BIC | Approx. LRT sigma |
|---|---:|---:|---:|---:|---:|
| LCDM | 0.000 | 0.000 | 0.000 | 0.000 | unavailable / reference |
| CPL | -4.626 | -4.605 | -4.680 | 6.215 | 2.473 |
| IDE Model 9 | -7.281 | -7.259 | -7.289 | 3.560 | 2.915 |
| IDE Model 10 | -5.804 | -5.783 | -6.948 | 5.037 | 2.677 |

## Model 10 recovery

The original failure was reproduced in the best-fit plotting stage: the artificial vector made from all marginal lower endpoints is invalid in the CLASS Model 10 background. The median vector is valid. The native plot now omits this endpoint band and labels the omission; no fallback rd or likelihood modification was introduced. A proper joint posterior predictive band would require a separate calculation. The completed corner plot and original chain were preserved. The expensive 6000-draw rd corner calculation was not repeated. The recovered aligned parameter table lists sampled parameters; the existing corner plot retains the derived rd summary.

Native statistical analysis was rerun with the original paper polishing options and all retained samples. The other three native results were reused. Four-model exports include separate posterior-preference and likelihood-ratio eligibility reports.

Model 10 LRT status: approximate Wilks: numerical nesting checked; asymptotic calibration unverified.

Model 10 optimizer runs: [(True, 1462.9107914799686, 'Optimization terminated successfully.'), (True, 1462.910791893695, 'Optimization terminated successfully.'), (True, 1462.910791981198, 'Optimization terminated successfully.')].

Validation: 79 regression tests passed, including checks that invalid endpoints omit the band, while invalid central values and programming errors still propagate. The saved chains, priors and sampler configuration were not modified. Existing CLASS-domain limitations and independent-ensemble caveats remain.
