# Bounded current-likelihood optimization

One-start Nelder-Mead fits, maximum 350 evaluations each, initialized from archived paper maxima. All four fits terminated successfully. No MCMC, DIC, or global-optimum certification. Same unrestricted nonlinear interaction, fixed Omega_b=0.048 and w_dm=0, free w and delta, original priors and CLASS settings.

| Data | LCDM chi2 | IDE chi2 | Delta chi2 | Delta AIC | Delta AICc | Delta BIC |
|---|---:|---:|---:|---:|---:|---:|
| SN+BAO | 1473.717715 | 1460.630677 | -13.087038 | -9.087038 | -9.065387 | 1.754119 |
| SN+BAO+CC | 1502.028810 | 1495.688453 | -6.340357 | -2.340357 | -2.319164 | 8.543456 |

Elapsed: 4.97 minutes. Total optimizer evaluations: 976.
Adding CC changes Delta AIC by 6.746681.

Historical paper Delta chi2=-11.298573, Delta AIC=-7.298573. Current no-CC results can be compared to these for this same model and prior setup; this is not a comparison to other external likelihood implementations. The baseline CC minimum agrees with its saved native result within 2e-6.

The updated no-CC fit favours IDE more strongly than the historical likelihood. Adding CC weakens that preference after both models refit. This supports CC as the main cause for this controlled comparison, not a failure of the IDE background or CLASS route. Results remain local optimization diagnostics; definitive posterior/DIC claims require fresh matched chains and independent convergence checks.
