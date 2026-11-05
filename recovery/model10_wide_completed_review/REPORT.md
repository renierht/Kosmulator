# Model 10 prior-sensitivity review

Both chains used burn=1000 and all 120 walkers. New run converged at 44700 steps.

| Quantity | Original | Wider prior |
|---|---:|---:|
| Chi_squared | 1462.910791 | 1462.910791 |
| AIC | 1472.910791 | 1472.910791 |
| AICc | 1472.946849 | 1472.946848 |
| BIC | 1500.013686 | 1500.013685 |
| D_bar | 1467.930406 | 1467.900553 |
| D_at_mean | 1464.105441 | 1468.834478 |
| p_D | 3.824964 | -0.933925 |
| DIC | 1471.755370 | 1466.966628 |

Wider-run differences relative to the saved LCDM benchmark (not the zero self-comparisons in its native single-model table):
- Delta Chi_squared: -9.804401
- Delta AIC: -5.804401
- Delta AICc: -5.782750
- Delta DIC: -11.736784
- Delta BIC: 5.036756

The DIC decrease is driven by D_at_mean increasing, while D_bar and the best fit barely change. Negative p_D means posterior-mean DIC is problematic as a complexity/model-ranking summary here. The curved Omega_m–delta ridge provides a plausible explanation: averaging parameter vectors can leave its high-likelihood region. Do not interpret this as improved best-fit quality.

26.33% of new samples have delta<-1; 10.50% have Omega_m>0.5. Restricting the new chain to the old rectangular bounds reproduces the original constraints closely. This supports prior truncation as the explanation of the shifts. Only 0.103% have delta<-1.9; 0.048% have delta<-1.95; none have Omega_m>0.65.

Burn=1000/3000/5000 medians are nearly identical. Larger burns leave insufficient length and Rhat marginally above the configured thresholds; they must not be labelled converged under the existing policy. The primary burn=1000 chain passes. Independent-ensemble confirmation remains outstanding.

The wider Omega_m prior differs from the LCDM benchmark, so no new common-prior LRT certification is asserted. No likelihood, chains or original reports were modified. See comparison.json for raw checks and posterior_comparison.png/pdf for the overlay.
