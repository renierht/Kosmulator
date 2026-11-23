# Expanded unrestricted nonlinear IDE / LCDM check

Fourteen matched parameter points: archived LCDM MLE and four nearby points; archived nonlinear IDE MLE and eight nearby points. Perturbations: H0 +/-0.8, Omega_m +/-0.008, and IDE w +/-0.04, delta +/-0.002. Every current point also tests the old distance calculation, old SN redshift prefactor, and both together. All toggles are process-local; production files and chains unchanged.

| Evaluation (no refit) | LCDM chi2 | IDE chi2 | IDE minus LCDM |
|---|---:|---:|---:|
| Original SN+BAO | 1472.715200 | 1461.416626 | -11.298573 |
| Current distance, old SN prefactor | 1474.228025 | 1461.548567 | -12.679458 |
| Current SN+BAO | 1473.772022 | 1460.836197 | -12.935825 |
| Current SN+BAO+CC | 1504.191986 | 1501.839206 | -2.352780 |

Current CC differential: 10.583045; old CC differential: 9.706988.
Distance correction contribution to relative chi2: -1.380885; subsequent SN prefactor contribution: -0.256367.
Maximum |relative H difference|: 6.66134e-16; maximum |relative CLASS rd difference|: 2.02709e-08.
Largest residual SN+BAO chi2 mismatch after restoring both old numerical conventions in current code: 3.07817e-05.

The old SN+BAO likelihood at both archived best fits reproduces archived minimum chi2 within 1e-5. Distances are patched at both utils and UDM references (SN and DESI respectively). CLASS settings and parameters are identical.

Interpretation: for these old-fit points CC is the dominant change weakening IDE preference. The distance and SN corrections instead modestly strengthen it here. Even the original CC compilation penalizes the IDE old-fit point relative to LCDM; this effect is not solely the new CC covariance. The fitted models can move when CC is added, so none of the new-column values is a reoptimized minimum or a final AIC/DIC result. The audit does not explain differences to external papers with other priors or baryon treatments, nor different interaction kernels. These 14 points cannot exclude an error elsewhere in parameter space.

Next decisive step: bounded matched optimizations for current SN+BAO and current SN+BAO+CC, followed by posterior runs only if DIC/posterior shifts are needed. No new MCMC run started by this audit.
