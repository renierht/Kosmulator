# Three-case EOS corner figures

Each PDF/PNG overlays the three cases within one kernel. Uncoupled, Model 4 and Model 5 are complete. Model 2/3 await complete three-case chains. No LCDM fourth curve is added to the interacting figures.

All retained samples enter the density histograms, with original postprocessing burns (see provenance.json). Fixed parameters are reference lines, never artificial narrow probability distributions. The w_de versus w_dm panel has no joint contour because neither is free simultaneously. M_abs is marginalized, matching the reference figure. One-dimensional curves are peak-normalized for display.

To resolve the nearly singular Model 4 w_dm/delta ridge without a noisy Cartesian KDE, two-dimensional densities are smoothed on a whitened 360x360 histogram grid then transformed back. Gaussian bandwidth uses a fixed effective size of 10000 for consistent presentation, not a new ESS estimate. Density outside rectangular priors is masked, and contour masses are normalized on remaining support. This smoothing is a visualization choice, not a change to samples, priors, likelihood or tabulated credible intervals. It is not an exact correction for coupled physical boundaries. Display axes cover the union of 0.05–99.95 percentile ranges with padding and fixed/reference values; contour levels are computed on the full histogram domain. All draws, including tails beyond the displayed axes, contribute.

Reproduce using the Kosmulator_test Python environment: python Plots/EOS_case_comparisons/plot_eos_cases.py. No CLASS evaluations or sampling are performed. Source paths, hashes, burns, retained counts and numerical enclosed contour masses are recorded in provenance.json.
