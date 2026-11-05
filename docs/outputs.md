# Outputs and statistics

Every run writes its results into three folders, each with a subfolder named after the run's `--output_suffix`:

```text
MCMC_Chains/<suffix>/
├── run_settings.json                       # the run's settings (read by --peek)
└── <model>/<group>/<group>.h5              # the chain (HDF5); <group>_zeus.h5 for zeus

Plots/Saved_Plots/<suffix>/<model>/
├── corner_plots/corner_<model>.png         # posteriors of every group of the model
├── best_fits/<group>/bestfit.png           # the model against each data set
└── auto_corr/<group>.png                   # the convergence rule, redrawn at every check

Statistical_analysis_tables/<suffix>/
├── citations.md, citations.bib             # what to cite for this run
└── <model>/
    ├── stats_summary.txt                   # chi^2, AIC, BIC, AICc, DIC, WAIC and differences
    ├── interpretations_summary.txt         # the same in words, against the reference model
    ├── convergence_summary.txt             # tau, ESS, split-Rhat, acceptance per group
    └── aligned_table_pretty.txt            # parameter table (median and 68% interval, LaTeX)
```

A group's folder name joins its tags with underscores (`DESI_DR2_PantheonP`); its label in tables and plots joins them with `+` (`DESI_DR2+PantheonP`).

## Plots

**Corner plots** show the marginalised posteriors of all groups of a model on one figure, so that the constraints of different data combinations can be compared directly. With `--plot_table` the posterior summary table is drawn on the figure. Nuisance parameters are left out unless `corner_show_nuisance = True`.

**Best-fit plots** show, for every data set of a group, the data with the model's prediction at the posterior medians, a shaded pointwise 68% band from posterior draws, and the residuals below.

**Convergence plots** (`auto_corr/`) show, at every check, $\tau_\mathrm{max}$, the effective sample size and split-$\hat R$ against the thresholds of the convergence rule. They are redrawn while the run goes on, so they can be watched.

## The statistics

For every model and group, Kosmulator first polishes the best sample of the chain into the best fit (maximum posterior) and then reports:

| Quantity | Definition |
|---|---|
| $\chi^2$, $\ln\mathcal{L}$ | at the best fit; $\ln\mathcal{L} = -\chi^2/2$ |
| $\chi^2_\nu$ | $\chi^2/(N - k)$, with $N$ data points and $k$ fitted parameters (including analytically fitted supernova offsets) |
| AIC | $2k - 2\ln\mathcal{L}$ |
| BIC | $k \ln N - 2\ln\mathcal{L}$ |
| AICc | AIC $+\, 2k(k+1)/(N - k - 1)$ |
| DIC | $\hat D + 2p_D$, with $\hat D$ the $\chi^2$ at the best fit and $p_D = \bar D - \hat D$ (Spiegelhalter et al.) |
| WAIC | from the pointwise log-likelihoods of 1000 posterior draws (n/a for groups with Planck data unless `waic_include_cmb = True`) |

For every model other than the reference model each quantity is also given as a difference, $\Delta X = X_\mathrm{model} - X_\mathrm{reference}$: **negative values favour your model**. The $\Delta\chi^2$ is turned into a significance as in DESI DR2: the probability of a $\chi^2$ improvement at least as large, for a $\chi^2$ distribution with as many degrees of freedom as the model has extra parameters, written as a two-sided Gaussian $N\sigma$.

::::{admonition} CMB groups
:class: note
For groups with CMB data, $\Delta\chi^2$ and the significance use the $\chi^2$ including the priors on the CMB nuisance parameters ($-2\Delta\ln$ posterior at the maximum, as DESI DR2 does); $\chi^2_\nu$, AIC, BIC and DIC are computed from the data alone.
::::

::::{dropdown} How the interpretations are worded
`interpretations_summary.txt` turns each difference into words, using $|\Delta|$ for the strength and its sign for the direction ("in favour of" or "against" the model):

| Criterion | Indistinguishable | Slight / Weak | Positive / Moderate | Strong |
|---|---|---|---|---|
| $\Delta$AIC, $\Delta$AICc, $\Delta$DIC, $\Delta$WAIC | < 2 | 2 - 4 (Slight) | 4 - 7 (Positive) | $\geq$ 7 |
| $\Delta$BIC | < 2 | 2 - 6 (Weak) | 6 - 10 (Moderate) | $\geq$ 10 |

and the significance as inconclusive ($\leq 1\sigma$), slight (1 - 2$\sigma$), considerable (2 - 3$\sigma$), strong (3 - 4$\sigma$), very strong (4 - 5$\sigma$) or overwhelming ($> 5\sigma$). It also comments on $\chi^2_\nu$, for example when supernova covariances with conservative systematics make it well below 1.
::::

Every statistics file ends with a `Settings:` line that lists the run's physics switches and marks any that differ from the defaults with `*`, so that a table can always be traced back to how it was made.

## Convergence summary

`convergence_summary.txt` lists, for every group, the engine, the number of steps and the burn-in, $\tau_\mathrm{max}$ (and the parameter it belongs to), $N_\mathrm{post}/\tau_\mathrm{max}$, the effective sample size, the largest split-$\hat R$, the acceptance fraction (emcee) and whether the convergence rule was met. A chain that reached `nsteps` without meeting it is flagged, and the run log says how many more steps are needed; continue it with `--resume` and a larger `nsteps` (see [Tutorial 1](tutorials/desi_w0wa.md#4-a-chain-that-has-not-converged)).

## Citations

Every run writes `citations.md` and `citations.bib`: the papers to cite for exactly the data sets, samplers and codes that run used, with their BibTeX. Its path is printed at the end of the run.
