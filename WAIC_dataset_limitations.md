# WAIC and Model-Comparison Methods

## Short recommendation

WAIC is not the best first addition to Kosmulator. Ordinary pointwise WAIC
requires a log-likelihood contribution for each independent observation and
for every posterior draw. The current likelihood API generally returns only a
single total likelihood or chi-squared value for a dataset.

The most useful broadly applicable companion to DIC is **posterior predictive
checking**: generate observables from posterior draws and compare them with
the measured data, residuals, chi-squared contributions, or summary
statistics. It works with diagonal, correlated, and black-box likelihoods as
long as the model can generate predictions.

For scalar model-comparison tables, retain:

- **AIC/AICc**: based on the maximised total likelihood.
- **BIC**: based on the maximised total likelihood and an effective data-vector
  size.
- **DIC**: based on posterior deviance, with care for non-Gaussian or
  multimodal posteriors.

PSIS-LOO is a strong alternative to WAIC, but it has essentially the same
pointwise log-likelihood requirement. It is not a solution for the current
CLIK or covariance limitations by itself.

## Dataset-by-dataset WAIC assessment

| Dataset | Current likelihood structure | Ordinary pointwise WAIC? | Reason |
|---|---|---:|---|
| `CC` | Diagonal Gaussian errors through `Calc_chi` | Yes | Each measurement has an independent residual and a natural log-likelihood contribution. |
| `OHD` | Diagonal Gaussian errors through `Calc_chi` | Yes | Same as `CC`, provided the supplied errors are independent. |
| `f` | Sum of squared residuals with individual errors | Yes | Each growth measurement can normally be represented as one likelihood term. |
| `f_sigma_8` | Sum of squared residuals with individual errors | Yes | Same qualification as `f`: correlations must not be hidden in the error array. |
| `BBN_DH` | Independent systems or one weighted mean | Conditional | Separate systems can be pointwise; a weighted mean is only one observation. Asymmetric errors must use the same piecewise likelihood used by the fit. |
| Generic SNe with diagonal errors | Independent Gaussian errors | Yes | Only applies to configurations without a full covariance matrix. |
| `PantheonP` / `PantheonPS` | Full covariance, evaluated with a Cholesky solve | Not ordinary pointwise | The residuals are correlated. Assigning each raw supernova its own independent likelihood term is incorrect. A block-level likelihood is possible. |
| Union3-style SNe | Full inverse covariance when available | Not ordinary pointwise | The quadratic form couples all supernova residuals through the covariance matrix. Diagonal fallback data can support pointwise WAIC. |
| `BAO` | One 12-element vector with a 12x12 covariance matrix | Not ordinary pointwise | Correlated BAO observables are evaluated jointly. Use one BAO block or a carefully defined independent-block decomposition. |
| `DESI_DR1` / `DESI_DR2` | Mixed observable vector with `cov` or `inv_cov` | Not ordinary pointwise | The covariance matrix couples entries, including measurements at related redshifts. Use dataset-level or scientifically justified block-level contributions. |
| `CMB_hil` | Planck CLIK joint TT/TE/EE likelihood | No with current API | CLIK returns a joint black-box likelihood, not a documented pointwise decomposition. Multipoles cannot safely be treated as independent observations. |
| `CMB_hil_TT` | Planck CLIK joint TT likelihood | No with current API | Same black-box limitation; a total likelihood is available, but not valid pointwise terms. |
| `CMB_lowl` | Planck CLIK low-ell likelihood | No with current API | The likelihood includes joint treatment and nuisance/mode structure that the current wrapper does not expose pointwise. |
| `CMB_lensing` | Planck lensing CLIK likelihood | No with current API | The wrapper exposes a total lensing likelihood only; lensing bins are not automatically independent pointwise observations. |

## What to do in practice

For the current project, use the following priority:

1. Keep AIC, AICc, BIC, and DIC for the existing global statistics table.
2. Add posterior predictive checks for every dataset.
3. Add pointwise WAIC later for `CC`, `OHD`, `f`, `f_sigma_8`, independent BBN
   systems, and diagonal-error supernova configurations.
4. If WAIC is needed for covariance-based data, define likelihood blocks
   explicitly and label the result as block-level WAIC rather than ordinary
   pointwise WAIC.

## Important interpretation caveat

For correlated data, a multivariate Gaussian likelihood can be algebraically
factorised into conditional terms, but the resulting terms depend on the
ordering of the data vector. That is not the same as a unique observation-wise
decomposition. Therefore, an arbitrary split of a correlated chi-squared into
one term per raw datum should not be presented as standard pointwise WAIC.
