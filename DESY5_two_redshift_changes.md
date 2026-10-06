# DESY5: two-redshift distance convention (change record)

Date: 2026-10-06
Companion file: `DESY5_two_redshift_changes.patch` (exact unified diff of every line changed)

## 1. Purpose

Kosmulator's DESY5 results did not reproduce the DESI DR2 paper. DESI + DESY5, $w_0w_a$CDM against LCDM, gave $\Delta\chi^2_{\rm MAP} = -22.2$ with the original code, against -13.6 in the paper.

The cause was how redshift entered the supernova distance. Kosmulator used one redshift column for both the comoving-distance integral and the $(1+z)$ prefactor. The DES-SN5YR likelihood in Cobaya uses two:

$$D_L = (1+z_{\rm HEL})\,D_M(z_{\rm HD}), \qquad \mu_{\rm th} = 5\log_{10}(D_L/{\rm Mpc}) + 25 .$$

Here $z_{\rm HD}$ is the Hubble-diagram redshift (CMB-frame, peculiar-velocity corrected) and $z_{\rm HEL}$ is the heliocentric redshift. In Cobaya's `desy5.py`, the file column `zhd` is mapped into the variable called `zcmb`; the file's own `zCMB` column is never used.

Evidence (DESI + DESY5, $\Delta\chi^2_{\rm MAP}$):

| redshift treatment | $\Delta\chi^2$ |
|---|---|
| zCMB for everything (original Kosmulator behaviour) | -22.2 |
| zHD for everything | -12.3 |
| zHEL for everything | -29.6 |
| zHD in the integral, $(1+z_{\rm HEL})$ prefactor | -13.4 |
| DESI paper (read from a screenshot, not verified against the source) | -13.6 |

## 2. What was changed and where

Five files. Only DESY5 changes numerically; every other dataset takes the same code path as before.

| file | change | why |
|---|---|---|
| `Kosmulator_main/Config.py` (`load_named_sne_with_zcmb`, CSV branch) | `zhd` is now the first redshift candidate, so `redshift` holds zHD for DESY5. The `zhel` column is also read and returned as `z_hel`, cleaned with the same row mask. The error message and docstring were updated. | DESY5 needs zHD for the integral and zHEL for the prefactor, so the loader must supply both. |
| `Kosmulator_main/utils.py` (new function `sn_luminosity_distance`, after `Comoving_distance_vectorized`) | Returns `d_c * (1 + z_fac)`, where `z_fac` is `z_hel` if supplied and `z` otherwise. | One place holds the convention, so the sampler, the MAP polish and WAIC cannot drift apart. |
| `Kosmulator_main/Kosmulator_MCMC.py` (`model_likelihood`, SNe branch) | `y_dl = d_c * (1.0 + z)` became `utils.sn_luminosity_distance(d_c, z, obs_data.get("z_hel"))`. | Sampling likelihood. |
| `Kosmulator_main/Post_processing.py` (`statistical_analysis`, nested `_compute_chi2_total`, SNe branch) | `comoving_distances * (1 + redshift)` became `U.sn_luminosity_distance(comoving_distances, redshift, obs_data.get("z_hel"))`. | Statistics step: MAP polish, AIC, BIC, AICc, DIC. |
| `Kosmulator_main/Statistical_packages.py` (`build_log_like_matrix`, SNe branch) | `d_c * (1.0 + z)` became `utils.sn_luminosity_distance(d_c, z, obs_data.get("z_hel"))`. | WAIC log-likelihood matrix. |

Each call site keeps its own comoving-distance function (the sampler uses `utils.Comoving_distance_vectorized`, `Post_processing` uses the `UDM` wrapper), so the surrounding numerics are untouched.

### Deliberately not changed
- The function name `load_named_sne_with_zcmb`, to avoid editing its two call sites in `load_all_data`. The name is now misleading; only the docstring was updated.
- The Pantheon+ branches, which use the `zHD` column with a single redshift.
- The smooth model curve in `Plots/Plot_functions.py` (around line 1256), which is drawn on a redshift grid and not per supernova.
- Union3 (whitespace branch of the loader), JLA and Pantheon. None of them provides `z_hel`, so the helper reduces to the original `d_c * (1 + z)`.

## 3. Validation (run from the scratchpad, through the project's own functions)

1. **Loader:** DESY5 returns `redshift` equal to the file's zHD and `z_hel` equal to zHEL, both with N = 1829, and the covariance is still 1829 x 1829. Union3 has no `z_hel` key.
2. **Helper:** with `z_hel=None` the result is bit-for-bit `d_c * (1.0 + z)`.
3. **Sampling likelihood** (`model_likelihood`) at a fixed $w_0w_a$CDM point, DESY5 block: $\chi^2 = 1637.411376$ with `z_hel`, identical to an independent re-implementation of the formula (difference 0). With `z_hel` removed it gives 1638.246514, again identical to the independent single-redshift formula.
4. **WAIC builder** (`build_log_like_matrix`) gives the same $\chi^2$ (1637.411376) at that point.
5. **End to end** through the sampling likelihood (DESI DR2 + DESY5, Nelder-Mead from several starts): $\chi^2_{\min}$ = 1658.463 (LCDM) and 1645.046 ($w_0w_a$CDM), so $\Delta\chi^2 = -13.42$. This matches my earlier scratch calculation (-13.42) and is close to the paper's -13.6.

All five edited files compile (`py_compile`).

### Not tested
- The edited line inside `Post_processing._compute_chi2_total` was not executed. It is a nested function inside `statistical_analysis`, which needs a full run. It uses the same helper, and the diff was reviewed by hand.
- No full Kosmulator run was done (sampling, plots, statistics files).
- The remaining 0.2 difference from -13.6 is unexplained. I have not checked the DESI paper's exact data version or setup.
- The convention comes from the Cobaya DES-SN5YR implementation (`base_classes/sn.py` and `sn/desy5.py`, master branch at the time), not from a paper equation I could read. Please confirm with your supervisor, and check that `Observations/DESY5.dat` (1829 SNe) is the release the DESI paper used.

## 4. What you must do next

1. **Rerun all DESY5 chains with `--overwrite`.** The likelihood has changed, so existing DESY5 chains, plots and statistics tables are no longer consistent with the code. Do not use `--resume` on old DESY5 chains.
2. **Check line 187 of `Config.py` is as intended.** Earlier trials put `zhel` first in the candidate list. It now reads `("zhd", "zcmb", "z_cmb", "z")`.
3. After a full run, check that DESI + DESY5 gives $\Delta\chi^2_{\rm MAP}$ close to -13.4.
4. Commit the change (see below). Your working tree has other uncommitted edits unrelated to this record.

## 5. Related problems found but NOT fixed here

These affect DIC, WAIC and sigma, not the MAP-based $\Delta\chi^2$, AIC, BIC and AICc:
- The stored `log_like` is in walker-major order, but it is paired with step-major samples (`Kosmulator_MCMC.py` around lines 1037 - 1049, `Post_processing.py` around `calculate_asymmetric_from_samples`). DIC's mean is unaffected, but the MAP starting candidates take the wrong samples.
- Unconverged Zeus walkers in short chains (for example 32 x 2500 with burn-in 250) inflate DIC, $p_D$ and WAIC. Use a longer burn-in and check per-walker behaviour.
- A resumed Zeus run stores a longer chain than its `log_like`; the length check in `utils.load_or_run_chain` then silently drops `log_like`, giving NaN DIC, $p_D$ and WAIC.
- `significance()` uses the rounded difference in DIC effective parameters as its degrees of freedom, which gives wrong or NaN sigma for contaminated or negative $\Delta p_D$.
- The WAIC subsample is unseeded (`np.random.choice`), so WAIC is not reproducible.

## 6. Reverting

From `/home/user/Kosmulator`:

```bash
patch -R -p1 < DESY5_two_redshift_changes.patch
```

This was dry-run checked and applies cleanly to the current files. If you later edit the same lines, the patch may need manual adjustment. Committing the change to git first gives the cleanest way to revert.
