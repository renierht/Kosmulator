# Reproducing the Kosmulator IDE SAIP 2026 proceedings analysis

Updated: 4 October 2026. **Validated development workflow; final release and DIC convention pending.**

This guide describes the existing-chain analysis of LCDM and three regimes of `NonLinear_IDE_2` using DESI DR2 BAO and Pantheon+ with the SH0ES anchor. It replaces the earlier checklist, which used N=1715 and conflated sampler steering settings with the retention settings of the final chains.

## 1. Version and scope

The CLASS/model-derived-r_d baseline is commit:

```text
1d1578fb9649c19b8ec7a455a1c1997fe048df49
```

Postprocessing was developed on `Kosmulator_IDE_v2_postprocessing` from that commit. The validated postprocessing changes are currently uncommitted. **Checking out the baseline alone does not reproduce the updated results.** Before publication, replace this development status with the final release commit/tag and record the dependency versions and CLASS build details.

The paper uses two model implementations, `LCDM_v` and `NonLinear_IDE_2`. The broader repository contains other IDE models, but the checks described here do not establish their paper-level validation. The IDE CLASS implementation used here supplies background and thermodynamics calculations; IDE CMB perturbation likelihoods are outside this reproduction.

## 2. Data and selected observation count

Use the observation tags `DESI_DR2` and `PantheonPS`. `PantheonPS` means Pantheon+ with SH0ES calibrators included, rather than Pantheon+ alone. SH0ES calibration is already part of this likelihood; do not add an independent SH0ES H0 prior when reproducing these chains.

The Pantheon selection is:

```python
mask = (zHD > 0.01) | (IS_CALIBRATOR > 0)
```

There are 1701 raw SN rows, 1657 retained SN entries, and 13 DESI BAO measurements. Thus the information criteria use **N=1670**, not 1715. The covariance must be sliced using the same selected indices as the SN data. N counts the selected measurements; correlations are represented in their covariance, not by substituting the raw table size.

The compared working copies had these identical SHA256 digests:

| File under `Observations/` | SHA256 |
|---|---|
| `PantheonP.dat` | `22269701ae91d365292b415d823439f1b4ed3fb3a918fe5760628d22877c25c1` |
| `PantheonP.cov` | `abf806d966485e64afdb359c87bffc0ecc00d05eff0a31ced66f247385df0fdc` |
| `DESI_DR2_synced.txt` | `cac0ba3c098b031f19e16a8fb8a641431a9bc343c61c38ead4bded84010b2b54` |

Before release, inventory and hash every additional DESI covariance/input file actually loaded. The three hashes above are not a complete data manifest. Provide original dataset references and download instructions, respecting their distribution terms.

## 3. Parameters, priors and regimes

The parameter order of the saved chains is:

- LCDM: `Omega_m, H_0, M_abs` (k=3).
- IDE: `Omega_m, w, delta, H_0, M_abs` (k=5).

The rectangular prior limits used to reconstruct the likelihood context are:

| Parameter | Lower | Upper |
|---|---:|---:|
| `Omega_m` | 0.1 | 0.5 |
| `H_0` | 60 | 90 |
| `M_abs` | -20.5 | -18 |
| `w` | -2 | -0.33 |
| `delta` | -1 | 1 |

Apply the model's mathematical and physical restrictions in addition to these limits. The regime switches are:

| Regime | `ALLOW_NEGATIVE_ENERGIES` | `ALLOW_BIG_RIP` | `ALLOW_DOOM_FACTOR_INSTABILITIES` |
|---|---|---|---|
| iw | True | True | True |
| +iw | False | True | True |
| Siw | True | True | False |

In particular, +iw is not restricted to w>-1. Siw uses w<-1 and can have negative delta. Preserve the implemented coupled restrictions; replacing them with guessed rectangular cuts changes the model support.

The fixed model-derived sound-horizon settings are:

```python
DERIVE_RD_WITH_MODEL_CLASS = True
RD_CLASS_OMEGA_B = 0.048
RD_CLASS_N_EFF = 3.044
RD_CLASS_SUM_MNU_EV = 0.06
RD_CLASS_N_NCDM = 3
```

Here Omega_b is a density fraction, not omega_b=Omega_b*h**2. The derived r_d is not an additional free parameter. Retain the baseline flat geometry and late-time analytic background assumptions; the CLASS calculation includes the radiation/neutrino treatment needed to compute r_d. Do not replace that calculation with the late-time radiation approximation or a generic LCDM/Eisenstein-Hu sound horizon.

## 4. Exact chains and retention

The development chain root is the sibling checkout's `Kosmulator/MCMC_Chains`. Every path below ends in the observation directory `DESI_DR2_PantheonP_SH0ES/`.

| Regime | Directory under chain root | Filename | Completed steps | Retained samples |
|---|---|---|---:|---:|
| LCDM | `RD_CLASS_DIAG_5K/LCDM_v` | `DESI_DR2+PantheonP_SH0ES.h5` | 4700 | 444000 |
| iw | `RD_CLASS_SEEDED_2K/NonLinear_IDE_2` | `DESI_DR2+PantheonP_SH0ES_FINAL_8500.h5` | 8500 | 900000 |
| +iw | `RD_CLASS_PLUS_IW_8K/NonLinear_IDE_2` | `DESI_DR2+PantheonP_SH0ES.h5` | 7300 | 756000 |
| Siw | `RD_CLASS_SIW_8K/NonLinear_IDE_2` | `DESI_DR2+PantheonP_SH0ES.h5` | 8200 | 864000 |

Each chain has 120 walkers. Reproduction of these validated summaries uses **discard=1000, thin=1**, followed by flattening. Apply burn-in once, along the step axis, before flattening. Do not discard a second time in the plotting or statistics routines. All retained samples were finite in the validated runs.

Do not substitute the earlier iw chain for the `FINAL_8500` chain. Directory names do not establish the completed chain length. The old guide's 100000-step target and burn=10000 are not the retention recipe validated for these saved outputs.

For fresh sampling, publish the actual original run/resume settings, initialisation, stopping rule and convergence diagnostics separately. Their complete provenance has not yet been assembled here. A successful postprocessing check does not establish chain convergence. Do not describe the test harness's nsteps=8500 override as the original sampling configuration for every model.

## 5. Statistical methodology

Use likelihood values aligned with the retained parameter samples. Saved likelihood blobs were used for the four validated chains. Their saved log_prob values matched the blobs in the earlier diagnostics. This equality is specific to these chains: a saved posterior generally requires subtracting its log prior before using it as a likelihood.

Let D(theta)=-2 log L(theta), with the same likelihood convention for every model, and D_bar=mean(D(theta)) over the retained samples. Arithmetic parameter means are taken from those same samples.

### Current DIC implementation: agreement pending

```text
D_at_mean = D(mean(theta))
p_D       = D_bar - D_at_mean
DIC_mean  = D_bar + p_D = 2*D_bar - D_at_mean
```

Christoffel's proposed likelihood-mode convention is:

```text
p_D_mode = D_bar - D_min
DIC_mode = D_bar + p_D_mode = 2*D_bar - D_min
```

These are distinct conventions, not interchangeable labels. The validation exports below use the posterior-mean convention. They have not yet been approved as the final submission convention. If the mode convention is adopted, update the implementation, p_D/DIC values, captions, interpretation and regression targets together. Do not clip p_D to k.

### Likelihood minima and other criteria

Use a polished likelihood maximum, rather than a posterior median, posterior mean or merely the best stored sample, for D_min and the likelihood-based criteria. The current search uses three interior starts with maxfev=1500, xatol=1e-6 and fatol=2e-5 for the -log L objective.

For +iw, a separately checked sequence approaches delta=0 from above at delta=1e-6, 1e-8 and 1e-10. The selected result is the validated terminal positive-delta limit; it does not evaluate exact delta=0. For Siw, the sequence fixes w=-1-epsilon for epsilon=1e-4, 1e-6 and 1e-8, approaching the strict-domain boundary from below. Report these as boundary-limited likelihood results rather than asserting an attained interior maximum. Keep k=5 for both regimes.

```text
AIC          = D_min + 2*k
AICc         = AIC + 2*k*(k+1)/(N-k-1)
BIC          = D_min + k*ln(N)
Reduced chi2 = D_min/(N-k)
Delta IC     = IC(model) - IC(LCDM)
```

Reduced chi2 is an exported descriptive statistic; do not treat it alone as a formal goodness-of-fit test. Retain full precision for calculations and exports, rounding only for presentation.

## 6. Validated numerical checkpoints

The following values come from `Kosmulator_four_model_validation.zip` produced on 4 October 2026. DIC and p_D use the current posterior-mean convention.

| Model | D_min | D_bar | D_at_mean | p_D | DIC_mean |
|---|---:|---:|---:|---:|---:|
| LCDM | 1472.7151995107 | 1475.7214474057 | 1472.7191391563 | 3.0023082494 | 1478.7237556551 |
| iwCDM | 1461.4166260371 | 1466.4140976961 | 1461.4258506095 | 4.9882470867 | 1471.4023447828 |
| +iwCDM | 1470.8504822138 | 1476.7357016816 | 1472.6461946763 | 4.0895070054 | 1480.8252086870 |
| SiwCDM | 1465.7444534970 | 1471.5344670239 | 1467.4332593797 | 4.1012076442 | 1475.6356746680 |

| Model | Delta AIC | Delta AICc | Delta DIC_mean | Delta BIC |
|---|---:|---:|---:|---:|
| LCDM | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| iwCDM | -7.298573 | -7.276922 | -7.321411 | +3.542584 |
| +iwCDM | +2.135283 | +2.156935 | +2.101453 | +12.976441 |
| SiwCDM | -2.970746 | -2.949094 | -3.088081 | +7.870412 |

Small changes below the validation tolerances can arise from numerical optimisation and CLASS evaluations. The validator checks |Delta D_min|<=5e-4 and |Delta DIC_mean|<=1e-4 against recovered-method reference values, together with optimizer success and correct N/k. Preserve the diagnostic sequences, not just a PASS label.

## 7. Commands: current development arrangement

Run from the postprocessing repository root in the working scientific environment:

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -p 'test*.py' -v
MPLBACKEND=Agg python -u validate_paper_postprocessing.py \
  --chain-root /absolute/path/to/MCMC_Chains
```

The validator uses the repository-local `reproducibility/SAIP2026_IDE/scripts/paper_context.py`. That module preserves the historical regime and configuration/covariance setup; optimisation remains in the production `Model_comparison` module. No sibling source checkout is required. Chains default to `MCMC_Chains/` under the repository; use `--chain-root /absolute/path/to/MCMC_Chains` for an external archive. The script writes CSV, JSON and LaTeX comparison tables and an information-criteria plot into a temporary directory, printed as `OUTPUT`. It permits fresh likelihood evaluations with existing CLASS binaries, but prohibits rebuilding and does not run MCMC or alter chains.

Existing-chain loading, figures and exports have also been checked through the real `Kosmulator.main()` and `MCMC_setup.main()` using `check_kosmulator_entrypoint.py`:

```bash
for regime in iw plus_iw siw; do
  MPLBACKEND=Agg python -u /path/to/check_kosmulator_entrypoint.py \
    --repo "$PWD" --regime "$regime" --copy-to /existing/results/directory
 done
```

That checker is currently a separately supplied engineering script. It uses runtime paper settings, read-only chain routing, disables polishing, and defaults to 80 derived-r_d samples for speed. It verifies exact sample-array agreement after retention. Its figures and unpolished criteria are engineering outputs, not final Table 5 or Overleaf assets. It exercises the public main routine in serial mode, not the `__main__` multiprocessing startup block.

**The untouched `python Kosmulator.py` defaults are not a paper reproduction command.** They select all eight IDE models, an additional CC dataset, burn=10000 and disabled model-derived CLASS r_d. Use the explicit project configuration and launcher instead:

```bash
MPLBACKEND=Agg python -u reproducibility/SAIP2026_IDE/scripts/reproduce.py \
  --chain-root /absolute/path/to/MCMC_Chains \
  --regime all --copy-to /existing/results/directory
```

The launcher reads `configs/paper.json`, then calls the normal `Kosmulator.main()` with supported optional `workflow_options`. Setup loads the configured files through the production loader's read-only `load_only` branch, then uses the normal plotting and statistical routines. Missing chains abort rather than trigger sampling. Resume, overwrite and parallel execution are rejected for this reproduction mode. The launcher guards existing CLASS binaries against rebuilding without modifying the frozen CLASS implementation.

Default outputs use polishing and 6000 derived-r_d evaluations per model/pair. Each regime is processed with an LCDM reference, and `combined_tables` collects the requested comparisons. Figures, tables, environment details, configuration, source hashes and a run log are placed in a fresh output directory and ZIP. `--output` accepts a new, nonexistent directory; `--copy-to` copies the ZIP into an existing directory. DIC remains pending consensus even for the full output.

For a faster engineering integration check:

```bash
MPLBACKEND=Agg python -u reproducibility/SAIP2026_IDE/scripts/reproduce.py \
  --chain-root /absolute/path/to/MCMC_Chains \
  --regime all --derived-rd-samples 80 --no-polish \
  --copy-to /existing/results/directory
```

This explicitly produces unpolished criteria and small-subset r_d uncertainties; do not use it for publication. No runtime replacement of the chain loader or plotting routine is used by this launcher. It runs serially and does not exercise multiprocessing sampling startup. The validated numerical checkpoints remain applicable, but the new launcher needs its own end-to-end check in the CLASS environment.

## 8. Posterior summaries and figures

Report sampled-parameter medians with 16th/84th-percentile intervals from the retained arrays. These summaries are distinct from the arithmetic parameter means used by the current DIC definition and from polished likelihood parameters.

For derived r_d, evaluate the matching model CLASS calculation on a reproducible subset of 6000 retained samples, using `derived_rd_seed=20260929` under the current plotting implementation. Report the resulting 16th/50th/84th percentiles to 0.1 Mpc. The 80-sample engineering check is insufficient for final uncertainty reporting. Subset stability checks compared 1000, 3000 and 6000 evaluations; they assess derived-summary sensitivity, not chain convergence.

The current best-fit plotting diagnostics also print r_d at parameter medians and S at parameter medians. These are plug-in summaries, not posterior medians of the derived quantities. The posterior-r_d annotation must be distinguished from the single r_d used for the plotted median-parameter prediction.

The production plotting workflow generates corner plots, data/model comparison plots, parameter tables and statistical exports. Some manuscript figures have separate paper-specific scripts. Before release, include and name the exact final H0 whisker and grouped information-criteria scripts, their inputs and export commands, and state whether any manuscript panels were assembled manually. Do not assume the engineering plot matches the final manuscript figure simply because it represents the same numbers.

## 9. Release checklist and future project layout

Suggested repository layout:

```text
reproducibility/
  README.md
  SAIP2026_IDE/
    REPRODUCE.md
    configs/
    scripts/
    manifests/
    reference_results/
```

The patch supplies this project guide, its local context script and the top-level index. The patch additionally supplies `configs/paper.json`, `scripts/reproduce.py` and numerical `reference_results`. A complete input/chain manifest is still a release requirement. Keep each future project's inputs, configuration, version and expected outputs separate. The top-level README should link to each project guide. Large chains can live in a versioned external archive with checksums and a persistent identifier rather than in Git.

Before advertising this project as independently reproducible:

- Pin the final release commit/tag, Python/package environment, CLASS source/build signature and platform requirements.
- The sibling source-helper dependency has been replaced by the local paper context. Validate the supplied production configuration/runner end to end in the CLASS environment.
- Publish a complete observation/covariance manifest and chain checksums, plus download locations or explicit access limitations. Previous size/mtime safety checks are not cryptographic chain checksums.
- Record fresh-sampling commands, initialisation/seed information, resume history and convergence diagnostics if claiming reproduction from sampling. If the original seed was not recorded, say so; new chains need not be bit-for-bit identical.
- Agree the DIC convention with the statistics coauthor and synchronise code, reference exports and manuscript.
- The optional data-point model-overlay fix from upstream commit `310034aded9f64d5b78ac01631af938dcf3d26c0` is included. Check existing non-IDE functionality affected by shared changes.
- Regenerate final figures with the publication settings, including 6000 derived-r_d samples, and include their exact generation scripts.
- Run the documented procedure from a clean checkout without private paths or unpublished helper files. Archive its full-precision outputs and diagnostics.

Until these items are completed, the results above document a validated development analysis on the existing chains, with a clear route to a reproducible release.
