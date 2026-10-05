# Reproduce the SAIP 2026 Kosmulator IDE analysis

Use this guide to generate your own chains with the paper's model and likelihood settings, then produce Kosmulator figures and statistical tables. Independent runs should give statistically consistent constraints; they will not reproduce every reported digit.

## 1. Set up Kosmulator

Use the **Kosmulator_IDE_v2** branch and follow the [installation instructions](../../README.md). Run the commands below from the repository root in your Kosmulator environment.

Keep the Pantheon+ and DESI DR2 observation and covariance files in `Observations/`. CLASS must be buildable in your environment, or the matching binaries for `LCDM_v` and `NonLinear_IDE_2` must already be prepared. The normal sampling workflow prepares CLASS when needed. This analysis uses background datasets only, without an IDE CMB likelihood.

## 2. Paper settings

The launchers apply these settings automatically from [`configs/paper.json`](configs/paper.json). You do not need to edit the defaults in `Kosmulator.py`.

| Setting | Value |
|---|---|
| Models | `LCDM_v` and `NonLinear_IDE_2` |
| IDE regimes | iw, +iw and Siw, each compared with LCDM |
| Observation tags | `DESI_DR2`, `PantheonPS` |
| Walkers | 120 |
| Burn-in | First 1000 steps per walker |
| Thinning | 1 |
| Sound horizon | Model-derived CLASS `r_d`; not a free parameter |
| Derived `r_d` summary | 6000 posterior samples; seed 20260929 |
| Likelihood polishing | Enabled |
| Execution | Serial emcee |

`PantheonPS` includes the SH0ES calibrators. Do not add a separate SH0ES H₀ prior. The selected data contain 1657 Pantheon+ entries and 13 DESI measurements, giving **N = 1670**.

**Free parameters and priors**

LCDM uses `Omega_m, H_0, M_abs`. IDE uses `Omega_m, w, delta, H_0, M_abs`, in that order.

| Parameter | Prior range |
|---|---|
| Ωₘ (`Omega_m`) | 0.1 to 0.5 |
| H₀ (`H_0`) | 60 to 90 km s⁻¹ Mpc⁻¹ |
| M (`M_abs`) | −20.5 to −18 |
| w | −2 to −0.33 |
| δ (`delta`) | −1 to 1 |

The model's regime restrictions also apply. The launchers set the switches in `User_defined_modules.py` as follows:

| Regime | `ALLOW_NEGATIVE_ENERGIES` | `ALLOW_BIG_RIP` | `ALLOW_DOOM_FACTOR_INSTABILITIES` |
|---|---|---|---|
| iw | True | True | True |
| +iw | False | True | True |
| Siw | True | True | False |

**CLASS settings**

The launcher applies these values from the project configuration:

```python
DERIVE_RD_WITH_MODEL_CLASS = True
RD_CLASS_OMEGA_B = 0.048
RD_CLASS_N_EFF = 3.044
RD_CLASS_SUM_MNU_EV = 0.06
RD_CLASS_N_NCDM = 3
```


## 3. Generate your own chains and analyse them

Run one command per IDE regime; each includes LCDM as the reference:

```bash
python -u reproducibility/SAIP2026_IDE/scripts/run_analysis.py --regime iw --run-name MySAIP
python -u reproducibility/SAIP2026_IDE/scripts/run_analysis.py --regime plus_iw --run-name MySAIP
python -u reproducibility/SAIP2026_IDE/scripts/run_analysis.py --regime siw --run-name MySAIP
```

The launcher sets the paper priors, CLASS settings, 120 walkers and 1000-step burn-in. It runs the normal Kosmulator entry point, including sampling, plots and statistical analysis with likelihood polishing.

Use a new run name for a fresh experiment. Reusing a name invokes Kosmulator's normal existing-chain handling; it does not guarantee fresh sampling. To continue an incomplete run, repeat its command with `--resume`. The launcher does not offer an overwrite option.

The maximum is 100000 steps per walker, with the configured autocorrelation-change setting 0.01. Change the ceiling with `--max-steps`. These are settings for independent runs, not instructions to duplicate the original stopping history. Inspect convergence and posterior stability for your own chains before interpreting the results; reaching a maximum length alone does not establish convergence.

Add `--plan` to print settings without sampling or invoking CLASS. For a quicker preview of the derived sound-horizon summary, use `--derived-rd-samples 1000`; the default is 6000. This setting affects the posterior summary of derived `r_d`, not the model-specific sound horizon evaluated in the sampling likelihood.

## 4. Analyse your saved chains separately

Prepare a small JSON file, for example `my_chains.json`, containing the two chains for one regime:

```json
{
  "LCDM_v": "/absolute/path/to/my_LCDM_chain.h5",
  "NonLinear_IDE_2": "/absolute/path/to/my_iw_chain.h5"
}
```

Then run:

```bash
python -u reproducibility/SAIP2026_IDE/scripts/run_analysis.py \
  --mode analyse --regime iw --run-name MySAIP_analysis --chains my_chains.json
```

Choose the regime matching the chain's sampling restrictions. Chains must use the same likelihood, CLASS settings, parameter order and priors specified above. Use the corresponding JSON and regime for +iw or Siw. Relative chain paths are resolved from the JSON file's directory.

Analysis mode loads existing chains without starting or resuming sampling. It does not require the original filenames, completed lengths or retained sample counts. It applies the 1000-step burn-in once when loading the full saved backend. Chains need more than 1000 completed steps. Independent sampling and convergence assessment remain your responsibility.

## 5. Find the outputs

Chains are saved under `MCMC_Chains/`, and standard figures under `Plots/Saved_Plots/`, grouped by run suffix and model. The terminal identifies the chain locations. Each suffix includes the selected regime, for example `MySAIP_iw`. Statistical export locations are reported by Kosmulator.

Derived `r_d` appears in posterior tables and the BAO/DESI annotation, not as a corner-plot axis. Results are exported in JSON, CSV and LaTeX. The statistical implementation uses posterior-mean DIC. Paper-specific combined figure layouts may require separate assembly.

Record the code commit, environment, inputs and convergence diagnostics with your results.

## 6. Optional: replay the original saved-chain results

The existing `scripts/reproduce.py` launcher is for the original four analysis chains. It checks their recorded lengths and retention recipe against `configs/paper.json` and `reference_results/`. It is not the launcher for independently generated chains.

If you have those original files, place them under a common `MCMC_Chains` root using the paths in `configs/paper.json`, then run:

```bash
MPLBACKEND=Agg python -u reproducibility/SAIP2026_IDE/scripts/reproduce.py \
  --chain-root /absolute/path/to/MCMC_Chains --regime all \
  --output /absolute/path/to/new_SAIP2026_results
```

The output directory must not exist. Original chains are not included in GitHub; contact the authors for access. This optional launcher uses prepared CLASS binaries and prohibits rebuilding.

Its output directory contains per-regime plots and tables, `combined_tables/`, and a workflow log and report. A ZIP is created beside that directory.

