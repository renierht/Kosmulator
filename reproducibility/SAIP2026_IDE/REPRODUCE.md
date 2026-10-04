# Reproduce the SAIP 2026 Kosmulator IDE analysis

This guide produces Kosmulator figures and statistical tables from the four saved chains used for the SAIP analysis. It loads those chains without running new MCMC sampling.

## 1. Set up Kosmulator

Use the **`Kosmulator_IDE_v2`** branch and follow the [repository installation instructions](../../README.md).

You will need:

- A working Kosmulator Python environment.
- The observation files in `Observations/`, including the Pantheon+ and DESI DR2 data and covariance files.
- Prepared, model-specific CLASS binaries for `LCDM_v` and `NonLinear_IDE_2`. The reproduction command uses existing binaries; it does not build CLASS.
- The four saved chains listed below. These large files are not included in this repository; request the analysis chains from the authors if you do not have them.

## 2. Use these analysis settings

The launcher applies these settings automatically from [`configs/paper.json`](configs/paper.json). You do not need to edit `Kosmulator.py` to use the saved-chain workflow.

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
| Execution | Serial, existing-chain loading |

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

The model's regime restrictions also apply. The launcher sets the switches in `User_defined_modules.py` as follows:

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

## 3. Put the saved chains in one directory

Set `--chain-root` to your `MCMC_Chains` directory. Keep the following layout:

```text
MCMC_Chains/
├── RD_CLASS_DIAG_5K/LCDM_v/DESI_DR2_PantheonP_SH0ES/
│   └── DESI_DR2+PantheonP_SH0ES.h5
├── RD_CLASS_SEEDED_2K/NonLinear_IDE_2/DESI_DR2_PantheonP_SH0ES/
│   └── DESI_DR2+PantheonP_SH0ES_FINAL_8500.h5
├── RD_CLASS_PLUS_IW_8K/NonLinear_IDE_2/DESI_DR2_PantheonP_SH0ES/
│   └── DESI_DR2+PantheonP_SH0ES.h5
└── RD_CLASS_SIW_8K/NonLinear_IDE_2/DESI_DR2_PantheonP_SH0ES/
    └── DESI_DR2+PantheonP_SH0ES.h5
```

Use the `FINAL_8500` file for iw. The completed chain lengths are 4700, 8500, 7300 and 8200 steps for LCDM, iw, +iw and Siw respectively.

## 4. Run the analysis

Activate your Kosmulator environment, open a terminal in the repository root, and run:

```bash
MPLBACKEND=Agg python -u reproducibility/SAIP2026_IDE/scripts/reproduce.py \
  --chain-root /absolute/path/to/MCMC_Chains \
  --regime all \
  --output /absolute/path/to/SAIP2026_results
```

Replace both paths with your own locations. The output directory must **not already exist**.

To run just one IDE comparison, replace `all` with `iw`, `plus_iw` or `siw`. LCDM is included as the reference in each comparison.

The default 6000 CLASS evaluations for each derived `r_d` summary can take substantial time. For a quicker preview, add `--derived-rd-samples 1000`; use the default 6000 for the full run.

## 5. Find the figures and tables

Your chosen output directory contains:

- `iw/plots/`, `plus_iw/plots/` and `siw/plots/`: corner plots and data/model comparison figures, inside run- and model-specific subdirectories.
- `iw/tables/`, `plus_iw/tables/` and `siw/tables/`: statistical exports for each comparison.
- `combined_tables/`: the combined four-model statistical results.
- `workflow.log` and `workflow_report.json`: the run log and completion report.

A ZIP with the same name is created next to the output directory, for example `SAIP2026_results.zip`. The terminal prints both locations when the run finishes.

Derived `r_d` appears in the posterior tables and the BAO/DESI plot annotation; it is not a corner-plot axis. The workflow produces the standard Kosmulator outputs. Paper-specific multi-panel layouts or additional manuscript figures may require separate assembly.
