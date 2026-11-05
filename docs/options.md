# Settings and command-line flags

Every option of Kosmulator in one place: the switches in **`Kosmulator.py`**, which apply to every model and group of a run, and the **command-line flags**. The models, data groups, sampler settings and priors are explained in [Configuring a run](running.md#configuring-a-run); the data set tags are on [Data set tags](tags.md).

## Settings in Kosmulator.py

### Physics and likelihood switches

These switches apply to every model and group of the run. The defaults are the recommended choices.

| Setting | Default | What it does |
|---|---|---|
| `sample_H0_uncalibrated_SNe` | `False` | Supernovae without BAO or an $H_0$ anchor (CC, OHD, SH0ES, CMB) do not constrain $H_0$: it cancels against the marginalised magnitude offset. `False`: $H_0$ is not sampled. `True`: it is sampled anyway (its posterior is then the prior). It is not counted in the number of parameters either way. |
| `neutrinos`, `nu_mass_eV`, `nu_N_eff` | `"massive"`, `0.06`, `3.044` | One massive neutrino of 0.06 eV plus massless ones, $N_\mathrm{eff} = 3.044$ in total, as in DESI DR2 (Planck 2018: 3.046). `"massless"`: all massless. Used in CLASS, in the late-time radiation and in BBN. |
| `rd_background` | `"model"` | $r_d$ of BAO groups calibrated by BBN or the CMB. `"model"`: CLASS with the model's own $E(z)$. `"lcdm"`: CLASS's flat-ΛCDM $r_d$ for every model. |
| `growth_model` | `"gamma"` | Growth data (`f`, `f_sigma_8`). `"gamma"`: $f = \Omega_m(z)^\gamma$ with $\gamma$ sampled. `"geff"`: the model's own linear growth with $G_\mathrm{eff}/G$ from `<model>_Geff(z, p)`. |
| `cmb_nuisance` | `"baseline"` | Planck high-$\ell$ nuisances. `"baseline"`: Planck 2018's 21 (TT,TE,EE) or 15 (TT) sampled parameters. `"lite"`: plik_lite with `A_planck` only (much faster). `"extended"` and `"full"`: more of plik's inputs sampled. |
| `cmb_nonlinear`, `hmcode_version` | `None`, `"2020"` | Non-linear correction of the CMB lensing: `None` (linear), `"halofit"` or `"hmcode"` (HMcode 2020, or 2015 as Planck 2018 used). |
| `cmb_sample_theta` | `False` | `True` samples $100\,\theta_\mathrm{MC}$ with a flat prior instead of $H_0$ in groups with primary CMB, as Planck and DESI DR2 do; $H_0$ is then derived. |
| `lensing_only_priors` | `True` | Groups with CMB lensing but no primary CMB: $n_s \sim \mathcal{N}(0.96, 0.02)$ and $\tau_\mathrm{reio}$ fixed, as in lensing-only analyses. |
| `waic_include_cmb` | `False` | WAIC for groups with Planck data (slow: one CLASS run per WAIC draw). `False` reports it as n/a. |

### Sampler switches

| Setting | Default | What it does |
|---|---|---|
| `emcee_moves` | `"de_snooker"` | emcee's proposal: differential evolution (80% of the steps) plus a snooker move (20%), or `"stretch"` (emcee's own move). In Kosmulator's tests `de_snooker` needed 2 - 6 times fewer steps for the same posterior. |
| `zeus_light_mode` | `True` | Skips zeus's interval expansions after its tuning phase: more steps, but fewer likelihood calls per step and less time overall. |

Both may be changed between a run and its `--resume`.

### Plot switches

| Setting | Default | What it does |
|---|---|---|
| `corner_show_nuisance` | `False` | Draw the likelihoods' nuisance parameters (Planck foregrounds and calibrations, SN calibration, JLA's $\alpha$ and $\beta$) in the corner plots. The tables always list every parameter. |

### Less-used data options: `Kosmulator_main/constants.py`

A few data choices are made in `constants.py` rather than in `Kosmulator.py`, since they rarely change:

| Constant | Default | What it does |
|---|---|---|
| `PANTHEON_SYS_COV` | `None` | `"full"` adds the Pantheon release's systematic covariance (as in the published analysis). |
| `CC_SYS_COV` | `None` | `"full"` (or `"ooo"`) adds the SPS-model systematic covariance of the D4000 cosmic chronometers (Moresco et al. 2020). |
| `LATE_TIME_RADIATION` | `True` | Radiation in the late-time $E(z)$ through `radiation_density(p)`. |
| `MAIN_BLAS_THREADS_NO_POOL` | `4` | At most this many BLAS threads for groups that run in the main process without a worker pool (never more than the run's own threads; a run with `--num_cores 1` stays serial). |
| `CONV_*` | see file | Thresholds of the convergence rule. |

Every setting that differs from its default is printed in the `Settings:` line of each statistics file.

## Command-line flags

Kosmulator exposes a small set of command-line flags to control parallelism, the samplers, diagnostics and plotting. `Kosmulator.py` holds *what* is fitted; the flags control *how* the run goes. You can list them at any time with:

```bash
python Kosmulator.py --help
```

### General and output

| Argument | Type | Default | Description |
|---|---|---|---|
| `--output_suffix` | str | `Test_run` | Name of the run: the subfolder of `MCMC_Chains/`, `Plots/Saved_Plots/` and `Statistical_analysis_tables/` its results go to. |
| `--overwrite` | flag | off | Delete any existing chains of this run and sample from scratch. |
| `--resume` | flag | off | Continue incomplete chains (without it, existing chains are only loaded). A chain is continued only with the engine that made it; a group whose engine changed starts a new chain and says so. |
| `--init-log` | choice | `terse` | Start-up logging: `terse`, `normal` or `verbose`. |

::::{admonition} Running again with the same `--output_suffix`
:class: tip
Without `--overwrite` or `--resume`, a second run with the same suffix loads the saved chains and only redoes the statistics and plots. This is handy after changing a plot setting.
::::

### Parallelism

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num_cores` | int | `8` | Worker processes for groups that need a pool (CMB and BBN groups, models that cannot be vectorised, the official JLA). Other groups run vectorised in the main process. |
| `--class_threads` | int | `1` | CLASS threads per pool worker or MPI rank. A local pool then has `num_cores // class_threads` workers. Worth 2 or more only when `num_cores` is larger than `nwalkers/2`, since emcee and zeus evaluate half the walkers at a time. |
| `--use_mpi` | flag | off | Use an MPI pool (start the run with `mpiexec`; see [Clusters and long runs](running.md#in-parallel-and-on-clusters)). |

### Sampler and engine

| Argument | Type | Default | Description |
|---|---|---|---|
| `--engine-mode` | choice | `mixed` | `mixed` and `fastest`: per group, zeus for groups without CMB or BBN when the model is vectorised, emcee otherwise. `single`: one engine per model. |
| `--force_zeus` | flag | off | Use zeus wherever possible (not for CMB groups: they always run emcee, through the worker pool). |
| `--force_emcee` | flag | off | Use emcee everywhere. |
| `--force_vectorisation` | flag | off | Treat every model as vectorised (skips the detection). |
| `--disable_vectorisation` | flag | off | Evaluate every model point by point, even if it is vectorised. |

### Convergence

The convergence rule is described in [Configuring a run](running.md#sampler-settings); these flags tune when it is checked.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--autocorr-check-every` | int | `100` | Check the convergence rule (and redraw its plot) every `N` steps. |
| `--tau-consecutive` (alias `--consecutive-required`) | int | `1` | Number of consecutive checks at which all conditions must hold before the run stops (2 or more for a stricter stop). |
| `--autocorr-buffer` | int | none | Optional minimum run length: the rule may end a run only after `burn` + this many steps. |

### Plotting and presentation

| Argument | Type | Default | Description |
|---|---|---|---|
| `--latex_enabled` | flag | off | LaTeX rendering of the plot labels (needs LaTeX; see [Installation](getting_started.md)). |
| `--plot_table` | flag | off | Draw the posterior summary table on the corner plot (in the empty upper-right triangle when it fits, otherwise above the grid). |
| `--corner-show-all-cmb-params` | flag | off | Corner plot: show every CMB parameter, including the nuisances. |
| `--corner-table-full` | flag | off | Corner plot table: keep the full parameter list, including CMB nuisances. |

### Looking at a run and debugging

| Argument | Type | Default | Description |
|---|---|---|---|
| `--peek` | flag | off | Do not run: show the run `--output_suffix` as it stands (corner plots of the chains so far, means and the convergence table). Safe while the run goes on. |
| `--peek_burn` | int | run's burn-in | With `--peek`: steps to discard. |
| `--print_loglike [N]` | int (optional) | off | Every `N` steps (1 when given without `N`), for the walker with the highest posterior: each data set's $\chi^2$, the log prior and the total log-posterior. Each printed step evaluates the data once more, so use it to debug a likelihood. |

```text
[LCDM_v | CC+DESI_DR2] step 2000, best walker (7): CC chi2 = 14.62 | DESI_DR2 chi2 = 10.29 | log prior = 0.00 | TOTAL log-post = -12.45
```
