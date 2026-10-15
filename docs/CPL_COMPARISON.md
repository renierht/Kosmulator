# CPL comparison with the IDE models

`w0waCDM` is a separate, flat, noninteracting comparison model. It does not change
Models 1–10 or their equations. Dark matter is cold; curvature is zero.

## Equations and implementation

$$
w(a)=w_0+w_a(1-a), \qquad a=(1+z)^{-1}.
$$

$$
\rho_{\rm de}(a)=\rho_{{\rm de},0}\,
a^{-3(1+w_0+w_a)}\exp[3w_a(a-1)].
$$

For late-time distance likelihoods, matching the existing `LCDM_v` convention:

$$
E^2(z)=\Omega_m(1+z)^3+(1-\Omega_m)
(1+z)^{3(1+w_0+w_a)}\exp\left[-\frac{3w_a z}{1+z}\right].
$$

`User_defined_modules.py` implements this expression directly. At
`w0=-1, wa=0` it reduces to the existing LCDM distance model.

For the sound horizon, `rd_helpers.py` uses the native CPL fluid in the
`Class/LCDM_v` backend, with `Omega_Lambda=0`, `w0_fld=w0`, `wa_fld=wa` and
`use_ppf=yes`. At the exact Lambda limit it uses identical CLASS inputs to LCDM.
The CLASS cache key includes both EOS parameters. A failed model-specific
calculation is rejected; no EH98 or fixed-r_d substitution is permitted.
No CMB spectra or growth likelihood are added by this comparison launcher.

The late-time distance model omits radiation and explicit neutrino evolution,
just as the existing paper's LCDM model does. CLASS includes radiation and
massive neutrinos for r_d, with the existing convention
Omega_cdm = Omega_m - Omega_b. Consequently the late-time formula is not claimed
to equal the full radiation/neutrino CLASS expansion history at all redshifts.
This preserves the existing comparison convention rather than redefining Omega_m.

## Data, priors and sampler

The launcher uses the exact `paper.json` PantheonPS (Pantheon+ with SH0ES)
and DESI_DR2 likelihoods, data and covariance files. It adds no CMB or BBN
observational likelihood. CLASS BBN consistency is a thermodynamics assumption,
not an additional observational dataset.

| Parameter | Uniform bounds |
| --- | --- |
| Omega_m | 0.1 to 0.5 |
| H_0 | 60 to 90 km/s/Mpc |
| M_abs | -20.5 to -18 |
| w0 | -3 to 1 |
| wa | -3 to 2 |

There is also an explicit joint prior **w0 + wa < 0**, enforced before likelihood
evaluation. These EOS bounds and the joint condition follow the
[DESI DR2 cosmological analysis, CPL discussion](https://journals.aps.org/prd/pdf/10.1103/tr6y-kpc6).
They permit crossing w=-1 and ensure DE becomes subdominant to matter in the
asymptotic past. They do not guarantee negligible DE at every finite early epoch.
The uniform density is on the permitted part of the rectangle; posterior results
and Bayesian evidence depend on this prior choice. IDE positivity, Big Rip and
doom-factor switches do not impose extra restrictions on CPL.

Fixed CLASS assumptions remain Omega_b=0.048, N_eff=3.044, sum(m_nu)=0.06 eV,
three massive species, and model-derived r_d. The comparison has five sampled
parameters: Omega_m, w0, wa, H_0, M_abs.

Production defaults: emcee, 120 walkers, fixed 1000-step burn, no thinning,
full-chain saving, 100000-step cap. The existing convergence policy requires
N_post/tau_max >= 100, approximate pooled ESS >= 2000, split Rhat < 1.01,
relative tau stability < 0.05, and one passing check. Reaching the cap alone
is not evidence of convergence. A later change in discarded burn requires
rechecking diagnostics. A separate chain is required for this new model.

## Known shared limitation

The current CLASS BBN helium table supports omega_b only up to about 0.03287.
With fixed Omega_b=0.048 this corresponds to H_0 about 82.75, below the nominal
upper prior of 90. These evaluations can fail and effectively restrict the prior.
This implementation does not fix YHe, extrapolate the table or narrow H_0.
The issue applies to the existing comparison runs too; review failures and
posterior boundary behaviour before publishing comparisons. A consistent repair
that changes accessible likelihood support requires new production sampling.

## Launch from WSL

Run in the development checkout containing this change, not the active IDE
checkout. Preview the settings first:

```bash
python reproducibility/SAIP2026_IDE/scripts/run_analysis.py \
  --only-model cpl --run-name CPL_comparison_20261015 --plan
```

Then, with `Kosmulator_test` active:

```bash
MPLBACKEND=Agg python -u reproducibility/SAIP2026_IDE/scripts/run_analysis.py \
  --only-model cpl --run-name CPL_comparison_20261015 \
  --max-steps 100000 --burn 1000 --num-cores 8
```

The launcher fixes OpenMP/BLAS thread counts to one per worker. Output suffix:
`CPL_comparison_20261015_cpl`. Resume using the same command plus `--resume`.
The launcher also supports `--mode analyse --chains mapping.json`, whose mapping
must contain exactly `w0waCDM`. It does not automatically rerun or load LCDM.
Compare saved runs only after verifying their shared scientific assumptions.
No production CPL chain is started by the implementation/validation work.

## Reproducible validation

```bash
python -m unittest discover -s tests -p test_cpl.py
python validation/validate_cpl.py
```

The unit tests check the exact LCDM distance limit, independent numerical
integration of the DE conservation equation, parameter selection, joint prior,
resume metadata and CLASS parameter/cache routing. The compiled validation
checks native CLASS densities against the analytic CPL expression, the exact
LCDM sound-horizon and paper-likelihood limit, and a small emcee save/resume run
in temporary files. This smoke run is not a convergence test. Results are saved
in `validation/cpl_results.json`.

Validated on 2026-10-15 in `Kosmulator_test`: all 68 repository tests passed.
Five real CLASS cases (including crossing and near-boundary EOS choices) had
maximum DE-density discrepancy scaled by total density of 3.59e-8. The exact
Lambda limit gave r_d=140.63274956513345 Mpc and log likelihood
-736.3578067034409, identical to the audited LCDM point. The 12-walker,
five-step temporary sampler run passed save/resume with finite probabilities
and mean acceptance 0.65. These are integration checks, not posterior results
or evidence of convergence for a production CPL run.
