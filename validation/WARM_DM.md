# Ten-model warm-DM background update

Validated 13 October 2026 in WSL Conda `Kosmulator_test` (Python 3.11,
NumPy 2.4.6, SciPy 1.17.1). Original implementation branch: `codex/warm-dm-ten-models`.
The prepared local branch `codex/warm-dark-matter` includes that implementation,
the subsequent convergence/performance work and general fixes from `fec7ac2`.
It has not been pushed. The full current [equations and conditions](../docs/WARM_DM_EQUATIONS.md)
include all ten density solutions and their exceptional limits.

## Audit and preservation

The baseline is `Kosmulator_IDE_v2` at
`c2bc139951067d78f5884742efda458b468dc933`. WSL Git found no tracked changes
in the original checkout, only untracked `upload/`. Apparent Windows changes
to CLASS external links were filesystem artifacts. Work was isolated in:

`C:/Users/Marcel/.codex/visualizations/2026/10/13/01a1358d-d46e-7da0-8eda-70bb634653cf/Kosmulator_warm_dm`

Original files, analysis chains and environment packages were preserved.
CLASS binaries were built in the isolated worktree. Nothing was pushed.

After fetching origin, the baseline/main comparison had 19 IDE-only commits
and one main-only commit, `310034aded9f64d5b78ac01631af938dcf3d26c0`.
Its optional `overlay_model_at_data_points` fix was already implemented in
this baseline; the regression test passes. No additional merge was needed.

The handoff and all eight pages of the supplied clarified PDF were read;
the conditions/addendum pages were also rendered and visually inspected.
They supplied scientific requirements, not separate action authorization.

## Selecting the EOS and models

Set these independent switches in `User_defined_modules.py`:

```python
FREE_W = True
FREE_W_DM = False
```

| FREE_W | FREE_W_DM | Sampled EOS | Fixed EOS |
|---|---|---|---|
| True | False | w | w_dm = 0 (legacy default) |
| False | False | none | w = -1, w_dm = 0 |
| True | True | w, w_dm | none |
| False | True | w_dm | w = -1 |

`Config.create_config(..., free_w=..., free_w_dm=...)` can override these
defaults. Fixed parameters are removed before priors and dimensions are
built, then injected into likelihoods, CLASS sound horizons and plots.
They are reported separately from sampled parameters in model comparisons.
`Kosmulator.py` supplies an editable `w_dm` prior of (-0.1, 0.1), with guess 0.
The default selected model list remains the original eight. Add
`SignSwitch_IDE_1` and `SignSwitch_IDE_2` to select Models 9 and 10.

| Selectors | Registry names | Q / (3H) |
|---|---|---|
| 1 | Linear_IDE_1 | delta_dm rho_dm + delta_de rho_de |
| 2 | Linear_IDE_2 | delta (rho_dm + rho_de) |
| 3 | Linear_IDE_3 | delta (rho_dm - rho_de) |
| 4 | Linear_IDE_4 | delta rho_dm |
| 5 | Linear_IDE_5 | delta rho_de |
| 6 | NonLinear_IDE_1 | delta rho_dm rho_de / (rho_dm + rho_de) |
| 7 | NonLinear_IDE_2 | delta rho_dm^2 / (rho_dm + rho_de) |
| 8 | NonLinear_IDE_3 | delta rho_de^2 / (rho_dm + rho_de) |
| 9 | SignSwitch_IDE_1 | delta q_dark rho_dm |
| 10 | SignSwitch_IDE_2 | delta q_dark rho_de |

Here Q > 0 transfers DE to DM, and
`q_dark = ((1+3 w_dm) rho_dm + (1+3 w) rho_de) / (2(rho_dm+rho_de))`.
Baryons and radiation do not enter this proxy.

## Mathematical and physical conditions

`ide_background.py` separates local density evaluation, the selected global
real domain, global component positivity, future behavior and the optional
doom criterion. The three existing ALLOW switches retain separate effects.
All solutions require finite positive present dark densities. Expansion
also requires finite positive H squared at every requested point.

Write `u=w-w_dm`, `r=rho_dm0/rho_de0`, and
`P=(1+3 w_dm)r+1+3w`. Models 9/10 assume w < -1/3, w_dm > -1/3 and P < 0.

| Model | Selected global real domain | Global component positivity |
|---|---|---|
| 9 | delta <= -2u/P OR delta >= -2/3 | delta <= -2u/P OR delta >= 0 |
| 10 | -2ur/P <= delta <= 2/3 | -2ur/P <= delta <= 0 |

Under those assumptions, Model 9 has a Big Rip when w < -1 and
delta > -1-w_dm. Model 10 has no Big Rip for delta <= 1+w, a Big Rip
for 1+w < delta <= 2/3, and a finite-scale singularity above 2/3.
The latter is classified independently even though the global real-domain
prior excludes it from sampling. Unclassified futures and negative future
total density are rejected when `ALLOW_BIG_RIP=False`.

For Models 1–8, future classification uses surviving asymptotic density
modes, including vanishing coefficients, rather than applying qualified
PDF inequalities unconditionally. For example, Model 6 changes its future
dominant component when u+delta changes sign. Models 7/8 use the stated
sufficient real branch u < 0 and u < delta <= -u/r or -ur respectively.
Exceptional limits include repeated eigenvalues, triangular logarithmic
solutions, Model 6's constant ratio, and Models 9/10's zero ratio exponent
and delta = -2/3 or +2/3. Local limits can exist outside a global domain.

This is a homogeneous-background and sound-horizon extension. No warm IDE
perturbation equations or stability proof were added. Enforcing the
available interacting doom prior with warm DM, fixed w=-1, or Models 9/10
raises a configuration error explaining the unsupported classification.
Use `ALLOW_DOOM_FACTOR_INSTABILITIES=True` for those background analyses;
this permits running without that prior and does not assert stability.

Low-redshift Python backgrounds retain the legacy default Omega_r=0.
The sound-horizon backend retains calibrated photons/neutrinos. Comparisons
of Python with full CLASS explicitly match radiation normalization.
Warm pressure and its derivative enter CLASS's homogeneous equations;
standard thermodynamics microphysics is retained. A parameter choice that
precludes CLASS's required early radiation era can still fail CLASS setup.

## Chain compatibility and build safety

New chains have `.h5.metadata.json` sidecars recording parameter order,
fixed values, EOS settings, background version and physical-prior switches.
Load/resume rejects incompatible policies, including equal-dimensional
chains with different parameter meanings. Metadata-free legacy IDE chains
are accepted only for the original eight models with free w and fixed
w_dm=0. Historical prior provenance cannot be established without metadata.
Use distinct output locations for new EOS or prior choices. The paper
launcher explicitly pins the old EOS settings.

All ten models route to the shared `IDE_background` CLASS build. The sound
horizon cache key includes w_dm. A changed IDE source hash must rebuild;
an unverified model-local binary is no longer promoted into a new cache.
Zeus now force-flushes the final sample block between callback intervals,
including on resume, avoiding lost tails in short runs.

## Validation evidence

- Baseline: 32 original tests passed before implementation.
- Original warm-DM integration suite: 50 tests passed, including all 40 model/EOS combinations,
  fixed injection, chain mismatch protection, C kernel boundaries, and
  Models 9/10's model-derived BAO sound-horizon configuration.
- Independent DOP853 conservation ODEs: all ten models, 50 random cases,
  exact and near exceptional branches, and separate uncoupled densities.
  Tests use absolute component errors scaled by total dark-density magnitude
  near component zero crossings; they do not divide by a vanishing density.
- `warm_legacy_results.json`: 128 cold Python cases, maximum H relative
  difference 2.67e-15. CLASS selector 0 is unchanged; selectors 1–8 have
  H differences at floating-point precision and r_d relative changes below
  2.5e-8 (required tolerance 1e-7).
- `warm_class_results.json`: 20 full CLASS backgrounds, 40,000 points each,
  ten zero-coupling routes for both cold and warm DM, and ten production
  sound-horizon routes with massive-neutrino calibration. H and densities
  agree at floating-point precision; finite-difference pressure derivatives
  pass the 1e-5 scaled tolerance. Python/CLASS agreement is a consistency
  check; the separate ODE integrations provide independent validation.
- `warm_future_results.json`: 27 independent fraction/total ODE cases to
  ln(a)=60, asymptotic slopes and proper-time tails, plus Model 10's finite
  scale endpoint. This finite coverage complements analytic classification;
  it is not an exhaustive proof over all parameter values.
- `warm_sampling_results.json`: twelve real-CC emcee save/load/resume runs
  (Models 7, 9, 10 × four EOS policies), with best-fit plot generation.
- `warm_zeus_results.json`: two vectorized Zeus save/load/resume runs, covering
  both fixed and both free EOS. All smoke runs use 16 walkers, 24 steps,
  then resume to 28. They verify workflow, not posterior convergence.

From this worktree in WSL, run with `/home/marcel/miniconda3/bin/conda run
--no-capture-output -n Kosmulator_test python` followed by:

```text
-m unittest discover -s tests -p "test*.py"
validation/validate_warm_class.py
validation/validate_warm_future.py
validation/validate_warm_legacy.py --baseline /mnt/c/Users/Marcel/PhD/Kosmulator_test_Class/Kosmulator_referee_clean
validation/smoke_warm_sampling.py
validation/smoke_warm_zeus.py
```

Use `MPLBACKEND=Agg` for headless plotting. Smoke chains use temporary
directories; generated plots and CLASS builds are ignored artifacts.

## Subsequent verification

The general integration at `fec7ac2` passed all 61 tests and the two Zeus
save/load/resume smoke cases. It adds CLASS relink/stale-cache checks, confirmed-empty
emcee recovery and aligned Zeus posterior storage. See
[the integration checkpoint](GENERAL_FOLLOWUP_CHECKPOINT.md).
The documentation update does not change equations, likelihoods or active chains.

## Explicit PDF-form equivalence check

The public model functions in `User_defined_modules.py` delegate to the shared
analytical implementation; they do not repeat the PDF's long formulas verbatim.
See [the implementation mapping and measured comparison](../docs/WARM_DM_EQUATIONS.md#how-the-pdf-equations-map-to-the-implementation).

`validate_pdf_equivalence.py` directly transcribes the generic PDF-form densities
and compares them against production over 100 cases / 8,100 scale-factor points.
The maximum scaled density error is 5.34e-15 and maximum relative H error is
2.22e-15; the Model 9/10 switching residual is at most 4.61e-16.
`pdf_equivalence_results.json` records the exact coverage and tolerances.
This finite formula comparison is distinct from the independent conservation-ODE
and exceptional-limit checks in `tests/test_warm_background.py`. It does not
validate new perturbation equations or remove the documented CLASS prior limitation.
