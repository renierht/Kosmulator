<p align="center">
  <img src="assets/Kosmulator.png" alt="Kosmulator logo" width="300">
</p>

# Kosmulator: ten interacting backgrounds with warm dark matter

This branch extends the eight IDE backgrounds to a constant dark-matter EOS
`w_dm` and adds two dark-sector sign-switching interactions (Models 9–10).
It includes independent free/fixed EOS switches, mathematical and optional
physical-domain filters, and a shared CLASS background for model-derived
baryon-drag sound horizons. “Warm” denotes a background EOS extension;
IDE perturbations and particle free streaming are not implemented.

**Additional comparison:** [CPL w0waCDM equations, priors, validation and launch instructions](docs/CPL_COMPARISON.md).

**Full equations and conditions:** [Ten-model equation reference](docs/WARM_DM_EQUATIONS.md).
This reference supersedes the earlier cold-only formulas in this README.
It contains each kernel and density solution, exceptional limits, global
real-domain and positivity conditions, future classification and supported
doom criteria. The cold-DM limit is `w_dm=0`.

For general installation and upstream usage, see the
[main repository](https://github.com/renierht/Kosmulator).
For implementation evidence, see [the warm-DM report](validation/WARM_DM.md).

## Repository layout and IDE additions

| Location | Purpose |
|---|---|
| `Kosmulator.py` | Main entry point: models, datasets, priors and sampler settings. |
| `User_defined_modules.py` | Analytical IDE models and physical-domain switches. |
| `Class/IDE_background/` | Shared IDE CLASS background backend for derived sound horizons; IDE perturbations are not implemented. |
| `Kosmulator_main/` | Sampling, likelihoods and postprocessing, including `rd_helpers.py`, `Model_comparison.py` and `Plot_metadata.py`. |
| `Plots/` | Corner plots, data/model figures and parameter tables. |
| [reproducibility/](reproducibility/README.md) | Project settings and instructions for generating and analysing chains. |
| [validation/](validation/README.md) | CLASS validation report and checks of the original analysis chains. |
| [tests/](tests/README.md) | Automated checks of configuration, loading, statistics, boundaries and plotting metadata. |

For SAIP settings and commands, see
[the reproduction guide](reproducibility/SAIP2026_IDE/REPRODUCE.md).

---

## Conservation equations and EOS selection

```math
\dot\rho_{\rm dm}+3H(1+w_{\rm dm})\rho_{\rm dm}=Q,\qquad
\dot\rho_{\rm de}+3H(1+w)\rho_{\rm de}=-Q.
```

Positive `Q` transfers energy from DE to DM. Models 9 and 10 use
`Q = 3 H delta q_dark rho_dm` and `Q = 3 H delta q_dark rho_de`, respectively,
where

```math
q_{\rm dark}=\frac{(1+3w_{\rm dm})\rho_{\rm dm}+(1+3w)\rho_{\rm de}}
{2(\rho_{\rm dm}+\rho_{\rm de})}.
```

This proxy excludes baryons and radiation. See the equation reference for its
assumptions and the complete ten-model kernel table.

Set the independent defaults in `User_defined_modules.py`:

```python
FREE_W = True
FREE_W_DM = False
```

| `FREE_W` | `FREE_W_DM` | Sampled EOS | Fixed EOS |
|---|---|---|---|
| True | False | `w` | `w_dm=0` |
| False | False | none | `w=-1`, `w_dm=0` |
| True | True | `w`, `w_dm` | none |
| False | True | `w_dm` | `w=-1` |

`Config.create_config(..., free_w=..., free_w_dm=...)` overrides the defaults.
Fixed EOS values are injected into likelihoods, CLASS and plotting; they are
not counted as sampled parameters. The editable default `w_dm` prior is
`(-0.1, 0.1)`. Add `SignSwitch_IDE_1` or `SignSwitch_IDE_2` to the model list
for Model 9 or 10. The default model list remains the original eight.
The paper launcher explicitly uses free `w` and fixed `w_dm=0`.

## Domain and physical-prior switches

The selected global real branch and finite positive expansion are always
required. These independent settings disable optional filters when `True`:

- `ALLOW_NEGATIVE_ENERGIES`: allow violation of global component positivity.
- `ALLOW_BIG_RIP`: allow futures other than the classified `no_big_rip` branch.
- `ALLOW_DOOM_FACTOR_INSTABILITIES`: run without the supported doom prior.

The positivity inequalities, Big-Rip boundaries and exceptional coefficients
are listed in [the equation reference](docs/WARM_DM_EQUATIONS.md).
The doom prior is available only for Models 1–8 with free `w` and fixed
`w_dm=0`. Warm/free DM EOS, fixed `w=-1` and Models 9–10 have no supported
classification; enforcing the prior raises a configuration error.
Disabling it does not assert stability.

## Shared CLASS background backend and model-derived sound horizon

This branch contains a single shared CLASS source tree at

```text
Class/IDE_background/
```

for the ten IDE background models. A single backend is used instead of maintaining ten duplicated CLASS trees. Kosmulator selects the required homogeneous IDE background at runtime with the following selector:

| `ide_model` | Kosmulator model |
|---:|---|
| 0 | Standard CLASS / IDE switched off |
| 1 | `Linear_IDE_1` |
| 2 | `Linear_IDE_2` |
| 3 | `Linear_IDE_3` |
| 4 | `Linear_IDE_4` |
| 5 | `Linear_IDE_5` |
| 6 | `NonLinear_IDE_1` |
| 7 | `NonLinear_IDE_2` |
| 8 | `NonLinear_IDE_3` |
| 9 | `SignSwitch_IDE_1` |
| 10 | `SignSwitch_IDE_2` |

For `Linear_IDE_1`, CLASS receives the two independent interaction parameters `delta_dm_ide` and `delta_de_ide`. Models 2--10 use the single interaction parameter `delta_ide`.

`w_dm` is passed as `w_dm_ide`; the dark-energy EOS is passed as `w0_fld`. The sound-horizon cache includes the DM EOS.

The CLASS implementation modifies the **homogeneous dark-sector background only**. The exact analytical IDE densities are evaluated inside the shared background module, while the ordinary CLASS thermodynamics calculation is retained. This allows Kosmulator to obtain the baryon-drag sound horizon

```math
r_d = r_s(z_{\rm drag})
```

from a CLASS calculation whose expansion history matches the selected IDE model.

The Python-side physical-domain switches remain the authoritative filters for optional conditions such as positive dark-sector energy densities, future big-rip avoidance, and doom-factor stability. The CLASS background contains only the mathematical/domain safeguards needed for safe evaluation of the analytical solutions.

### Enabling model-derived `r_d`

The optional switch

```python
DERIVE_RD_WITH_MODEL_CLASS = True
```

activates the model-derived CLASS sound-horizon route for `LCDM_v` and the ten IDE models listed above. When enabled for a BAO or DESI likelihood:

1. `r_d` is removed from the sampled parameter list for supported models;
2. Kosmulator passes the selected model name and interaction parameters to the matching CLASS background;
3. CLASS computes `rs_drag()` using the homogeneous model background and standard thermodynamics;
4. a mathematically invalid CLASS background is rejected as an invalid likelihood point; and
5. the strict route does **not** replace a failed model calculation with either the Eisenstein--Hu approximation or the legacy fixed sound horizon.

Models outside the validated set retain the existing legacy `r_d` policy.

The fixed thermodynamic calibration settings used by this route are

```text
Omega_b = 0.048
N_eff = 3.044
sum(m_nu) = 0.06 eV
N_ncdm = 3
```

with the neutrino mass distributed as three equal massive species in the CLASS call.

---

## Validation, saved chains and current limitations

The [warm-DM report](validation/WARM_DM.md) records independent conservation-ODE
comparisons for all ten backgrounds, exceptional-limit checks, full CLASS
background/sound-horizon checks, cold-limit regressions and sampling smoke tests.
The [original CLASS report](validation/VALIDATION.md) remains a historical
record of the eight-model cold baseline. The latest general integration passed
61 tests and the short Zeus save/load/resume checks; see the
[general update checkpoint](validation/GENERAL_FOLLOWUP_CHECKPOINT.md).
Short smoke chains establish workflow correctness, not convergence.

New chain metadata records parameter order, fixed EOS and physical-prior
policy. Changing these settings requires compatible metadata or separate
chains. Legacy metadata-free chains are accepted only for the original eight
IDE models with free `w` and fixed `w_dm=0`.
See [convergence and resume guidance](reproducibility/SAIP2026_IDE/CONVERGENCE.md).

The backend is homogeneous-background plus standard thermodynamics only.
It is not a self-consistent IDE CMB-anisotropy or warm-DM perturbation solver;
IDE CMB likelihoods are guarded against using it.
Some backgrounds do not admit CLASS's required early radiation era.

The current CLASS BBN helium table also has a finite baryon-density range:
with fixed `Omega_b=0.048`, its upper `omega_b=0.03287` corresponds to
`H_0` approximately `82.752` km/s/Mpc. Thus some of the nominal paper prior
`H_0=60–90` fails CLASS. This computational prior restriction remains unresolved;
no helium value or prior bound has been silently changed. See the
[measured performance and BBN report](validation/PERFORMANCE_CHECKPOINT.md).

## References

Users should also cite the original observational datasets used in any Kosmulator analysis; see the `main` branch README for the general Kosmulator and dataset references.

### Analytical solutions and positive-energy conditions

1. M. van der Westhuizen, A. Abebe, and E. Di Valentino, *Phys. Dark Univ.* **50**, 102119 (2025), arXiv:2509.04495 [gr-qc].
2. M. van der Westhuizen, A. Abebe, and E. Di Valentino, *Phys. Dark Univ.* **50**, 102120 (2025), arXiv:2509.04494 [gr-qc].
3. M. van der Westhuizen, A. Abebe, and E. Di Valentino, *Phys. Dark Univ.* **50**, 102121 (2025), arXiv:2509.04496 [gr-qc].

### Crunching and bouncing cosmologies

4. M. van der Westhuizen and A. Abebe, *Class. Quantum Grav.* (2026), https://doi.org/10.1088/1361-6382/ae849d.

### Doom-factor analysis

5. M. Gavela, D. Hernandez, L. Lopez Honorez, O. Mena, and S. Rigolin, *J. Cosmol. Astropart. Phys.* **2009**(07), 034.

### Previous observational constraints using these analytical solutions

6. M. van der Westhuizen, D. Figueruelo, R. Thubisi, S. Sahlu, A. Abebe, and A. Paliathanasis, *Phys. Dark Univ.* **50**, 102107 (2025), arXiv:2505.23306 [astro-ph.CO].
7. D. Figueruelo, M. van der Westhuizen, A. Abebe, and E. Di Valentino, *Phys. Dark Univ.* **52**, 102238 (2026).
