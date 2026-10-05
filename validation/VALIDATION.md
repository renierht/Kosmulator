# Validation of the IDE CLASS background implementation

This document records the numerical and integration tests performed for the shared `Class/IDE_background/` implementation used by Kosmulator.

The validation applies to the **homogeneous IDE background** and to the calculation of the baryon-drag sound horizon using **standard CLASS thermodynamics**. It does **not** validate an IDE perturbation sector.

---

## 1. Validation scope and backend architecture

All eight interacting-dark-energy models are routed to a single CLASS tree:

```text
Class/IDE_background/
```

The selector mapping is:

| `ide_model` | Model |
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

`Linear_IDE_1` uses the independent couplings `delta_dm_ide` and `delta_de_ide`. Models 2--8 use the single coupling `delta_ide`.

The shared implementation was tested independently at the C-background level, the CLASS `rs_drag()` level, the Python routing layer, the BAO/DESI likelihood layer, the normal Kosmulator likelihood runner, and the final parameter-configuration policy.

---

## 2. Analytical background validation

Each IDE selector was compared directly against the exact analytical expressions for `rho_dm(a)`, `rho_de(a)`, and the corresponding Hubble history.

Each test used 40,000 points spanning approximately

```text
0 <= z <= 1e14
```

The maximum relative differences were:

| Model | `rho_dm` | `rho_de` | `H` |
|---|---:|---:|---:|
| `Linear_IDE_1` | 1.738157e-12 | 1.779564e-12 | 9.430637e-13 |
| `Linear_IDE_2` | 1.695792e-12 | 1.769273e-12 | 7.694101e-13 |
| `Linear_IDE_3` | 1.734410e-12 | 1.762528e-12 | 9.444943e-13 |
| `Linear_IDE_4` | 5.97e-13 | 2.12e-13 | 2.58e-13 |
| `Linear_IDE_5` | 1.856373e-12 | 6.707211e-13 | 7.674715e-13 |
| `NonLinear_IDE_1` | 1.799259e-12 | 7.116150e-13 | 8.996348e-13 |
| `NonLinear_IDE_2` | 1.769845e-12 | 1.687467e-12 | 7.780219e-13 |
| `NonLinear_IDE_3` | 1.795159e-12 | 6.688672e-13 | 7.458180e-13 |

These tests validate the implemented **homogeneous background equations**. They do not test IDE perturbations.

---

## 3. Standard CLASS regression (`ide_model = 0`)

The shared `IDE_background` tree was compared with the untouched `Class/LCDM_v` implementation with the IDE selector switched off.

For the tested wCDM configuration:

```text
array shapes: identical
maximum absolute difference: 0
maximum relative difference: 0
```

Thus, within the tested configuration, `ide_model = 0` reproduces the untouched standard CLASS background exactly.

---

## 4. Zero-coupling background regression

All eight IDE selectors were evaluated with the interaction coupling set to zero and compared with the corresponding uncoupled wCDM background.

The relevant dark-sector densities and Hubble history agreed at numerical precision. Typical relative differences were between approximately `1e-13` and `1e-11`.

Quantities that pass close to zero can show a comparatively large relative difference despite a very small absolute difference. These cases were checked using their absolute differences and did not indicate a physical regression.

---

## 5. Zero-coupling sound-horizon regression

A full-precision `rs_drag()` regression was performed using a common uncoupled wCDM reference with

```text
H_0 = 70 km s^-1 Mpc^-1
Omega_m = 0.307
Omega_b = 0.048
w = -0.95
N_eff = 3.044
sum(m_nu) = 0.06 eV
N_ncdm = 3
```

The reference result was

```text
r_d(reference) = 144.098082927208281 Mpc
```

The zero-coupling selectors gave:

| Model | `r_d` [Mpc] | Absolute difference [Mpc] | Relative difference |
|---|---:|---:|---:|
| `Linear_IDE_1` | 144.098083584466934 | 6.573e-07 | 4.561e-09 |
| `Linear_IDE_2` | 144.098083584466934 | 6.573e-07 | 4.561e-09 |
| `Linear_IDE_3` | 144.098083584466934 | 6.573e-07 | 4.561e-09 |
| `Linear_IDE_4` | 144.098082885301778 | 4.191e-08 | 2.908e-10 |
| `Linear_IDE_5` | 144.098082885301778 | 4.191e-08 | 2.908e-10 |
| `NonLinear_IDE_1` | 144.098082885301778 | 4.191e-08 | 2.908e-10 |
| `NonLinear_IDE_2` | 144.098082885301778 | 4.191e-08 | 2.908e-10 |
| `NonLinear_IDE_3` | 144.098082785962049 | 1.412e-07 | 9.802e-10 |

All eight selectors passed the adopted relative regression threshold of `1e-7`.

---

## 6. Non-zero-coupling `rs_drag()` smoke test

Representative finite, non-zero-coupling test points were evaluated for all eight IDE models.

The resulting CLASS sound horizons were:

| Model | `r_d` [Mpc] |
|---|---:|
| `Linear_IDE_1` | 151.349491888442 |
| `Linear_IDE_2` | 151.816842693137 |
| `Linear_IDE_3` | 150.000200902015 |
| `Linear_IDE_4` | 150.891096602350 |
| `Linear_IDE_5` | 145.005466719099 |
| `NonLinear_IDE_1` | 144.534402343273 |
| `NonLinear_IDE_2` | 150.423678934640 |
| `NonLinear_IDE_3` | 144.560769060988 |

The LCDM reference smoke-test value was

```text
144.098088944993 Mpc
```

These numbers are **test outputs for the selected parameter points**, not universal predictions of the corresponding models.

---

## 7. Shared backend routing and cache reuse

All eight IDE model names were confirmed to resolve to the same backend:

```text
IDE_background
```

Repeated calls from different IDE model names reused the same compiled CLASS backend rather than building separate model-specific copies.

The source-tree signature used by the validated cache was

```text
8b81a76fa7fc20140418b9643351f29e66a51f8a54120df2de30cfa719305fac
```

The source hashing deliberately excludes generated build directories and generated Cython `classy.c` / `classy.cpp` files so that compiling CLASS does not invalidate the source signature.

---

## 8. Mathematical-domain rejection and fallback protection

A deliberately invalid `NonLinear_IDE_2` point was tested:

```text
w = -0.95
delta = -1.0
```

For this model the implemented mathematical domain requires `delta > w`. CLASS correctly rejected the point in `background_ide7_densities`.

The strict sound-horizon route was then tested through all relevant helper paths. The result was:

```text
direct CLASS rejection: PASS
no EH98 fallback: PASS
no fixed-r_d fallback: PASS
strict likelihood rejection: PASS
```

Thus a mathematically invalid model point is not silently replaced by an unrelated sound-horizon prescription.

---

## 9. BAO and DESI likelihood-level integration

The likelihood-routing layer was tested with a valid `NonLinear_IDE_2` point:

```text
H_0 = 70
Omega_m = 0.307
w = -0.95
delta = 0.05
```

For this routing test CLASS returned

```text
r_d = 170.325447517887 Mpc
```

Using a deliberately simple finite background function to isolate the `r_d` routing layer, the likelihood calls returned

```text
BAO chi^2  = 144.489261925
DESI chi^2 = 5.86264919278
```

For the deliberately invalid `NonLinear_IDE_2` point described above:

```text
BAO chi^2  = 1e300
DESI chi^2 = 1e300
```

Instrumentation of the two forbidden fallback paths gave

```text
EH98 fallback calls      = 0
fixed-r_d fallback calls = 0
```

The synthetic distance function in this test was used only to isolate the likelihood-to-CLASS routing and fallback policy. The IDE background equations themselves were validated separately in Sections 2--5.

---

## 10. Kosmulator likelihood-runner validation

The normal

```python
Kosmulator_main.Kosmulator_MCMC.model_likelihood()
```

path was tested using the same valid `NonLinear_IDE_2` routing configuration.

The runner returned

```text
log-likelihood    = -72.24463096237417
-2 log-likelihood = 144.48926192474835
```

which agrees with the direct BAO likelihood test above.

The test also confirmed that:

```text
__model_name__ injected on every model-derived r_d call: PASS
r_d not sampled/injected beforehand: PASS
EH98 fallback calls: 0
fixed-r_d fallback calls: 0
```

Three model-derived `r_d` calls occurred during this BAO test because the BAO vector evaluates the `D_M/r_d`, `D_H/r_d`, and `D_V/r_d` routes separately.

---

## 11. Configuration-policy validation

Configuration generation was tested with model-derived CLASS `r_d` enabled for both a supported IDE model and an unsupported dummy model.

For `NonLinear_IDE_2`:

```text
['BAO', 'CC']      -> ['Omega_m', 'w', 'delta', 'H_0']
['DESI_DR2', 'CC'] -> ['Omega_m', 'w', 'delta', 'H_0']
```

For the unsupported dummy model:

```text
['BAO', 'CC']      -> ['Omega_m', 'w', 'delta', 'H_0', 'r_d']
['DESI_DR2', 'CC'] -> ['Omega_m', 'w', 'delta', 'H_0', 'r_d']
```

The checks passed:

```text
supported IDE removes r_d: PASS
unsupported model keeps legacy free r_d: PASS
supported IDE still contains H_0: PASS
```

This demonstrates that strict model-derived `r_d` behaviour is limited to the validated model set.

---

## 12. Repository and source-integrity checks

The final modified tracked Python files passed

```text
git diff --check
```

with exit code `0`.

The validated C implementation had the following SHA-256 hashes:

```text
80a783fa6830a164aefe439e5b93286c2f867e79ee3a192241f4d64baf322bbe  Class/IDE_background/source/background.c
8aae7b5fdc6a0c75bec0cbdb39aff6684dedaefb12c9dfb5259468b39d5ca345  Class/IDE_background/source/input.c
098796f74741c7b2297e0ec3e0f23df5bb7a0058b22f2f818b0f925247ee9a1c  Class/IDE_background/include/background.h
```

The validated CLASS routing file had

```text
c145e29d7d39805ec2f1165a2579a7829e2c30332558efb5766de72e66774291  Kosmulator_main/Class_run.py
```

These hashes provide traceability for the implementation used during the numerical validation described above.

---

## 13. Scope and limitations

The validation establishes that the eight analytical IDE backgrounds are implemented consistently in the shared CLASS background, that the zero-coupling limit recovers the corresponding standard background and sound horizon, and that Kosmulator's BAO/DESI routing uses the selected model-derived `r_d` without silent fallback in strict mode.

It does **not** establish a perturbation-complete IDE implementation.

The shared backend modifies the homogeneous background only. Standard CLASS thermodynamics are retained, but IDE perturbation equations are not implemented. The backend therefore must not be used to claim IDE CMB-anisotropy predictions or other results that require a self-consistent interacting-dark-energy perturbation sector.
