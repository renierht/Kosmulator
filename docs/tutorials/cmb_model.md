# Tutorial 3: Your model with CMB data

Late-time data need only $E(z)$. CMB data need the model's **CMB power spectra**, which Kosmulator computes with CLASS. This tutorial gives the wCDM model of [Tutorial 2](own_model.md) the two things CMB data need:

1. a CLASS build for the model, in `Class/<model>`, and
2. a *CMB helper*, a function `<model>_CMB(p, mode)` in `User_defined_modules.py` that sets up CLASS for a point and returns its spectra.

::::{admonition} What you need installed
:class: note
CMB likelihoods need the `cmb` extra (`pip install -e ".[cmb]"`, part of the conda environment) and a C compiler, which `kosmulator-doctor --setup` checks.
::::

## 1. A CLASS build for the model

Each model fitted to CMB data has its own copy of CLASS in `Class/<model>`. Kosmulator compiles it the first time the model meets CMB data, caches the build (`Class/.cache/`), and rebuilds it only when its sources change. Models with modified perturbations (for example a modified-gravity model) change the CLASS sources in their own folder; that is why every model has its own.

wCDM needs nothing beyond stock CLASS (its fluid dark energy, `w0_fld`), so a copy of the ΛCDM folder will do:

```bash
cp -r Class/LCDM_v Class/wCDM_v
kosmulator-doctor --build-class wCDM_v
```

The second line builds it straight away (about 1 - 3 minutes) so that any compiler error shows now rather than at the start of a run; the compiler output goes to `Class/wCDM_v/build.log`.

## 2. The CMB helper

The helper receives the full parameter dictionary of a point (the model's parameters together with $H_0$, $\Omega_b h^2$, $\Omega_\mathrm{dm} h^2$, $n_s$, $\ln(10^{10}A_s)$ and $\tau_\mathrm{reio}$) and a `mode`: `"lowl"` asks for unlensed spectra up to $\ell = 31$, anything else for lensed spectra to the $\ell_\mathrm{max}$ the run needs. It returns CLASS's $C_\ell$ dictionary, or `None` to reject the point.

The easiest start is to copy the helper of a similar shipped model. For wCDM that is `wowaCDM_v_CMB` with $w_a = 0$. Add this below the other CMB helpers in `User_defined_modules.py`:

```python
def wCDM_v_CMB(p: dict, mode: str = "hil"):
    """
    CMB helper for wCDM_v (constant w0): CLASS's fluid dark energy with wa_fld = 0.
    """
    p = _ensure_background_params(p)
    is_lowl = (mode or "").lower().startswith("low")

    Om = float(p.get("Omega_m", 0.0))
    Ob = float(p.get("Omega_b", 0.0))
    if Om <= 0 or Ob <= 0 or Ob >= Om:
        return None

    import classy
    cosmo = classy.Class()

    if is_lowl:
        class_params = {"output": "tCl,pCl", "l_max_scalars": 31, "lensing": "no"}
    else:
        class_params = {"output": "tCl,pCl,lCl", "l_max_scalars": _K.CMB_CLASS_LMAX, "lensing": "yes"}

    base_params = {
        "n_s": float(p["n_s"]),
        "h": float(p["H_0"]) / 100.0,
        "omega_b": float(p["Omega_bh^2"]),
        "omega_cdm": float(p["Omega_dh^2"]),
        "tau_reio": float(p["tau_reio"]),
        "A_s": float(np.exp(p["ln10^10_As"]) * 1e-10),
        # Dark energy: no Lambda, a fluid with constant w
        "Omega_Lambda": 0.0,
        "use_ppf": "yes",
        "w0_fld": float(p["w0"]),
        "wa_fld": 0.0,
        "cs2_fld": 1.0,
    }

    quiet = {k: 0 for k in ("input_verbose", "background_verbose", "thermodynamics_verbose",
                            "perturbations_verbose", "transfer_verbose", "primordial_verbose",
                            "lensing_verbose", "output_verbose")}

    cosmo_params = {**base_params, **class_params, **_class_standard("wCDM_v", not is_lowl), **quiet}

    try:
        cosmo.set(cosmo_params)
        cosmo.compute()
        return cosmo.raw_cl() if is_lowl else cosmo.lensed_cl()
    except classy.CosmoComputationError:
        return None      # CLASS cannot compute this point: Kosmulator rejects it
    finally:
        _free_class(cosmo)
```

The pieces that every helper shares:

- `_ensure_background_params(p)` fills in $\Omega_m$, $\Omega_b$ and the physical densities from whichever of them the group samples;
- `import classy` *inside* the function picks up the build Kosmulator loaded for this model;
- `_class_standard(model_name, lensed)` adds the run's neutrino settings and, for lensed spectra, its non-linear correction (`cmb_nonlinear`), so that every model gets the same CLASS settings;
- `_free_class(cosmo)` releases CLASS's memory after the spectra are copied out.

::::{admonition} Name the helper exactly `<model>_CMB`
:class: warning
Kosmulator finds the helper by its name. Without one it warns (`No model-specific CMB helper (wCDM_v_CMB). Using generic CLASS fallback.`) and computes the spectra with stock ΛCDM settings, which ignore your model's parameters.
::::

## 3. Run with CMB data

```python
model_names: List[str] = ["wCDM_v"]
observations: List[List[str]] = [
    ["CMB_hil", "CMB_lowl_TT", "CMB_lowl", "DESI_DR2"],
]
cmb_nuisance: str = "lite"      # plik_lite: much faster than the full Planck nuisances
```

```bash
python Kosmulator.py --num_cores 16 --output_suffix wCDM_CMB --plot_table
```

CMB groups run emcee through the worker pool, one CLASS run per point, so they take hours rather than minutes; see [Clusters and long runs](../running.md#in-parallel-and-on-clusters) for running them on many cores and for following them with `--peek`.

## 4. A modified CLASS

For a model that changes the physics inside CLASS (its background or its perturbations), the steps are the same, but the CLASS sources in `Class/<model>` are edited and the helper passes the model's own CLASS input parameters. `f1CDM_v` is the shipped example: its CLASS reads the extra parameter `n_fT`, and `f1CDM_v_CMB` passes the model's `n` to it. Run `kosmulator-doctor --build-class <model>` after every change to the sources; Kosmulator would also rebuild by itself at the next run.
