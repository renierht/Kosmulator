# clipy support (Planck likelihoods without compiled clik)

## What and why
The cluster installs of `clik` in `/home/user/clik` import but lack the compiled `clik.lkl` and
`clik.lkl_lensing` extensions, so `clik.clik(...)` fails (`'NoneType' object has no attribute 'clik'`)
and no CMB likelihood can be evaluated. `clipy` (PyPI: `clipy-like`, K. Benabed and L. Balkenhol) is a
pure-Python reimplementation of clik with the same API, so no compiled code is needed.

## Changes (3 files, imports and doctor only; no likelihood logic touched)
- `Class_run.py` (line ~33): `import clipy as clik`, falling back to `import clik`, then `None`.
- `Statistical_packages.py` (line ~47): same preference order. File keeps its CRLF line endings.
- `__init__.py` (`kosmulator-doctor`): the load test and import checks accept clipy; a working clipy
  makes compiled clik optional.

Behaviour: if clipy is installed it is used, otherwise clik exactly as before. Everything the code calls
(`clik.clik`, `clik.clik_lensing`, `get_lmax`, `get_extra_parameter_names`) exists in clipy 0.15.
`get_lkl_length` does not, but the code already guards it with `hasattr`.

## Install (cluster, py312 or any env with Python >= 3.10)
    pip install --user clipy-like        # NOT `clipy`, which is a different package

## Tested
- Downloaded `clipy-like` 0.15 and loaded all five Planck files in `Observations/` in an isolated
  scratchpad run: `plik_rd12_HM_v22b_TTTEEE`, `plik_rd12_HM_v22_TT`,
  `simall_100x143_offlike5_EE_Aplanck_B`, and both `smicadx12_..._consext8` lensing files.
  Each reproduced its built-in reference value (|diff| <= 2e-5).
- Patched files pass `py_compile`; `git apply --check` succeeds against the current working tree.

## Not tested
- No end-to-end Kosmulator CMB run (needs CLASS and the cluster environment). Do a tiny run first
  (`CMB_lowl`, then `CMB_hil`) and compare log-likelihoods with a laptop run that still uses compiled clik
  at the same parameter point before trusting new chains.
- JAX is optional in clipy; without it everything runs on NumPy. If JAX is present but lacks 64-bit
  support clipy warns that results may differ at the 1e-4 level in log-likelihood.

## Apply
    cd ~/Kosmulator && git apply Claude_Code/clipy_support/clipy_support.patch
(or copy the three modified files over `Kosmulator_main/`). Citation: Planck 2018 V, arXiv:1907.12875.

## Follow-up fix (fix1)
The first version renamed the doctor item "import clik (Planck/CLIK)", but the doctor's overall check
(`required_names` in `__init__.py`) matches that exact name, so the rename would have let the report say
READY even if neither clipy nor clik imported. The name is restored; its `ok` is still `clik or clipy`.
If you already applied `clipy_support.patch` before this fix, apply `clipy_support_fix1.patch` on top.
`clipy_support.patch` itself is regenerated against git HEAD and already includes the fix.
