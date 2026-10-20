# stats_summary.txt redesign: proposed changes

Status: proposal only. Nothing in `Kosmulator_main/` or elsewhere has been modified.

## Files in this folder

| File | Purpose |
|---|---|
| `utils.py` | Copy of `Kosmulator_main/utils.py` with `save_stats_to_file` rewritten (CRLF line endings preserved) |
| `utils.patch` | `diff -u Kosmulator_main/utils.py Claude_Code/utils.py` (only that function changes) |
| `test_stats_summary.py` | Test script (run from anywhere: `python3 Claude_Code/test_stats_summary.py`) |
| `test_stats_summary/out/` | Test outputs (`.txt` and `.csv`, three cases) |

## What changed (in `save_stats_to_file`)

1. **Transposed `stats_summary.txt`**: one row per statistic, one column per observation. Rows are grouped into three blocks separated by rules: fit quality, information criteria, differences from the reference. Width is now about 50 characters plus about 30 per observation, instead of about 230 characters for a single observation.
2. **Header states the reference model and sign convention**: `dX = X(model) - X(reference)`, and Sigma is described (wording supplied by the user: the reference is the null hypothesis, and Sigma tests whether the user model is compatible with it). This matches `Post_processing.py:815-823`, where `d = stat - reference` and `sigma = significance(dChi, d_npar)` only when `d_npar > 0`, otherwise 0.0.
3. **Two extra rows**: `Free parameters` (from `n_params`, already in the stats dict) and `DOF` (recovered as chi^2 / reduced chi^2, exact by definition). The number of data points is not stored in the results, so it is not shown.
4. **New `stats_summary.csv`** beside the text file: one row per observation, full precision (`repr`), columns `Model, Reference_Model, Observation, Free_parameters, DOF, ...`. NaN is written as `nan`.
5. Signed formatting (`+1.620`) for the d-rows, Sigma to 1 decimal (it is rounded to 1 decimal in `significance`). The stray `|{sigma` formatting slip and trailing `|` are gone.

## Signature change and call sites

`save_stats_to_file(model, folder, stats_list, reference_model=None)`. The new argument is optional, so existing calls still work, but the header then says the reference was not passed. To get the reference name into the file, update the two callers:

- `Marcel_chain.py:242`: `save_stats_to_file(model_name, out_dir, stats_list, reference_model=reference_model)` (`reference_model` is defined at line 41).
- `Plots/Plots.py:647`: `save_stats_to_file(model, model_folder, stats_dict[model], reference_model=reference_model)`. I did not check whether `reference_model` is in scope in that enclosing function (`generate_plots` has it as a parameter at line 344, but line 647 may sit in a helper). Please check before editing.

## Testing

`test_stats_summary.py` loads the proposed `utils.py` and feeds it the real wowaCDM_v row from `Statistical_analysis_tables/DESI_DESY5_zHEL_32X2500/wowaCDM_v/stats_summary.txt` plus a synthetic second observation (long name, NaN DIC/WAIC, numpy scalar and 1-element array inputs). It checks the CSV contents, the recovered DOF (1837), that NaNs survive, that all table lines have equal width, and that the old three-argument call works. All checks pass.

## Caveats

- Not run through a full pipeline (`Marcel_chain.py` or `Plots.py`) because that needs the chains. Only the writer function was tested.
- The test numbers for the second observation are synthetic and carry no physical meaning.
- Anything that parses the old `stats_summary.txt` layout will break. I found no such parser in the `.py` files (grep for `stats_summary` only hits the writers), but please check notebooks such as `Sandbox.ipynb`.
- `Plots/Plot_functions.py:757` contains a separate, older wide-format writer with the same columns. I left it untouched; say if you want it aligned too.
- Sigma comes from `significance()` in `Post_processing.py:91`, which uses `abs(dChi)`, so it is never negative. If dChi > 0 (user model fits worse, possible with an imperfect minimiser) with extra parameters, Sigma is still positive. Read it together with the sign of dChi. A guard returning 0 for dChi > 0 is possible but not part of this proposal. Sigma = 0.0 also means "not applicable" when the model has no extra parameters (Wilks test needs nested models).
- The long observation name makes its column wide. That is intended, since names are never truncated.
