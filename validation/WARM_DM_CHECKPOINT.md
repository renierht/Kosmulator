# Warm-DM development checkpoint — 13 October 2026

**Historical pause record, superseded after resumption.** The outstanding
items below have now been addressed. See [WARM_DM.md](WARM_DM.md) for the
completed implementation, final 50-test suite, numerical evidence and limits.
The remainder is retained solely as a record of the earlier pause.

Paused at the user's explicit request because the five-hour usage window is nearly exhausted. Continue in the same chat. **This is an incomplete development checkpoint, not a finished or release-ready implementation.**

## Workspace and preservation

- Original checkout: `C:/Users/Marcel/PhD/Kosmulator_test_Class/Kosmulator_referee_clean`.
- Original branch/base: `Kosmulator_IDE_v2`, `c2bc139951067d78f5884742efda458b468dc933`.
- WSL audit found no tracked modifications; only untracked `upload/`. Windows-reported CLASS external-link modifications/deletions were filesystem artifacts. Original files and upload were not edited.
- Worktree: `C:/Users/Marcel/.codex/visualizations/2026/10/13/01a1358d-d46e-7da0-8eda-70bb634653cf/Kosmulator_warm_dm`.
- WSL worktree: `/mnt/c/Users/Marcel/.codex/visualizations/2026/10/13/01a1358d-d46e-7da0-8eda-70bb634653cf/Kosmulator_warm_dm`.
- Development branch: `codex/warm-dm-ten-models`. Initially created beside original, then relocated with git worktree move into this chat's writable workspace.
- Use `/home/marcel/miniconda3/bin/conda run --no-capture-output -n Kosmulator_test ...` in WSL. Python 3.11, NumPy 2.4.6, SciPy 1.17.1, classy and clik available. WSL execution requires elevated tool access because sandboxed WSL returned E_ACCESSDENIED.
- No push, original-chain write, or environment package installation was performed. CLASS built into the new worktree and its Class/.cache.

## Audit / reference

Read `C:/Users/Marcel/Downloads/Kosmulator_update_handoff.md` and extracted all eight pages of `Warm_DM_10_models_Kosmulator_BigRip_Clarified_portrait.pdf` from Downloads. PDF render/visual verification remains outstanding.

Fetched origin. Comparison remains 19 commits unique to baseline and one unique to main: `310034aded9f64d5b78ac01631af938dcf3d26c0`, the optional overlay_model_at_data_points plotting fix. Its implementation is already present in baseline Plots/Plots.py and its regression test passes. No merge/cherry-pick was needed or performed; document this in final audit.

## Implemented draft

- New `Kosmulator_main/ide_background.py`: density solutions for ten models, independent domain/global positivity/future/doom checks, expansion wrapper.
- Models 1–5 use real matrix modes, including repeated eigenvalues; Model 6 has constant-ratio limit. Models 9/10 use integrated linear density-ratio equations, including logarithmic-ratio and delta = -2/3, +2/3 limits. Models 7/8 retain the specified sufficient real branch.
- User_defined_modules delegates eight existing names plus `SignSwitch_IDE_1` (9), `SignSwitch_IDE_2` (10). Independent `FREE_W=True`, `FREE_W_DM=False` defaults preserve cold-DM sampling defaults. Fixed values are -1 and 0.
- Config.create_config accepts free_w/free_w_dm, removes fixed parameters before prior/dimension construction, records fixed_parameters and eos_policy. Injection added to likelihood, vectorization detection, plot medians and derived-r_d samples. New default prior/guess for w_dm in Kosmulator.py; default model selection still the original eight.
- Shared CLASS header `include/ide_densities.h`, selector range 0..10, input `w_dm_ide`, true DM pressure and pressure derivative, pressure-based matter/radiation bookkeeping. All ten routes share backend and r_d cache includes w_dm.
- New chain_policy metadata sidecars distinguish parameter meaning even at identical dimensions. Checks inserted in load-only, common reuse and direct sampling entry points. Legacy metadata-free IDE chains accepted only for free w / fixed cold DM and original eight models. Paper launcher explicitly restores these EOS switches.
- Future behavior for 1–8 uses surviving density modes rather than blindly trusting the PDF's qualified inequalities. Negative future total is separate. Models 9/10 use supplied addendum under its assumptions, independent of positivity. Warm perturbative stability and Models 9/10 stability are unclassified; current doom_allowed rejects interacting unclassified points if that optional prior is enforced. This needs clear user-facing documentation and configuration diagnostics.

## Checks already completed

1. Baseline: all 32 original automated tests passed in WSL Kosmulator_test.
2. Latest suite: all **40 tests passed**, including 40 model/switch configurations, fixed-parameter injection, same-dimensional chain-policy mismatch rejection, ten independent DOP853 conservation-equation comparisons, uncoupled/cold limits, exceptional limits, and condition independence.
3. Compiled CLASS successfully and loaded the isolated binary (signature prefix aa728ad2 at build time).
4. `validation/validate_warm_class.py` passed: 20 full backgrounds (10 selectors x w_dm=0,.025), 40,000 points each, matched radiation normalization. Max H relative discrepancy ~1.67e-15; max component-density error scaled by total absolute dark densities ~7.40e-15; pressure scaled discrepancies <7e-16. These are consistency checks, not independent physical validation.
5. All ten zero-coupling CLASS sound horizons agree within 1e-7 relative for cold and warm DM, with w=-1. All ten production r_d routes with massive-neutrino calibration return finite positive results. Full recorded values in `validation/warm_class_results.json`.
6. Independent ODE generic points passed over ln(a)=-7..3, comparing absolute component error scaled by dark-density magnitude with threshold 2e-8. Exceptional branches checked locally at ln(a)=+/-0.05 to remain before possible finite branch endpoints.

During development independent ODE checks caught and corrected Model 10's integral sign. All results above are after that correction.

## Outstanding work / known issues to investigate before completion

- **CLASS cache safety:** existing Class_run.ensure_class_ready promotes any model-local .so to a new source hash without proving it matches updated sources. Current tested binary was freshly built. Before more C edits, fix/guard this fallback for IDE or force rebuilding; do not accidentally test stale binaries.
- Matrix formulas presently combine large terms: tiny individual component densities can lose precision even when H/total agrees. Strengthen individual-density uncoupled tests, consider exact zero-coupling branch and coefficient-first modes, and validate near repeated eigenvalues. Current uncoupled tests mainly check the total.
- Numerically inspect the ratio-integral formula near delta=+/-2/3 and k=0; current exact-limit tests pass, but near-limit cancellation and large dynamic ranges need scrutiny. A log1p formulation for the reciprocal-ratio integral may improve this.
- Explicit cold-DM comparison against the original checkout's eight Python functions and original compiled CLASS/selector-0 regression still required. Current tests check new Python versus new CLASS plus independent ODEs, not that full legacy comparison.
- Broaden domain/positivity and future-singularity tests (including boundary coefficients and negative-density branches); independently validate the future asymptotics/proper-time integral. Do not claim the entire Big-Rip audit is complete.
- Audit whether Model 9/10 sufficient assumptions and exceptional/global domains are consistent in Python versus C. Scientific density evaluator is local-branch; expansion and C impose global selected domain.
- No new perturbation equations implemented. Current policy for enforcing an unclassified doom prior should be made explicit and fail clearly at configuration time if appropriate, not leave sampling stuck with all invalid walkers.
- Complete short sampling smoke runs for all four EOS modes (at least Models 9/10 and existing model); test actual chain save/load/resume metadata and plots. Current tests cover config and likelihood injection, not full sampling workflows.
- Audit remaining reporting/statistics paths for fixed-parameter injection and metadata. Model_comparison MLE_parameters currently lists sampled parameters; add separate fixed metadata. Verify plot labels/support for warm/free modes and guard cold-specific boundary polishing.
- Preserve legacy geometry semantics: low-z Python defaults Omega_r=0, while sound-horizon CLASS uses calibrated photons/neutrinos. Full consistency comparison above explicitly matches radiation. Decide/document radiation convention without silently changing old-chain likelihoods.
- Three one-use integration helper scripts were created under validation during edits; do not rerun them. They are removed at checkpoint (or should be excluded from final deliverables).
- Latest `git diff --check` reported inherited trailing whitespace exposed by file newline rewriting in Config.py, Kosmulator_MCMC.py, utils.py and User_defined_modules.py. Clean formatting and review diff size before final commit/release; do not reset existing original files.
- Final documentation, full regression after fixes, diff review and final delivery remain outstanding. Do not push unless requested.

## Resume commands

From the isolated WSL worktree:

```sh
MPLBACKEND=Agg /home/marcel/miniconda3/bin/conda run --no-capture-output -n Kosmulator_test python -m unittest discover -s tests -p 'test*.py' -v
/home/marcel/miniconda3/bin/conda run --no-capture-output -n Kosmulator_test python validation/validate_warm_class.py
```

User's current instruction is to stop and checkpoint; resume implementation only when they return.
