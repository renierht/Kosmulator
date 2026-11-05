# Model 9 completion audit and next runs

The fresh Model 9 chain completed at 7200 steps, with 1000 discarded and 744000 retained samples. The stored convergence flag and independently recomputed native diagnostics agree: tau_max=59.46194665, Npost/tau=104.2684, approximate ESS=12512.20, relative tau change=0.0034744, maximum walker-based split Rhat=1.00996544. This is the configured single-ensemble policy, not independent-ensemble rank-normalized convergence evidence.

Every saved coordinate and log posterior is finite. No stationary walkers; per-walker acceptance is 0.5375–0.5643. Median shifts between post-burn halves are below 0.029 posterior standard deviations. No postprocessing samples were excluded.

## Native comparison

The completed native postprocessing results were combined with ONLY the LCDM row from Model9_LCDM_native_preliminary_20261016_iw. The old exploratory Model 9 row was not used. Both used burn=1000, polished likelihood maxima, posterior-mean DIC and N=1670 (1657 PantheonPS + 13 DESI DR2). No new costly CLASS-derived plot computations were needed. The native compare_models/export_comparison functions produced Statistical_analysis_tables/Model9_fresh_vs_LCDM. Differences are Model 9 minus LCDM: delta chi2=-11.28080925, delta AIC=-7.28080925, delta AICc=-7.25915732, delta DIC=-7.28917564, delta BIC=+3.56034856.

The previously documented CLASS BBN-table restriction remains. Passing sampler diagnostics does not eliminate that numerical restriction or establish absence of disconnected posterior modes.

## Next runs (prepared, not started)

Use the CPL-capable Kosmulator_followup checkout. Shell wrappers load the existing WSL environment. Both preserve datasets, fixed CLASS assumptions, full-chain saving, 120 walkers, burn=1000, cap=100000, 8 workers and convergence policy. Model 10 is SignSwitch_IDE_2, unrestricted iw, free w, fixed w_dm=0. CPL uses w0 in [-3,1], wa in [-3,2], with w0+wa<0 and cold dark matter. Common priors match the completed runs.

Initialization differs from the pilot-covariance start used for Model 9: each new model uses its own native optimized initial point, with init_jitter_frac=0.01 instead of 0.10. This reduces prior-width scattering without changing target priors. The native initializer evaluates positions and regenerates nonfinite ones. These are prepared launch configurations, not claims of validated new-model convergence; early walker diagnostics should be inspected. No Model 9 samples are transferred.

Validation: both launcher --plan checks passed; 7 CPL tests and 9 warm-background tests passed; git diff --check passed. No new long MCMC or new-model full CLASS smoke run was executed in this review.

From WSL:
```bash
cd "/mnt/c/Users/Marcel/.codex/visualizations/2026/10/13/01a1358d-d46e-7da0-8eda-70bb634653cf/Kosmulator_followup"
bash reproducibility/SAIP2026_IDE/scripts/run_cpl_comparison.sh
# After CPL, launch Model 10:
bash reproducibility/SAIP2026_IDE/scripts/run_model10_comparison.sh
```

To resume either saved run after interruption, append --resume to its wrapper command. Do not append --resume to the first launch. Output suffixes are CPL_comparison_v1_cpl and Model10_unrestricted_v1_iw. Each single-model run will need its completed IC results compared against the same saved LCDM reference afterwards.
