# Linear sum/difference EOS study — 2026-11-18

Order: Model 2 Q=3H delta (rho_dm+rho_de), then Model 3 Q=3H delta (rho_dm-rho_de). Each has cold Lambda (w=-1, w_dm=0), free DE (w free, w_dm=0), warm Lambda (w=-1, w_dm free). Delta remains free in every case. Unrestricted iw background regime.

Pantheon+SH0ES, DESI DR2 BAO, correlated 36-point CC. Same fixed Omega_b=0.048 and model-derived CLASS sound horizon. Uniform priors: Omega_m [0.1,0.5], H0 [60,90], M_abs [-20.5,-18], delta [-1,1], w [-2,-0.33] or w_dm [-0.15,0.15] where free. Existing numerical validity restrictions unchanged.

120 walkers; emcee DE/snooker; eight workers, one BLAS/OpenMP thread each; burn 3000; 100000-step cap; full chain saving. Existing stopping policy unchanged (100 tau, ESS 2000, Rhat 1.01, tau change 0.05). Initial scatter is 0.002 of prior width, as adopted after the Model 4 initialization problem; it does not narrow the sampling prior. Single-ensemble diagnostics cannot establish independent mode coverage.

A separate run_linear_eos_analysis.py preserves the original launcher's source hash for historical resumes. Its own hash is included in new manifests. No kernel or core sampler changed. The queue stops on any subprocess failure, including postprocessing errors; inspect before restarting. Do not restart the entire queue blindly after interruption: select the interrupted case with --case and --resume, then launch the remaining cases separately.

Validation: tests/test_sum_difference_study.py (12 passing tests), validation/sum_difference_eos_preflight.json (12 background/CLASS cases), validation/sum_difference_plans.jsonlog (six settings plans). Maximum scaled ODE difference 1.25e-10; zero-coupling cold CLASS r_d agrees with LCDM to 1.95e-9 relative. These are bounded preflight tests, not exhaustive prior-domain validation.

Launch script: run_logs/start_sum_difference_eos.sh. Log: run_logs/LinearSumDifference_EOS_CC_v1.log. Each case has separate MCMC_Chains, Plots/Saved_plots and Statistical_analysis_tables output suffixes starting LinearSumDifference_EOS_CC_v1_model2_ or model3_.
