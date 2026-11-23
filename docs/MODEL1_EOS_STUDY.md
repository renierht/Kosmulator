# Model 1 no-CC EOS study

Interaction: Q = 3 H (delta_dm rho_dm + delta_de rho_de), with independent uniform delta_dm and delta_de priors [-1, 1]. Sign reversal is allowed, not imposed; opposite coupling signs alone do not establish a crossing within the observed redshift interval.

Datasets: Pantheon+SH0ES and DESI DR2; no CC. Shared no-CC LCDM benchmark is Model4_EOS_noCC_v1_lcdm_iw.

Cases: cold_lambda (w=-1, w_dm=0); cold_free_de (w free in [-2,-0.33], w_dm=0); warm_lambda (w=-1, w_dm free in [-0.15,0.15]). Other priors: Omega_m [0.1,0.5], H_0 [60,90], M_abs [-20.5,-18]. Sampled dimensions: 5, 6, 6.

120 walkers; eight workers; DE/snooker; burn 3000; full chain saving; cap 100000. Existing convergence policy: post-burn length >=100 tau, ESS >=2000, split Rhat <=1.01, tau stability 5%, one passing check. Initial scatter is 0.002 of prior width, matching the sum/difference launcher. CLASS-derived sound horizon and fixed baryon/neutrino settings are inherited unchanged from paper.json. Existing unrestricted-regime and numerical-domain restrictions remain.

New isolated drivers: reproducibility/SAIP2026_IDE/scripts/run_model1_eos_analysis.py and run_model1_eos_study.py. Existing launchers and physics sources are unchanged. Output prefix Model1_EOS_noCC_v1.

Validation: validation/check_model1_eos_study.py and model1_eos_preflight.json check all three runtime parameter configurations, both coupling routes, 12 independent ODE/CLASS evaluations, and the cold zero-coupling LCDM limit. Maximum scaled density error was 1.25e-10. These bounded checks do not assert every point in the broad prior is numerically valid.

Queued after the final Model5 no-CC case, before the older Model2/3 CC queue. The queue stops if any sampling or postprocessing command fails. See run_logs/model1_noCC_queue.json.
