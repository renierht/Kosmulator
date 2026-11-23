# Model 4 burn-in review — 2026-11-17

A separate snapshot at 19730 steps was evaluated with the native convergence functions. The live chain and its metadata were not modified. All 120 walkers were retained; no likelihood, prior, model or dataset changes were made.

Burn 1000 fails length, ESS and split-Rhat. Burns 3000 and 4000 pass all four configured checks, including tau stability against the preceding 100-step checkpoint. At burn 3000: tau_max=19.2407, N_post/tau=869.509, pooled ESS=104341, Rhat_max=1.001176. These remain single-ensemble diagnostics, not independent-ensemble confirmation.

The maximum median shift between burns 3000 and 4000 is 0.001805 times the marginal 68-percent interval half-width. Full intervals and block medians are in report.json. The burn change is motivated by the previously identified initial settling through approximately step 2783, not by seeking a different posterior preference.

Future Model 4 and Model 5 EOS study launchers default to --burn 3000, including the queued fresh Model 4 warm-Lambda case. Explicit --burn overrides are supported and command construction was checked for all nine study cases at defaults and overrides 1000/4000. The lower-level run_analysis.py remains unchanged for historical reproducibility; direct use requires --burn 3000 to opt into this policy. When resuming the still-active historical run, use --burn 1000 unless performing a separately documented policy migration. Existing completed reports retain their original burn.

The current sampler still uses burn 1000; this report does not silently replace its native stopping record or regenerate final tables. Reprocessing with burn 3000 must consistently apply that burn to all posterior-dependent statistics and figures. No fresh sampling chain is required solely for a burn-in change.


## Finalization

The user authorized stopping and postprocessing on 2026-11-17. The original queue was stopped with the sampler, preserving all 19932 steps. The chain checksum is `34ffe721d48d00636543ccaa28f3116e9e667d09d8c5753995e694ae283841f7`. Native postprocessing uses a distinct output suffix `Model4_EOS_CC_v2_model4_free_de_burn3000_iw_cc`, burn 3000 and 2,031,840 retained samples. Its manifest guard verified scientific identity before reading the chain. The original sampling manifest and chain attributes remain unchanged.

The next fresh run, `Model4_EOS_CC_v2_model4_warm_lambda_iw_cc`, is launched independently via `run_logs/start_model4_warm_burn3000.sh` using eight workers, burn 3000, and the existing priors, datasets, walker count, move mixture and likelihood. It fixes w_DE=-1 and samples w_DM in [-0.15,0.15].

Postprocessing status: `run_logs/Model4_free_de_burn3000_analysis.log` and its exit-status sidecar. Warm-run status: `run_logs/Model4_warm_burn3000.log`. Initial failed-attempt status files are archived separately.
