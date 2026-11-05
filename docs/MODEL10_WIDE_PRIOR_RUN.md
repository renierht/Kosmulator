# Model 10 wider-prior sensitivity run

Launch with `bash reproducibility/SAIP2026_IDE/scripts/run_model10_wide_priors.sh` from Kosmulator_followup in WSL. Append `--plan` for a dry run; append `--resume` only to continue this new run with identical settings.

New output suffix: Model10_wide_priors_v2_iw. Uniform rectangular priors: delta [-2,2], Omega_m [0.1,0.7]. The existing Model 10 global real-branch restriction still applies, including delta <= 2/3 and its parameter-dependent lower bound. This is a prior-sensitivity run, not a likelihood change; wider rectangular bounds do not remove physical/numerical domain restrictions. A wider interval cannot guarantee prior independence; inspect new boundary occupancy before final interpretation.

All other settings preserved: DESI DR2 + Pantheon+SH0ES, unrestricted regime, w free, w_dm=0, 120 walkers, burn=1000, thin=1, full-chain saving, 100000 cap, native convergence policy, eight processes, one numerical-library thread per process. Initial scatter remains 1% of prior width. Never append the previous Model 10 chain. Prior-run settings are recorded and mismatched resumes rejected.

The Omega_m widening avoids replacing the lower delta boundary with a correlated upper Omega_m boundary. This differs from the original benchmark prior, so do not blindly reuse a common-prior nesting/significance certification. Re-evaluate comparison eligibility after the new run; the old tables remain unchanged.

Validation: launcher dry-run verified; original defaults preserved; malformed prior bounds rejected; Python/shell syntax checks; five background/CLASS evaluations spanning delta=-0.65 to -2, with independent conservation integrations. See validation/model10_wide_prior_checks.json. These are representative domain checks, not a full likelihood or sampler convergence test. The run has not been started.
