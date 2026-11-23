# Model 4 fixed-EOS diagnostic — 2026-11-15

The running study was not interrupted or modified. Diagnostics used a private
snapshot at step 15271 and independent processes; live CLASS precision, likelihood,
priors, burn and convergence policy remain unchanged.

## Findings

One current outlier (walker 30) has log probability -1845.844 versus -749.370
for the best last-step walker. It started at delta=0.20282 and logp=-5425.57;
its current delta=0.22651, H0=60.148. It moved 61 times in the last 1000 steps,
so it is trapped in a poor region rather than literally frozen. Walker 97 was
also poor after burn and escaped later. Full post-burn split Rhat for delta is
3.337. The last 4000 steps of the other 119 walkers give Rhat=1.0033--1.0040;
this diagnostic exclusion is NOT a corrected posterior or valid stopping result.

The original initial scatter is 0.10 of prior widths: delta scatter=0.2,
versus a main-region posterior width about 0.0008. Tests of the existing
--init-jitter-frac 0.002 option (delta scatter 0.004), with 120 positions around
the original printed center [0.307,0,70,-19.35], gave no invalid sentinel
likelihoods. Initial logp ranged -922.748 to -752.585, median -765.764.
This is a starting-cloud test, not a converged new MCMC run.

Saved main/outlier likelihoods reproduced exactly under original settings.
A straight bridge between the two regions passes through worse likelihoods;
it is not proof that every possible path has a barrier. Three scaled local
Nelder-Mead probes were each capped at 100 evaluations. Their best logp values
were -749.259 (main start), -1099.802 (outlier start), -761.183 (dispersed start).
None reached optimizer convergence within the cap; these are diagnostic probes,
not certified maxima or an exhaustive multimodality search. The outlier region
has not shown a competitive fit.

## CLASS experiments (separate process only)

At representative main/outlier points and bridges, a_ini=1e-18 versus the
original 1e-14 changed the main log likelihood by about 2e-7; it did not remove
the poor-likelihood region. Direct r_d checks at delta=-0.15,-0.05,-0.002,+0.2
agreed to relative differences below 3e-8. At a_ini=1e-24 all tested likelihoods
failed, with direct checks reporting a z=0 table-endpoint roundoff mismatch.
Do not adopt that precision change.

Radiation-initialization failures were reproduced at delta=-0.2,-0.25,-0.3,
-0.35,-0.5 (Omega_m=.301, H0=72). Starting at 1e-18 did not make these pass.
For delta=-0.2, Omega_r improved from .993479 to .999835 but still missed the
1e-4 tolerance. Earlier radiation domination is not impossible for every such
point, so failed initialization cannot be declared a physical exclusion.
The numerical domain limitation remains; this bounded check did not solve it.

## Recommendation

Preserve the original chain. Stop its sequence cleanly before starting a fresh
fixed-EOS chain under Model4_EOS_CC_v2_model4_lambda, with
--init-jitter-frac 0.002. Keep all 120 walkers, original priors, datasets,
CLASS settings, burn=1000, max_steps=100000 and convergence policy unchanged.
No deletion of walkers and no reinterpretation of the old run as converged.
The proposed command was checked with --plan; no fresh production run was
started by this diagnostic. Existing queued cases must be managed when stopping
the original supervisor, so that a concurrent duplicate is not accidentally
launched. Independent dispersed-ensemble confirmation remains advisable.

Reports and scripts are in validation/model4_diagnostic/. The snapshot is
retained separately from the original chain. No scientific source files changed.

## Authorized restart

On 2026-11-15 the user approved stopping the original sequence and launching
the fresh chain. Old sampling stopped at 15360 saved steps; all 120 final
states and log probabilities are finite. The original chain and logs remain
in place, with SHA256 in run_logs/Model4_EOS_CC_v1_preservation.json.
The new detached session is model4_eos_cc_v2; log run_logs/Model4_EOS_CC_v2.log.
The fixed-EOS case uses initial jitter 0.002. The free-DE and free-DM cases
remain queued with their original settings (including jitter 0.10), under v2
output names. No new CLASS precision settings or priors were applied.
