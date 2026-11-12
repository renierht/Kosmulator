# IDE main-update release candidate — 2026-11-12

Local release branch: `codex/ide-release-20261112`.
Scope: the selected integration plan against upstream `34091d1`, not a wholesale
merge of main or a claim to include changes after that pinned revision.
Preservation baseline: `5e28be3`.

## Included

- Optional DE/snooker proposals (paper default), validated worker-context transport,
  run locks, provenance guards, read-only live chain inspection and IDE doctor.
- Dense background distance quadrature; explicit uncalibrated DESY5/Union3 offsets;
  DESY5 and Pantheon two-redshift geometry and Pantheon selection corrections.
- Explicit SN covariance loading, correlated CC updates and opt-in SPS covariance
  with explicit library/extrapolation choices.
- SN overlap guards, likelihood-consistent residual plots, comparison dataset
  fingerprints, and exported IC/offset/data-assumption explanations.
- Paper resumes now require source and data identity. Analysis verifies those plus
  priors/CLASS/physicality settings; changing burn is allowed. Dataset files are
  hashed in streaming chunks. Older incomplete manifests must not be relabelled.

## Physics and statistical conventions preserved

No differences in User_defined_modules.py, ide_background.py, rd_helpers.py or
Class/IDE_background source relative to the preservation snapshot. Ten IDE kernels,
conditions, free/fixed EOS switches and model-derived CLASS rd remain intact.
No perturbation physics, prior widening, YHe adjustment, convergence-rule change,
new Bayesian evidence estimator or silent DIC/IC redefinition was introduced.
Generic sampler default remains historical stretch; paper default is DE/snooker.

## Validation

Final suite: 112 tests / 730 subtests pass in activated Kosmulator_test.
Both real paper and Dovekie launcher --plan paths pass (69 source hashes and four
dataset inputs each), retaining 120 walkers, burn=1000, max_steps=100000.
Earlier full CLASS, independent PDF/ODE, save/resume, multiprocessing and native
visual checks remain recorded in docs/INTEGRATION_RELEASE_STATUS.md and validation/.
Protected-source diff and whitespace checks passed. Integration working tree is
committed; production dirty/untracked work is deliberately left untouched.

## Scientific limitations carried into the release

CC source-level overlap review is complete in docs/CC_CROSS_STUDY_REVIEW.md.
A full cross-study covariance cannot be reconstructed from summary H(z) rows.
Do not claim all 36 entries independent, or all SPS uncertainties included.
These limitations are now carried into generated comparison reports. This is an
explicit scope limitation, not a pending attempt to invent missing correlations.
MPI remains untested; local spawn multiprocessing is validated. Numerical checks
cover specified cases, not every point in a prior. Posterior significance remains
conditional on priors/modes and independent-ensemble confirmation.

## Migration and publication

This branch is prepared locally only. No push, production merge, or chain migration
has been performed. Keep Kosmulator_followup for existing-chain reproducibility.
Use the integration checkout for new corrected-likelihood runs after choosing the
dataset/systematic policy. Start fresh chains for all models in an updated
comparison; do not resume old likelihood chains or combine old and new IC tables.
Run selection, CC sensitivity analyses and new MCMC results are separate scientific
work, not outstanding implementation in this scoped release candidate.
