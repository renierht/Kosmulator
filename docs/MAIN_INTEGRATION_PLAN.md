# IDE integration sections

Integration worktree Kosmulator_IDE_main_integration, branch
codex/ide-main-integration-20261105. Preserved source snapshot 5e28be3.
Reviewed upstream main 34091d1.

## Section 1 — sampling infrastructure and proposal validation
Completed. Paper launcher uses run locks and complete settings manifests.
DE/snooker is now the paper launcher's default, backed by two 2000-step
comparisons showing approximately 2.3–2.4x provisional effective samples/hour.
Use --emcee-moves stretch for a run whose complete manifest records stretch.
Old complete manifests missing a move field are interpreted as stretch.
Legacy chains without complete manifests continue through the original launcher.
The generic Kosmulator.py default stays stretch until its resume provenance is
covered. The new default applies to the integration checkout's paper launcher.
Production followup remains the source for reproducing earlier results.

## Section 2 — live inspection and generic launcher safeguards
Port --peek to copy snapshots, obey saved burn, display actual saved iteration
counts across resumes, and write separate output files.
Extend lock/provenance coverage to the generic launcher. Verify concurrent
inspection, exclusion of competing writers and legacy metadata handling.
Finish with a checkpoint before starting the next section.

## Section 3 — worker performance and environment diagnostics
Review immutable worker context, serialization, thread limits and logging.
Adapt doctor to the shared IDE CLASS backend and source-hash requirements.
Check identical numerical likelihoods and measure runtime/memory.
Checkpoint with measured results; avoid changing scientific settings.

## Section 4 — distance, supernova and CC review
Compare upstream distances against independent quadrature.
Review SN intercept marginalization, covariance and heliocentric redshifts.
Audit revised CC rows, correlations and optional stellar-population covariance.
Separate likelihood-changing corrections from equivalent implementations.
Document implications for existing results and fresh chains before adoption.

## Section 5 — reporting and scientific validation
Adapt citations and overlap reporting, review DIC/WAIC and retain significance
qualification and independent-ensemble status.
Run warm/cold IDE background, EOS-switch and CLASS regressions.
Verify PDF solutions, model conditions and model-derived sound horizons.
Review release diff and reproducibility, then prepare publication branch updates.

## Session budget
Work on one section or a bounded subsection per session, checkpoint its tests
and pending work, and stop there. Actual credit consumption is unavailable to
the agent; these are scope boundaries rather than guaranteed five-hour budgets.
No automatic production run is part of this integration plan.

Section 4 progress (2026-11-10): dense distance quadrature and old-chain resume/
analysis guards integrated and tested. See checkpoint for numerical errors and
fresh-chain requirements. SN/CC review has identified remaining dependencies;
no SN likelihood convention or CC data changes adopted yet.
Section 4c: DESY5/Union3 explicit offset projection enabled and validated with
real loader data. Dedicated Dovekie and sampled Pantheon M_abs unchanged.
Redshift/covariance review and CC integration still pending; see checkpoint.
Section 4d: DESY5 zHD/zHEL geometry and explicit DESY5/Union3 covariance loading
implemented, real-file quadratics validated. CC source/covariance review remains
next; no new production chains launched.
Section 4e: verified CC row corrections and published correlations integrated;
CC data hashes protect generic resumes. Optional Moresco SPS covariance and
sample-overlap assessment remain pending (docs/CC_INTEGRATION_AUDIT.md).
Section 4f: optional Moresco SPS covariance audited and integrated, off by default;
explicit library and endpoint policies. Next: Pantheon/reporting consistency and
sample-overlap review. Production dataset choice remains separate.
