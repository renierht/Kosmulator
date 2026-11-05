# Main integration checkpoint

Preserved IDE snapshot 5e28be3abfac1f4124a3afa51d02db9fc40c44c1.
Upstream reviewed at 34091d1. Integration branch codex/ide-main-integration-20261105.

The snapshot includes tracked edits, untracked analysis scripts/reports and the
normally ignored DES-Dovekie STAT+SYS precision matrix. Ignored chains, binaries,
logs and generated plots remain in the production followup worktree.
This is a source snapshot, not a separate backup of the chains.

First integration batch adapts upstream run locking for the paper launcher.
The lock spans manifest checks, sampling and postprocessing. Plan-only invocation
does not acquire it. Linux lock errors fail closed. The generic Kosmulator.py entry
point is not covered yet, and older launchers do not cooperate with this lock.
Validated independent-process contention and subsequent reacquisition in WSL.
Launcher Model 9 DES-Dovekie plan succeeds. Windows locking remains untested.

Next batches pending
- Generic launcher locking and run provenance/resume validation.
- Adapt live inspection to historical IDE metadata and explicit burn policy.
- Worker transport and optional sampler moves, with targeted benchmarks.
- Review doctor/citations and extend model/backend recognition.
- Independently assess numerical distances, SN likelihoods and CC covariance.
- Retain IDE DIC definitions and significance caveats during reporting integration.

Protect User_defined_modules.py, ide_background.py, Class/IDE_background,
rd_helpers.py, CLASS routing, EOS policy, physical conditions and existing
convergence thresholds. A full upstream merge would remove IDE modules.
Production chains must keep using their original source until equivalence is
established. Likelihood-changing updates need a separately validated fresh run.
