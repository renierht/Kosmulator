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

Second batch records run_settings.json for every new paper-launcher sampling run.
Existing manifests are checked for datasets, model, priors, CLASS settings, regime,
EOS switches, walkers, burn, thinning and convergence settings. Operational
max_steps changes remain possible. Writes use an atomic replacement under the lock.
Legacy chains without complete provenance are refused by this isolated launcher;
use their original production launcher until explicit provenance validation.
Analysis mode does not overwrite sampling manifests.
Two focused regression tests passed, covering incompatible settings, operational
changes, manifest preservation and refusal to relabel legacy chains.
The warm-background regression suite was run; see /tmp/ide-physics-regression.log.
Source comparison confirms the protected IDE files still match 5e28be3.

Third batch ports the upstream sampler_moves module and wires both fresh and
resumed emcee constructors. The paper launcher exposes --emcee-moves stretch or
de_snooker. Stretch remains the default. Proposal choice is recorded and checked
in run manifests; missing values in earlier complete manifests mean stretch.
Only constants and proposal wiring change; IDE equations and CLASS density
sources remain identical to the preservation snapshot.
A seeded 48-walker, three-dimensional unit Gaussian smoke test (2500 steps,
500 discarded) recovered means within 0.024 and standard deviations within
0.010 for both options. Results are in validation/upstream_moves_smoke.json.
MCMC imports in Kosmulator_test and both emcee constructors expose moves.
This is not an IDE performance benchmark or full distributional proof.
The two run-manifest regressions also passed. No production chain was started.
