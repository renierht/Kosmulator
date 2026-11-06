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

## Usage-limit handoff — benchmark in progress

User requested checkpointing at 15% remaining usage.
Active bounded benchmark script validation/benchmark_ide_moves.py runs 100 steps
per proposal, stretch then de_snooker, 120 walkers, 8 fork workers, threads=1.
Uses the same final saved DES Model 9 ensemble, opened read-only from the
production followup worktree, and the actual DES-Dovekie + DESI DR2 likelihood.
CLASS was compiled locally in the isolated worktree. Initial configuration needed
the normal placeholder r_d prior before Config removes r_d in derived-CLASS mode.
Production chain files are never opened for writing.

Process was launched in persistent exec session 84018, Python PID 17969.
Check process existence rather than trusting the old PID after reboot.
Log is /tmp/ide-moves-benchmark.log.
Result validation/ide_moves_benchmark.json is written after each segment.
Only one result means the second segment is still pending or failed.
Benchmark ends automatically after both segments.
Do not launch a duplicate while it is active.

100 steps is deliberately a pilot. Its estimated tau/ESS is unreliable relative
to the production tau ~188; do not claim a convergence-speed improvement from it.
Next action is inspect log/results and compare timing, acceptance and jump size.
Before recommending new moves, perform a sufficiently long mixing comparison if
needed and obtain agreement of numerical likelihoods against the original setup.
Then continue pending live inspection, worker transport, environment tooling and
likelihood-change review batches. No full production run has been launched.

Benchmark finished and reviewed. Stretch 552.73 s / 100 steps, acceptance 0.4689;
DE/snooker 469.14 s, acceptance 0.1821. DE/snooker used 15.1% less elapsed time.
Mean squared jumps (including rejections) were 2.68x larger for delta and 3.07x
for H0; approximately 3.16x and 3.61x per unit time. These are movement diagnostics,
not independent-sample throughput or demonstrated convergence speed.
Only one short segment per method, with stretch first, so timing can contain
order/cache effects. Tau estimates near 8-10 from 100 steps are not trustworthy
given the full production chain tau ~188. Keep stretch as the default until a
longer replicated benchmark and posterior consistency check can be completed.
No sampler benchmark process remained at review.

## Overnight review, 2026-11-06

All four segments completed; no benchmark process remains.
Each used 120 walkers, 2000 steps, 8 workers, the same saved final ensemble.
Round 1 stretch / DE-snooker: 9867 / 9553 seconds; tau_max 88.58 / 39.56;
estimated ESS/hour 989 / 2286.
Round 2 DE-snooker / stretch: 9560 / 10107 seconds; tau_max 40.92 / 94.60;
estimated ESS/hour 2209 / 904.
DE/snooker therefore improves provisional effective-sample throughput 2.31x
and 2.44x. Acceptance 18.3-18.6% versus stretch 46.3-46.7%.
All stored samples and log probabilities are finite.
H0 medians: stretch 70.049,70.152; DE/snooker 70.535,70.481.
Omega_m medians agree within 0.0001; w within 0.0013; delta within 0.0005.
These short continuations share the initial ensemble and are not independent
confirmation of all posterior modes or full-policy convergence. Stretch segment
lengths span only ~21-23 estimated tau, DE/snooker ~49-51. Treat ESS as provisional.
Recommend DE/snooker as an opt-in candidate for a fresh, fully converged run;
retain stretch as default until the user's production choice is made.
The BBN domain and prior boundaries remain unchanged.
Review data validation/ide_moves_overnight_review.json; original metrics
validation/ide_moves_overnight.json; NPZ segment samples remain local and ignored.
Prior usage-limit approval failure blocked the handoff commit, not the launched
benchmark. The configurable script and review are now being checkpointed.

## Section 2 partial checkpoint
Added python -m Kosmulator_main.peek --chain PATH [--burn N].
Copies the HDF5 first, respects saved iteration count (not allocated capacity),
excludes last potentially incomplete step, reports percentiles and split Rhat.
Burn comes from run_settings.json, or must be explicit for legacy chains.
Synthetic preallocated-chain test verifies burn, final-step exclusion and byte
preservation of source. Real completed Model 9 inspected at 21200 steps with
burn=1000. Concurrent writer stress testing remains pending. A snapshot copy can
fail during concurrent HDF5 metadata updates; retry later rather than modifying
the source. This command is for emcee only; live plots/Zeus support are pending.
Direct Kosmulator.py now locks its output directory, rank zero for MPI.
Paper launcher already owns its lock when calling main directly.
Compiled launcher successfully; generic workflow/MPI tests and complete generic
provenance integration remain pending. Do not deploy this checkpoint yet.
