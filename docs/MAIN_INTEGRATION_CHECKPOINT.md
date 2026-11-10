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

## Section 2 concurrency validation
Three run-lock tests passed: competing process rejected, reacquisition after
normal release, reacquisition after owner termination, plus default-directory
agreement (the first test covers normal release and contention).
Corrected the generic lock pre-parser default from empty suffix to Test_run,
matching the actual CLI parser.
Live emcee writer test passed 20 snapshot reads during 100 appended steps.
Writer completed successfully at 120 steps; no transient copy errors occurred.
Results validation/live_peek_results.json; repeatable driver
validation/check_live_peek.py. Copying a live HDF5 is still non-atomic, so this
test does not guarantee all filesystem/load combinations.
Remaining: native live plots, complete generic provenance and MPI workflow tests.
Windows lock behavior remains untested. Production code remains separate.

## Section 2 snapshot plotting
The emcee inspection CLI accepts --plot and --plot-samples (default 10000).
Uses GetDist and Kosmulator greek_Symbols labels, in a lightweight monitoring
renderer rather than the full postprocessing table/derived-rd pipeline.
Deterministic without-replacement draw selection affects display only.
All reported percentiles and split Rhat use all post-burn snapshot samples.
Each invocation writes a unique directory under Plots/Saved_plots/Peek with
corner.png and snapshot.json recording source, burn, inspected steps and seed.
Successfully rendered and visually inspected DES Model 9 from its completed
21200-step production chain. Production plots and chains were not overwritten.
GetDist chatter is routed to stderr to preserve JSON stdout for scripting.
This renderer does not reproduce full publication plot settings or infer prior
bounds for legacy metadata; use normal postprocessing for publication figures.
Generic provenance, MPI workflow tests and optional Zeus inspection remain pending.

## Section 2 generic provenance
Direct Kosmulator.py now records normalized generic settings under the same lock,
including priors, model/EOS/regime switches, serializable uppercase constants,
CLI configuration and hashes of model, core Python and IDE CLASS source/header
files. Resume rejects changes; worker counts and resume flags are operational.
The snapshot is taken before MCMC_setup mutates runtime constants.
This is deliberately conservative: even non-physics edits to hashed core source
require provenance review. Dataset file contents are not yet hashed.
Generic and paper manifests are distinguished to prevent cross-launcher reuse.
Legacy chains without a manifest remain protected by refusal.
Three manifest tests and three lock tests passed. Generic settings regression
covers tuple/list normalization, changed baryon constant, changed physics source,
and operational worker-count changes.
MPI rank-zero validation now precedes a barrier; errors abort a multi-rank job.
mpi4py package is installed but no mpiexec was found on PATH or in Kosmulator_test.
Actual MPI smoke validation remains unperformed; do not claim MPI readiness.

## Section 3 worker transport checkpoint
Adapted upstream per-group pickle transport to pool_context.py. Both fresh and
resumed emcee constructors use the small callable when a pool exists; serial
calls retain original arguments. Each worker caches only its current group.
Context files are unique and cleaned on normal return or exceptions. No cleanup
of another process's files is attempted. Hard kills can leave private context
files behind. These pickle files are internal trusted artifacts.
Spawn unit tests passed for equivalence, sub-kilobyte callable size, normal and
exception cleanup, and serial identity. Real Model 9 CC likelihood agreement
across eight points was exact (max difference 0), callable 2335 -> 189 bytes.
All nine warm-background regressions passed.
Repeatable driver validation/check_ide_pool_context.py.
This does not yet establish DES/CLASS runtime gains, resumed HDF5 workflow
equivalence or MPI shared-filesystem behavior. Those benchmarks remain pending
before deploying this transport to production.

## Section 3 real transport validation
validation/validate_pool_transport.py completed under Kosmulator_test.
Original versus cached transport produced bitwise identical Model 9 CC chains
through 24 saved steps and resume to 28 with 16 walkers and DE/snooker.
This exercises both real M.run_mcmc constructor paths and verifies context cleanup.
DES-Dovekie + DESI DR2, model-derived CLASS rd, 4 spawn workers:
original 1.1293 / 1.1757 seconds; cached 0.9762 / 0.9638 seconds per 12 evaluations.
All corresponding posterior/likelihood pairs matched exactly. Both orderings
tested with fresh worker pools; startup/warm-up excluded.
Callable payload 26,577,117 -> 189 bytes.
This suggests ~14–18% less elapsed time for this small workload, not a guaranteed
full-chain acceleration. Resident data remains replicated once per worker;
this optimization reduces serialization, not necessarily peak worker memory.
Results validation/pool_transport_results.json.
MPI and hard-kill context-file cleanup remain unvalidated. Environment doctor
adaptation and longer production-shaped performance validation remain pending.

## Section 3 IDE environment doctor
Added kosmulator-doctor --ide via existing console entry point, also
python -m Kosmulator_main.ide_doctor. Reuses existing dependency checks, verifies
all ten IDE model names route to IDE_background, and checks the exact production
source/ABI/toolchain cache signature without building or loading CLASS.
Reports thread variables and MPI launcher availability. Current adapter imports
CLIK even for background calculations, so that dependency is checked, while
Planck data directories are not required. Binary loadability and likelihood
correctness are not established by this read-only cache-presence check.

Validation found that the benchmark binary was built with CONDA_PREFIX empty.
With CONDA_PREFIX=/home/marcel/miniconda3/envs/Kosmulator_test, the signature
differs and the doctor reports not ready. Restoring the benchmark environment
passes with cache 093eaa42. This is conservative cache identity, not evidence
of incompatible physics. Build under the intended activated environment before
a future production run; no automatic build was performed by this diagnostic.
validation/ide_environment_report.json records the activated-environment finding.
MPI launcher remains absent. Generic legacy doctor behavior remains available.

## Section 4a — corrected distance quadrature and chain safeguards (2026-11-10)
Ported only Comoving_distance_vectorized from upstream 34091d1. The integration
mesh now includes a uniform grid with dz <= 0.01 plus the data redshifts and zero.
Independent scipy adaptive quadrature at seven BAO redshifts, one representative
parameter vector each for LCDM, CPL, IDE 9 and IDE 10, gives maximum relative
errors 0.00569–0.00622 before versus 2.05e-6–2.84e-6 after. This is a numerical
likelihood correction, NOT a performance-only or scientifically equivalent change.
The audit script extracts the old implementation from preserved commit 5e28be3;
validation/distance_update_audit.json records the measurements. These checks do
not quantify posterior shifts or validate every point in the prior.

The paper and generic launchers now record distance_policy=dense-grid-dz0.01-v1.
Paper resumes reject old or absent policies; generic source hashes also protect
resumes. Paper analysis checks each input chain's nearest ancestor manifest and
rejects unmatched policies before likelihood-based postprocessing. Legacy chains
remain analyzable using their original production checkout, or read-only peek.
Do NOT relabel their manifests to bypass this check. Fresh corrected-distance
chains are needed for consistent updated LCDM/CPL/IDE model comparisons.
No production files, chains, IDE density equations, CLASS source, rd policy,
priors, datasets, or stopping criteria were modified.

Validation: 22 targeted tests plus 83 subtests passed (distance analytic reference,
ordering/duplicates/negative/zero/invalid redshifts, manifest guards, warm
background and CPL regressions). Existing CRLF in utils.py is preserved.

## Section 4b — supernova/CC audit checkpoint, integration still pending
DES_Dovekie has a separate full-precision-matrix likelihood in Kosmulator_MCMC,
projects out the magnitude intercept and retains its normalization term; its
luminosity distance already uses zHD for integration and zHEL for the prefactor.
Do not replace it with upstream's generic DESY5 route. These are different samples.
Current generic SN chi-square subtracts a sampled M_abs or defaults it to zero;
its comment saying "Marginalize" is misleading. Upstream adds explicit analytic
marginalization and further JLA/covariance machinery, so cherry-picking just the
function would omit dependencies. Current Pantheon paths use zHD in the prefactor;
loader/redshift conventions must be audited jointly before changing them.

Upstream CC changes include z .44497 -> .4497 and several error bars, notably
z .47: 23 -> 49.6. It adds correlation/SPS treatment. None adopted yet: check
original data sources, row identity and covariance assembly first. This changes
the likelihood and must be versioned with dataset provenance. The next bounded
section should address explicit uncalibrated SN offset handling with loader and
configuration tests, followed by the sourced CC covariance audit. Do not start
production sampling until the intended combined likelihood is finalized.

## Section 4c — explicit generic SN offset projection (2026-11-10)
Added upstream's explicit marginalise_offset policy to Calc_Generic_SNe_chi,
enabled by the DESY5 and Union3 loader. Their supplied distance moduli no longer
act as an absolute calibration. Full inverse covariance and diagonal-error paths
both project out the weighted constant offset. The implementation centers the
residual before forming its quadratic to avoid cancellation in A-B^2/E.
The data_is_distance_modulus flag prevents an unrelated sampled M_abs from
entering these distance-modulus datasets. Legacy callers without either flag keep
their sampled-M_abs behavior. No parameter list changes were required: these two
datasets already request H_0 without M_abs, while H_0 can still be constrained by
other datasets or background quantities in a combined analysis.

Normalization convention matches upstream: return A-B^2/E and omit the fixed
log(E/2pi) constant. This is the profiled chi-square and has the same cosmological
posterior shape as flat-offset marginalization for fixed covariance; it is NOT a
normalized Bayesian evidence calculation. DES_Dovekie's existing normalization
and dedicated likelihood are unchanged. PantheonP and PantheonPS retain their
sampled M_abs treatment; no SH0ES calibrator changes were made.

Validation: 23 tests and 83 subtests passed, including independent scalar
minimization with correlated/diagonal synthetic data, invariance under magnitude
shifts, legacy M_abs behavior, actual DESY5/Union3 loading, dimensions and offset
invariance using their real precision matrices. Warm background, CPL and resume
manifest regressions pass. Generic launcher source hashes reject continuation
of pre-change chains. The paper launcher does not select DESY5/Union3.

This is a likelihood-changing correction requiring fresh DESY5/Union3 chains.
It does not constitute full validation of those likelihoods: existing redshift
selection, covariance heuristics and missing-covariance fallback remain to be
reviewed. Dataset-specific offset conventions and effective nuisance-parameter
counting must also be documented consistently in future IC reporting. No new
production runs started. Next: finish DESY5 zHD/zHEL loader and covariance audit,
then review the upstream CC rows/correlations against their original sources.

## Section 4d — SN redshifts and explicit covariance loading (2026-11-10)
DESY5 CSV loading now prioritizes zHD and retains zHEL as z_hel. The generic
SN likelihood uses (1+z_hel) D_c(zHD) when supplied. DES_Dovekie's dedicated
zhel branch is unchanged; Pantheon tags have no new metadata and are unchanged.
Whitespace Union3 keeps its existing single-redshift binned convention.
Malformed CSV/whitespace rows raise rather than silently drop rows or partially
append arrays, protecting covariance alignment. Union3 zero placeholder dmb
values remain zero and are not used to augment its full covariance.

New sn_covariance.py loads the declared N-prefixed covariance explicitly:
DESY5 C_total = C_sys + diag(MUERR_FINAL^2); Union3 C_total = C_file.
It checks dimensions, finiteness and symmetry, and uses Cholesky solve to obtain
precision. Missing/malformed/non-PD covariance fails closed: no diagonal-only
or pseudoinverse fallback and no covariance-versus-precision diagonal heuristic.
No dataset files modified. Config.py's mixed line endings were normalized to
CRLF during editing; an optional attempt to restore unchanged line endings was
rejected by automatic review, so that cosmetic cleanup was omitted.

26 targeted tests and 83 subtests passed. Real-data audit confirmed 1829 DESY5
rows and 22 Union3 bins with exactly the explicit covariance recipes. Covariance
quadratics agree with independent linear solves at relative 2.22e-16 / 8.66e-15.
247 DESY5 rows have zHD != zCMB. At LCDM Omega_m=.3,H0=70, switching from the
old single-zCMB geometry to zHD/zHEL changes predicted distance modulus by up to
0.093685 mag. Both sides use the corrected dense distance mesh, isolating the
redshift correction. This does not quantify posterior shifts. Reproducible audit:
validation/audit_sn_loading.py and validation/sn_loading_audit.json.

Likelihood change: fresh DESY5 chains required; generic source-hash resume guard
rejects old code provenance. Production remains untouched. No claim of external
release-file identity or full upstream integration is made by these local checks.
Remaining: sourced CC row/covariance review, Pantheon redshift convention audit,
and reporting/offset nuisance-count consistency before production release.
