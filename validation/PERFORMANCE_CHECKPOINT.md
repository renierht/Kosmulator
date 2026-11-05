# Paper-run performance audit, 14 October 2026

The active serial run (PID 2966 at audit time) was not stopped, read through
HDF5, or modified. All benchmark outputs are separate. Scientific settings,
model code, datasets, priors, walker count, burn, cap and convergence remain unchanged.

## Changes

The paper launcher hard-coded num_cores=1. It now accepts --num-cores (default 1)
and sets OpenMP/BLAS thread counts to one before scientific imports.
The existing multiprocessing pool uses spawn. Runtime CLASS-r_d calibration
and IDE switches are now explicitly passed to its initializer. Without this,
workers would reimport DERIVE_RD_WITH_MODEL_CLASS=False, changing the likelihood.
This correction makes the new parallel path agree with the original serial path.

CLASS evaluations use a fresh Class instance with struct_cleanup/empty in
finally. Python caches and loaded extensions are process-local. The shared
binary cache has a build lock; the parent prepares backends. Benchmarks required
existing binaries and made no scientific source changes. This is process
parallelism, not concurrent threads sharing a CLASS instance.

## Short benchmark

32 identical parameter vectors per model and worker count, real DESI_DR2 +
PantheonPS likelihood, production covariance Cholesky, paper CLASS settings.
Single-threaded libraries. Each timed point clears the r_d cache, so worker
warm-up cannot create artificial cache-hit speedups. Spawn warm-up excluded
from timing (approximately 3 seconds for 8 workers, 4 seconds for 16).
The original serial production job remained active, so these are short,
contended fixed-work measurements, not full-chain timing guarantees.

| Model | Serial | 8 workers | 16 workers |
|---|---:|---:|---:|
| LCDM | 7.318 s | 1.849 s | 1.827 s |
| Model 9 | 8.244 s | 1.882 s | 1.818 s |

Maximum absolute log-likelihood/posterior difference: zero at all tested points.
Recommend 8 workers: about 4–4.4x faster, with little additional benefit from 16.
Measured worker maximum-RSS counters were at most 436 MiB for 8 workers and
478 MiB for 16 (ru_maxrss may include inherited high-water marks). A conservative
sum is roughly 3.4/7.5 GiB plus parent/other processes. Sampled free-memory
checks showed over 12 GiB available and zero swap; those snapshots are not a
continuous total-peak measurement. No memory exhaustion occurred.

The serialized likelihood context is 21.03 MiB, largely the SN covariance.
It is sent repeatedly with pool task chunks. This can limit scaling; caching
immutable context once per worker is a future performance-only improvement.
No broad likelihood refactor was attempted to conserve credits.
The r_d LRU key excludes M_abs and includes the background settings; ordinary
new proposals need fresh CLASS runs. The active-backend check avoids source
rehashing for every point. Failed strict DESI r_d evaluation can be retried
by the existing fallback plumbing; it never substitutes a fixed r_d, but this
is redundant work on failed points and remains unchanged in this patch.

## BBN helium limitation (not repaired by this performance patch)

Both bundled sBBN_2025 tables end at omega_b=0.03287. Fixed Omega_b=0.048 gives
omega_b from 0.01728 to 0.03888 over H0=60..90. The table upper bound corresponds
to H0=82.75213995. Thus about 24.16% of the nominal flat H0 interval fails
this calculation. Probes for BOTH models passed at H0=60,82 and failed at
83,87.71355,90 with the helium-table error. At 87.71355, omega_b=0.0369296.
The DESI failure path returns a huge chi-square penalty, effectively zero
likelihood. This is an unintended computational restriction, not evidence
that those H0 values are physically excluded.

No YHe fixing, extrapolation, prior narrowing, or BBN table replacement was
performed. A validated BBN treatment covering the full nominal range is a
separate scientific correction; use fresh chains for that corrected target.
Resuming the performance-only update preserves the CURRENT effective target,
including this limitation. Do not report it as a fully evaluated 60..90 prior.

## Stop / resume

The production emcee HDF backend stores completed iterations, samples,
log-probabilities and likelihood blobs. The resume path loads the last saved
coordinates and runs to the TOTAL requested cap, discarding burn only once.
It need not reproduce the same random trajectory. Existing regression tests
cover saved-chain resume and actual backend iteration counting (56 tests pass).

When ready, press Ctrl+C ONCE in the original run terminal, wait for Python
to exit and the shell prompt to return. Do not force-kill or start a second
writer. After exit, back up the complete run directory and metadata; only
then restart with the same run name and --resume --num-cores 8. The most
recent incomplete iteration may not be retained; a clean keyboard interrupt
avoids the risks of abruptly terminating an HDF write. Inspect the backend
read-only after exit if any HDF error is reported; never delete it to restart.

```bash
MPLBACKEND=Agg python -u reproducibility/SAIP2026_IDE/scripts/run_analysis.py \
  --regime iw --ide-model SignSwitch_IDE_1 \
  --run-name Model9_unrestricted_20261014 --max-steps 100000 --burn 1000 \
  --resume --num-cores 8
```

Measured values and BBN exception messages: pool_benchmark_results.json.
Reproducible bounded script: benchmark_paper_pool.py.
