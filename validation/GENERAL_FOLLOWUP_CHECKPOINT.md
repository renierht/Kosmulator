# General follow-up integration checkpoint (2026-10-14)

Branch: codex/general-followup-integration. Based on aba7a79; reviewed upstream origin/fR_followup_fixes at 8de59c3. Production warm-DM worktree and running chains are separate.

## Integrated

- CLASS forced extension relink (a59f421): rebuild_ext --force once, avoiding stale linkage to rebuilt libclass.a and removing a duplicate invocation.
- CLASS stale local artifact rejection/source ordering (89d2bf3): reject non-IDE local binaries older than build inputs; stable directory traversal. Cython .pyx/.pxd/.pxi hashing and generated-source exclusion already existed. Preserve the stricter IDE rule requiring a matching source-hash cache artifact.
- Confirmed-empty emcee recovery (823b53e): skip loading/resuming zero-step files, initialize normally, retain cross-engine fallback. Unknown/unreadable files are not classified as empty. Unlike upstream, foreign nonempty HDF5 files without an mcmc group are not classified as empty either.
- Zeus posterior/sample alignment (a0275ec, adapted): append two-dimensional log_prob_chain alongside step-major samples, including resumed segments and final forced flush. Legacy files with missing posterior history remain eligible for explicit reevaluation rather than fabricated history. Model comparison verifies exact sample alignment and subtracts the evaluated log prior; it does not label log posterior as likelihood. Ambiguous legacy flattened logs are not guessed/reordered.

These changes do not intentionally change the model, data, priors, proposal defaults, burn-in or convergence policy. Cache correctness can expose that an old artifact was stale; scientific equivalence of an actually stale binary is not guaranteed. Active production files have not received this batch.

## Validation

Kosmulator_test: all 61 unittest tests pass (56 existing, five focused regression tests). Checks cover Cython hash sensitivity/generated-file invariance, forced relink, empty versus unreadable/foreign files, Zeus append/resume alignment and prior subtraction, and legacy missing history. Actual Zeus smoke workflow passes for Models 9 and 10, saving/loading 24 steps and resuming to 28 with temporary CC-only chains. This is persistence verification, not a convergence claim or production CLASS benchmark.

## Deliberately not included / further review

- Distance integration correction b22555f changes numerical likelihood values. Dataset selections, redshifts, SN marginalization, radiation/neutrino and BBN changes likewise require separate scientific review and potentially fresh chains. Current BBN-table prior limitation remains documented in PERFORMANCE_CHECKPOINT.md.
- WAIC seed patch 8d42187 targets a WAIC implementation absent from this checkout; not applicable as a direct patch.
- Immutable worker-context transport, optional DE/snooker moves, cached toolchain/in-process shortcuts, read-only peek CLI and citation/overlap reporting remain candidates for later integration. No claim that every upstream general update has been ported.

## Concurrent production runs

Production aba7a79 adds --only-model ide/lcdm. Use the existing warm-DM worktree for the ongoing experiment; separate run names isolate outputs. Eight workers for each job is a reasonable initial split on 16 physical cores, based on observed memory headroom; aggregate throughput under two simultaneous jobs has not been benchmarked. The original two-model launcher still queues Model 9 after LCDM, so avoid a redundant third job when that transition occurs. Do not run two writers on the same chain. Model9_parallel_20261014 is the suggested separate run name; use --resume to continue that same run later.
