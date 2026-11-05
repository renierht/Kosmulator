# Convergence integration checkpoint — 14 October 2026

Worktree: `Kosmulator_warm_dm`, branch `codex/warm-dm-ten-models`.
Warm-DM implementation baseline commit: `0ed51ac`. Original checkout is preserved.

## Integrated

Selective port from `origin/fR_followup_fixes` at `8de59c3`:
shared emcee/Zeus monitor, post-burn diagnostics, 100 tau length,
approximate pooled ESS >= 2000, tau_max relative change < 0.05,
ordinary walker split-Rhat < 1.01, one successful check, checks every 100 steps,
and no extra burn buffer by default. Fixed burn-in semantics are retained.
No fractional 30% burn policy was added. The production launcher defaults to
the paper's fixed 1000 burn steps.

The diagnostic functions live in `Kosmulator_main/convergence.py`.
Undefined diagnostics in any parameter fail the rule (a defensive fix over
upstream's nanargmax reduction); finite diagnostic formulas are retained.
Reports are saved beside each chain as `.convergence.json` and
`.convergence.txt`, plus the convergence plot. The HDF converged flag reflects
the actual verdict. Manual fixed-length emcee runs are not marked converged.
Explicit resume bypasses an old converged flag and uses actual backend totals.
Zeus includes the saved prefix in resumed diagnostics and flushes its final block.

`run_analysis.py` now accepts `--ide-model SignSwitch_IDE_1`, `--burn`, and
`--tau-consecutive`. It runs fresh LCDM plus Model 9, with free w/fixed w_dm=0,
the paper priors, DESI_DR2 + PantheonPS, and paper CLASS sound-horizon settings.
Model 9's unclassified stability regime cannot be selected via `--regime siw`.

## Validation

56 unit tests passed, including real emcee HDF resume/ceiling tests and
constant-parameter rejection. Short Zeus save/load/resume smoke checks passed.
The Model 9 launcher's `--plan` output was verified. The additional real-CC
Model 9 emcee production-path check passed (120 steps, resumed to 160),
including plots, report settings, burn handling and a correct nonconverged
verdict at the ceiling: `validation/smoke_convergence_run.py`.
These are workflow checks, not evidence that a scientific posterior converged.

## Full run (from this worktree, in WSL Kosmulator_test)

```bash
MPLBACKEND=Agg python -u reproducibility/SAIP2026_IDE/scripts/run_analysis.py \
  --regime iw --ide-model SignSwitch_IDE_1 \
  --run-name Model9_unrestricted_20261014 --max-steps 100000 --burn 1000
```

Add `--resume` to continue the same run; increase `--max-steps` if the ceiling
is reached without convergence. Use a different run name for a fresh experiment.
The launcher runs LCDM first, then Model 9. It does not start four independent
ensembles; it matches the requested upstream single-ensemble policy.

## Other upstream changes: review candidates, not merged

Priority follow-ups (commit subjects inspected; individual ports need review):

- `a0275ec`: Zeus likelihood storage aligned with step-major samples, important
  for statistical analysis of Zeus chains. Production launch here uses emcee.
- `823b53e`: robust recovery of empty chain files. Full loader coverage remains
  a follow-up; fresh production output names avoid this path.
- `a59f421`, `89d2bf3`, `406c0fc`: CLASS build/cache correctness and avoiding
  repeated hashes. Preserve this branch's shared IDE backend/cache protection.
- `8bfcc8c`, `613417a`, `3368078`: batched SN likelihoods, vectorized emcee,
  and pooled initial walker evaluation; benchmark before adoption.
- `7af6b17`, `0fc51ff`, `5fe5cba`: optional scaled DE/snooker moves. Not required
  to match convergence criteria; current IDE proposal defaults retained.
- `32c146d`, `d39c735`: read-only live-chain peek, useful for long runs.
- Citation manifests and data-overlap warnings: useful after checking that
  their dataset definitions match this branch.

Do not merge the entire upstream branch blindly. It also changes Pantheon+
sample/redshift handling, SN offset marginalization, late-time radiation,
neutrino defaults, BAO/CC/growth datasets, CMB likelihoods and statistics.
Those change scientific results or require separate validation against the
paper setup. Keep them outside this production run until explicitly adopted.

No remote push or production sampling was performed during this checkpoint.
