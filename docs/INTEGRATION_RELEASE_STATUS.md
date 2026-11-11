# Integration release status — 2026-11-11

Worktree: Kosmulator_IDE_main_integration; branch codex/ide-main-integration-20261105.
Production Kosmulator_followup and saved chains are unchanged. This is a local
integration candidate, not a published replacement or a new posterior analysis.

## Validated

- Default-off optional CC SPS covariance, explicit covariance/redshift corrections,
  overlap guards and resume/analysis provenance documented in the audit files.
- Protected IDE analytic background, conditions, rd helpers and CLASS physics source
  show no diff against preservation snapshot 5e28be3.
- 110 tests and 730 subtests pass; later focused reporting export checks also pass.
- PDF expression checks: 100 cases / 8100 points; independent ODE/future checks.
- Full CLASS: 20 cold/warm cases / 800000 points and ten rd routes pass.
- Five native SN figures generated from real data at illustrative parameters and
  inspected. No production plots or statistical tables regenerated.
- Activated Kosmulator_test now has a matching shared CLASS cache. Doctor reports
  ready; signature 1de1fa3ca79dfa1aa40aea398de4d213222fb92e402962c6c5572267e8a7584c.
  The earlier activated-prefix cache issue is resolved in this checkout.
- Activated serial/worker save+resume validation: bitwise equal 16-walker chains
  through 24+4 steps. Four-worker CLASS benchmark likelihood differences zero;
  original 1.108/1.123 s versus cached 0.940/0.938 s per 12 evaluations. Short
  workload only; not a full production performance or convergence claim.

## Environment check

From WSL, in this integration checkout:

```bash
source /home/marcel/miniconda3/etc/profile.d/conda.sh
conda activate Kosmulator_test
source /mnt/c/Users/Marcel/PhD/Kosmulator_test/Clik/code/plc_3.0/plc-3.1/bin/clik_profile.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python -m Kosmulator_main.ide_doctor
```

Caches are local generated artifacts and are not included in commits. Other
checkouts/machines may require an explicit build. MPI launch remains untested
because no launcher is installed. Spawn multiprocessing was tested.

## Decisions and work before production release

- Select the intended dataset and systematic-covariance settings. Keep overlapping
  SN releases as alternative fits, not independent factors in one likelihood.
- CC cross-study overlap/systematic independence remains unaudited. Do not claim
  the compilation is fully independent or fully systematic-complete.
- IC convention is explicitly exported: k counts sampled parameters and excludes
  analytically removed offsets. No separate profiled-offset IC convention or
  new Bayesian evidence calculation was implemented. Numerical IC values are
  unchanged by this reporting stage.
- Review the complete integration diff for deployment and synchronize a release
  branch deliberately. No GitHub push or production migration occurred.
- Corrected distances/Pantheon/DESY5/CC likelihoods require fresh chains under
  the finalized configuration. Old chains remain with their original checkout;
  do not edit their manifests to bypass compatibility guards.

The build and backend checks do not establish independent-ensemble convergence
or validate an entire prior volume. Existing significance qualifications remain.
