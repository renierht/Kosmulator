# Kosmulator IDE v3 publication snapshot — 23 November 2026

This publication combines the November integration history through `60e80dc`
with the subsequent working-source EOS studies and Bayesian-evidence tooling
from `Kosmulator_IDE_main_integration`.

The active sampler uses that integration worktree. This separate publication
copy does not replace it or change its running queues, chains or manifests.

Included updates:

- Run locking, provenance validation and resume safeguards.
- Validated DE/snooker proposals and cached worker argument transport.
- Numerical distance, supernova likelihood/plotting and correlated CC updates.
- Fixed-parameter overrides and matched EOS studies for Models 1–5,
  with optional CC and zero-coupling comparisons where supported.
- Read-only Bayesian evidence estimation, explicit prior/domain assumptions,
  synthetic tests and parity checks against a pinned MCEvidence reference.
- Study documentation, source scripts and compact validation records.

The established ten IDE backgrounds, shared CLASS backend and physical-domain
policies remain those used by the integration worktree. This is not a full
merge of unrelated upstream changes.

Validation before publication: **161 tests and 730 subtests passed** in WSL
`Kosmulator_test`, with the production CLIK profile activated and numerical
threads limited to one. Copied source files were checked byte-for-byte against
the running worktree. Publishing does not establish posterior convergence for
unfinished studies or validate IDE perturbations.

New runtime queue files/logs, HDF5 chain snapshots, generated study PNG/PDF
figures and bytecode are excluded. Existing committed historical validation
artifacts are retained. Original local files are preserved. Some validation
scripts/configurations reference local historical chains by absolute path;
those paths must be adapted when reproducing the checks elsewhere. Chains
and installed CLASS/CLIK binaries are not bundled with this source release.
