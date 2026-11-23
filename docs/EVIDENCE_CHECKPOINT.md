# Evidence checkpoint — 2026-11-18

Nonlinear three-regime calculations are saved in Statistical_analysis_tables/Evidence_nonlinear_regimes_exploratory/REPORT.md and results.json. Reproduction script: validation/evidence/nonlinear_regimes/analyse.py. These remain exploratory; see the report for convergence and boundary limitations. No historical chain was changed.

Sign-switching pilot: Statistical_analysis_tables/Evidence_signswitch_legacy_v2. Implementation: postprocessing/evidence.py; documentation: docs/BAYESIAN_EVIDENCE.md. Existing validation includes upstream estimator parity and synthetic distributions. A bounded nonlinear burn-sensitivity check (1000/2000/3000) is now saved alongside the report. Outstanding: curved-boundary validation and independent-ensemble confirmation. Do not present these pilot estimates as finalized Bayes factors.
