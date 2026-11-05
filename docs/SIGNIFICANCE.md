# Significance and posterior preferences

Native postprocessing now exports two distinct summaries. Neither is Bayesian evidence.

## Model comparison: approximate likelihood-ratio significance

`model_comparison.csv/json/tex` and printed statistics include `LRT_sigma`.
`likelihood_ratio_significance.csv` supplies degrees of freedom, p-value and status.
The improvement is chi2_LCDM - chi2_alternative; p=chi2.sf(improvement, df),
Z=norm.isf(p/2), following the convention of DESI DR2 Eq. (22):
https://arxiv.org/html/2503.14738v3#S7
No absolute value is taken. A worse alternative fit is flagged, not converted to significance.
Degrees of freedom are extra sampled parameters, never rounded DIC p_D.

Enabled candidate pairs are LCDM_v against w0waCDM, SignSwitch_IDE_1 and
SignSwitch_IDE_2. Other IDE models still get posterior reports, but require a
separate nesting audit before this model-level statistic is enabled.

Automatic checks require matching dataset groups/counts and common parameter
priors, compatible fixed parameters, successful optimizer runs and interior
optima, an interior allowed null, and unrestricted IDE conditions. The actual
full likelihood is compared at three LCDM parameter points (absolute chi2
agreement within 0.001). Both sides of each extra parameter are evaluated near
the null; a finite-difference background Jacobian must have full extra-parameter
rank. Nonzero evaluated log priors are conservatively rejected because MAP/ML
equivalence has not been established. Null checks restore all global switches.

These are necessary numerical checks, not proof of the Wilks sampling
approximation, global optimality or global identifiability. Every reported value
remains labelled approximate. Boundary cases, failed checks and missing LCDM
references report unavailable, never a fabricated zero. A single-model launch
cannot supply a model-comparison significance until it is paired with LCDM.
A zero improvement with a valid comparison maps to zero sigma; an underflowed
p-value is not turned into infinite sigma.

## Separate posterior report

`posterior_preferences.md` and `posterior_preferences.json` contain checks for
sampled delta, delta_dm, delta_de (reference zero), w and w0 (reference -1),
wa and w_dm (reference zero). Fixed parameters are labelled fixed.
Reported intervals are equal-tailed 68%, 95%, 99.7%. Probabilities are computed
from all supplied retained samples; no sign-based filtering is performed.

The scalar Gaussian-tail equivalent is Phi^-1(1-min(Pbelow,Pabove)), with the
favored direction recorded. It is not a frequentist p-value or joint
model-rejection statistic. Sigma is suppressed when the reference is on or
outside a prior boundary, local prior probes fail, an IDE regime is restricted,
there is an atom at the reference, either tail is unresolved, or fewer than 20
raw tail draws occur. Tail counts alone do not establish precision.

When the exact retained 3D saved chain matches the input samples, contiguous
100/200/500-step batch means of the ensemble-wide event fraction estimate Monte
Carlo uncertainty while retaining simultaneous walker dependence. Otherwise
uncertainty is explicitly unavailable; samples are never assumed iid.
A saved convergence flag is reported if available; absent/false flags make the
summaries provisional. Even a true flag does not establish independent-run
agreement or absence of missed modes. CPL marginal w0/wa preferences cannot be
combined as independent evidence against LCDM.

All results remain conditional on the configured likelihood and numerical
domain, including the existing CLASS BBN-table limitation. Postprocessing does
not modify any priors, chains, sampler settings or likelihood evaluations.

## Validation checkpoint

The development suite passed 75 tests. Actual model-derived CLASS likelihood
null checks at three LCDM parameter points gave maximum absolute chi2
differences 3.625e-5 (Model 9), 2.132e-5 (Model 10), and zero (CPL).
All three local extra-parameter background Jacobians had rank 2.
Model 10 and CPL checks used synthetic interior fit metadata to exercise
nesting, not completed fits or measured significances. Model 9 used its saved
polished fit and LCDM reference. Results: validation/significance_checks.json.

Completed Model 9 reports were regenerated in the original checkout under
Statistical_analysis_tables/Model9_fresh_vs_LCDM_significance. The joint
approximate significance is 2.9154789; posterior delta tail is 3.0547225 and
w tail is 1.9761799. No source chain was modified.
