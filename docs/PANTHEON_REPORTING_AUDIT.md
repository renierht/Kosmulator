# Pantheon and reporting audit — 2026-11-11

Reference: [Cobaya PantheonPlus implementation](https://raw.githubusercontent.com/CobayaSampler/cobaya/master/cobaya/likelihoods/sn/pantheonplus.py)
and [release data](https://github.com/PantheonPlusSH0ES/DataRelease/tree/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR).
Unanchored PantheonP now includes every zHD>0.01 row and clears calibrator flags:
Cepheid-host objects above the cut contribute only their supernova observations.
PantheonPS keeps its existing zHD>0.01 OR calibrator selection and Cepheid-distance
substitution. Both use zHD in the comoving integral and zHEL in the luminosity
prefactor. The identical mask slices the data and covariance indices.

M_abs is still sampled for both tags. This retains the existing finite-prior
nuisance convention and is NOT an implementation of Cobaya's optional analytic
flat-offset marginalization. In unanchored SN-only analysis H0/M_abs are degenerate;
combined datasets and finite priors must be considered when interpreting results.

The paper launcher now records likelihood_policy=pantheon-zhd-zhel-v1 and checks
it for sampling resumes and external-chain analysis. Generic launch provenance
also records it and hashes source. Fresh Pantheon chains are required. Previous
chains remain reproducible through their original production checkout.

## Plot and statistics consistency

SN point predictions and residual plots now apply the same zHEL factors and
Cepheid substitutions. DESY5/Union3/Dovekie data offsets are fitted with their
precision weights at the plotted parameter vector; supplied full covariance
provides the point error bars. The smooth curve remains the cosmological distance
curve, not a curve through individual Cepheid distances. Error bars are marginal
per-point errors, not whitened independent residuals. Corner plots are unaffected.

Model_comparison and significance evaluate the shared MCMC likelihood; no second
SN/CC chi-square implementation was found in those paths. Existing saved/current
likelihood checks remain in place. Parameter-count convention remains k = sampled
parameter count. Analytically removed offsets are not automatically reintroduced
into k. This convention needs explicit publication documentation or a separate
profile-likelihood IC option; no silent IC redefinition was made here.

## Validation and remaining release work

Full suite: 107 tests, 730 subtests passed in Kosmulator_test. Tests cover real
Pantheon masks/covariance dimensions, independent chi-square solves, marginalized
plot residuals, Cepheid point predictions and legacy likelihood-policy rejection.
Explicit PDF validation: 100 cases / 8100 points, max scaled density error
5.343e-15; 27 future-evolution cases and finite endpoint passed. Real Model 9 CC
spawn likelihoods match serial exactly (callable 23157 versus 189 bytes).
Protected IDE source and CLASS source remain unchanged from preservation snapshot.

Still pending: sample-level cross-dataset overlap audit (do not assume SN releases
or CC studies independent), complete report metadata/IC nuisance convention,
visual inspection of regenerated production-style data/model figures, activated
Conda CLASS cache readiness/full-backend smoke, final release diff and deployment.
No production sampling, chain migration, or GitHub push occurred in this section.

## Local overlap and full CLASS follow-up

validation/audit_sn_overlap.py counts exact (case-normalized CID, IDSURVEY)
identifier matches, using each Pantheon tag's actual selection. Results are in
validation/sn_overlap_audit.json. DESY5/Dovekie share 1718 keys; PantheonP or PS
shares 332 with DESY5 and 337 with Dovekie. PantheonP/PS share 1590 keys (1473
unique CID strings). These are identifier matches, not independent row counts;
aliases and cross-release identifiers can leave additional overlap undetected.
The loader rejects pairs of these four releases within one likelihood group.
Separate observation groups remain allowed. No joint cross-release covariance
was constructed. Other sample overlaps, including CC studies, remain unaudited.

Full cached CLASS validation also passed: 20 cold/warm cases, 800000 background
points, maximum H relative error 1.67e-15 and scaled density error 7.40e-15.
All ten model-derived rd routes were finite. This used the existing matching
cache/environment without a rebuild; activated-CONDA_PREFIX cache identity still
needs reconciliation before release. Full suite after guard: 108 tests and
730 subtests pass. No production chains or GitHub publication changed.
