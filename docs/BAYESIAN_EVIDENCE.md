# Optional Bayesian evidence from completed chains

Kosmulator now provides a separate, read-only evidence CLI:

```bash
source /home/marcel/miniconda3/etc/profile.d/conda.sh
conda activate Kosmulator_test
cd /mnt/c/Users/Marcel/.codex/visualizations/2026/10/13/01a1358d-d46e-7da0-8eda-70bb634653cf/Kosmulator_IDE_main_integration
python reproducibility/SAIP2026_IDE/scripts/run_evidence.py \
  --config reproducibility/SAIP2026_IDE/configs/evidence_signswitch_legacy.json \
  --output Statistical_analysis_tables/Evidence_signswitch_new_review
```

Choose a **new** output directory; the CLI refuses to overwrite a report. It never samples, changes a chain, loads CLASS, or recomputes a likelihood. It uses one numerical thread. It is deliberately opt-in, separate from automatic tables: an evidence-prior audit must precede a new dataset/model combination. Existing AIC, DIC, BIC and significance tables retain their definitions.

The implementation is in `postprocessing/evidence.py`, outside the sampling source-identity set. No installed environment packages or sampling modules were changed. The study uses SciPy's cKDTree, already installed, rather than requiring the old upstream dependency stack. This is an independently implemented **MCEvidence-compatible equal-weight estimator**, not an assertion that the upstream package itself is installed.

## Estimator and validation reference

We implement the auto-neighbour MAP scale estimator of Heavens et al., *Marginal Likelihoods from Monte Carlo Markov Chains*, https://arxiv.org/abs/1704.03472. It matches https://github.com/yabebalFantaye/MCEvidence at commit `ade1dcb8b92309da991df85bf1bbe07f492cae0b` for unit-weight samples:

$$
\widehat Z_k=\frac{N}{Nk+1}\frac{|J|}{V_\pi}\sum_{i=1}^{N}L_i\,\frac{\pi^{d/2}r_{i,k}^d}{\Gamma(1+d/2)}.
$$

Here samples are whitened by their full covariance, $|J|$ is the transformation Jacobian, and $V_\pi$ is the normalized rectangular prior volume **in the original sampled coordinates**. Log-sum-exp and log determinants avoid underflow. All sampled nuisance parameters are included, including $M_{\rm abs}$; fixed EOS parameters and derived $r_d$ are excluded. This is not the harmonic-mean estimator or a conversion from an information criterion.

A pinned upstream copy and its MIT license are retained under `validation/evidence/upstream` solely for parity tests. The parity script supplies a validation-process-only alias for sklearn's moved DistanceMetric import; it makes no environment modification. Maximum upstream parity difference was $8.9\times10^{-16}$ for k=1,2,3.

## Explicit scope and prior convention

Current chain support: completed emcee HDF5 files with separately saved, aligned scalar log-likelihood blobs, metadata, and a passing completion record. Only committed iterations are read (some files preallocate to 100,000). The reader refuses missing blobs, invalid/sentinel retained states, mismatched models/parameter order, stale completion records, a burn different from the supplied convergence assessment, and saved posterior values inconsistent with zero log-prior inside the support. It does not silently treat log posterior as log likelihood.

Current prior support: explicitly audited normalized **uniform rectangular** priors. Priors must come from configurations/records, not posterior extrema. Nonuniform or conditioned/coupled priors are unsupported; CPL is explicitly refused until its $w_0+w_a<0$ normalized prior is implemented. Existing-chain metadata and an audit config are mandatory; there is no universal automatic-prior inference.

For this legacy IDE pilot, the real-background domain and CLASS failures enter through the likelihood. We use the full documented rectangular volume and preserve those exclusions as zero likelihood. We do **not** renormalize on solver success or estimate the volume of the physically accepted domain. Conditioning and renormalizing onto that domain would define a different prior and evidence. The high-H0 BBN-table domain is consequently retained. The estimates describe the implemented likelihood/domain, not a repaired solver or a claim of ideal physical-domain evidence.

Likelihoods retain their original saved normalization. Absolute logZ is therefore reported in saved likelihood units. Bayes factors are permitted only within an audited group with identical data hashes and compatible likelihood constants, CLASS calibration and rejection conventions. Different priors are allowed, but are distinct hypotheses and must be reported. The CLI checks the current dataset and provenance-file hashes before running. Legacy files lack contemporaneous hashes: the pilot uses an explicitly labeled reconstruction, not fabricated run-time provenance. Relevant historical source differences were checked: the likelihood additions concern DES-Dovekie/CPL, not the selected PantheonPS+DESI_DR2 paths; statistical-package functions match. The raw source diff is saved for review.

## Correlation, geometry and uncertainty checks

The default suite uses k=1,2,3; per-walker temporal strides ceil(tau_max), ceil(2 tau_max), ceil(5 tau_max); three reproducible randomized offsets; full and half-sized subsets; and first/second retained-time halves. Subsequent subsampling is without replacement. No iid bootstrap, duplicate jitter, walker deletion, or silent deduplication is used. Duplicates make a trial fail explicitly. The nominal result is the median k=1 estimate from three ceil(2 tau_max) full-time trials. The primary scalar does not override the detailed diagnostics.

Walkers within an ensemble are not independent chains. Residual within-walker lag correlations are reported. Neither thinning nor those correlations prove independence or mode completeness. Cross-ensemble evidence is not fabricated from a random split of one ensemble. Independent ensembles remain a desirable external validation.

The reported sensitivity envelope includes every valid tested k, stride, seed, subsample size and temporal block. Delta-logZ envelopes conservatively combine each model's and the reference's extreme values. **These ranges are not confidence or credible intervals and do not bound unknown systematic error.** We deliberately do not quote the estimator's iid formal precision as a total error bar. Any primary trial failure is flagged. >0.3 log-unit sensitivity and >1% k=1 neighbour balls crossing a rectangular prior boundary are flagged. These are review thresholds, not calibrated tests. Boundary-volume corrections and nonlinear reparameterizations are not applied silently.

## Validation

- `tests/test_evidence.py`: Gaussian integrals in 1,3,5 dimensions, correlated covariance, coordinate/Jacobian invariance, likelihood-offset and prior-volume shifts, HDF alignment and committed-iteration behavior, invalid/duplicate/singular inputs, incomplete provenance and mismatched data rejection.
- `validation/evidence/check_upstream.py`: direct pinned upstream numerical parity.
- `validation/evidence/validate_synthetic.py`: repeated analytic Gaussian tests, a curved unit-Jacobian posterior, a two-mode distribution, deliberately missed mode, duplicate rejection, and a bounded uniform density exposing boundary bias.
- The repeated 3D/5D Gaussian logZ errors were below 0.032. The curved example error was -0.0085. The complete two-mode example error was -0.0107; deleting one mode produced -0.712, close to the expected -log(2). This demonstrates why successful algebraic validation cannot certify a real chain's mode coverage.
- The uniform-box example correctly raises a boundary flag: its k=1 finite-sample error is about +0.037. Such bias is a limitation of the estimator, not silently corrected.

## Legacy sign-switching pilot

Final report: `Statistical_analysis_tables/Evidence_signswitch_legacy_v2/REPORT.md`; complete machine-readable diagnostics: `evidence.json`. v1 was the initial run; v2 adds saved-prior consistency and lag-correlation diagnostics. Both yield identical nominal values. All runs retain the original burn=1000 and all 120 walkers; the newer 3000 policy was **not** retroactively applied to these already-completed studies.

| Run | Sampled prior differences | lnZ (saved units) | Delta lnZ vs LCDM | Sensitivity envelope for Delta lnZ |
|---|---|---:|---:|---|
| LCDM | Om [0.1,0.5], H0 [60,90], Mabs [-20.5,-18] | -747.801 | 0 | reference |
| Model 9 | plus w [-2,-0.33], delta [-1,1] | -751.136 | -3.335 | [-3.401,-3.227] |
| Model 10 original | same rectangular priors as Model 9 | -748.342 | -0.541 | [-0.613,-0.283] |
| Model 10 wide | Om [0.1,0.7], delta [-2,2]; other bounds unchanged | -748.758 | -0.957 | [-1.095,-0.442] |

Positive Delta lnZ favours the alternative. These estimates do not favour the IDE models under the stated priors/domain. Better best-fit likelihood or AIC does not guarantee larger prior-averaged evidence. The Model 9 preference is stable across the tested settings, but remains provisional. For original Model 10 roughly 12% of primary k=1 neighbour balls cross the box, reflecting its prior boundary; the wide version has more pronounced sample-size dependence/curved geometry. Neither is a precision publication-ready Bayes factor. Broader priors changed both Omega_m and delta; the wide/original difference is not attributable solely to coupling width.

The full analysis remains background-only. Evidence does not validate IDE perturbation physics, numerical exclusions, independent-ensemble convergence, or sensitivity to alternative scientifically motivated priors.
