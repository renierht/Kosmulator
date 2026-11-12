# CC cross-study review — 2026-11-12

## Findings and implementation decision

The review does not establish a full 36-point joint covariance. Keep the supplied
compilation for upstream compatibility, with explicit assumptions in postprocessing.
Do not infer independence from different authors, different central redshifts, or
absence of a published cross-covariance. No unsupported off-diagonal terms or
unannounced sample cuts have been added.

| Studies | Evidence and conclusion |
|---|---|
| Wang 2026 vs Loubser DESI 2025 | Wang uses z<0.2; Loubser's fitted ranges are 0.3–1.0. Inference: these fitted galaxy samples do not overlap in redshift. This does not establish independence of calibration/model systematics. |
| Zhang 2014 vs Moresco 2012 | Both use SDSS DR7 LRG data. Shared parent catalogue means potential overlap; the H(z) summary files do not provide galaxy membership needed for an exact crossmatch. |
| Moresco 2012 vs Stern 2010 | Moresco explicitly includes a Stern sample. Treat cross-study covariance as unknown, not demonstrably zero. |
| Simon 2005 / Stern compilation | Stern's table republishes historical measurements. The current file lists those historical rows once, but unique H(z) tuples alone do not prove galaxy-level independence. |
| Moresco D4000 rows | Optional common SPS terms are implemented only on the designated rows. They do not complete cross-study statistical covariance. |
| CC vs DESI BAO | Different observables do not by themselves prove statistical independence when parent surveys overlap. No CC–BAO joint covariance is supplied here. |

## Primary sources

- [Wang et al. 2026, Sections II.2 and IV.3](https://arxiv.org/html/2601.07345):
  redshift selection and explicit omission of SPS-model uncertainty. The quoted
  71.33 +/- 4.20 measurement is confirmed; its error is not a complete SPS budget.
- [Loubser 2025, Sections 3.1–3.3](https://arxiv.org/html/2511.02730v1):
  fitted redshift intervals and the published within-study correlations.
- [Zhang et al., Section 2](https://arxiv.org/html/1207.4541): SDSS DR7 LRG selection.
- [Moresco et al. 2012, Section 2](https://arxiv.org/html/1201.3609):
  SDSS and Stern samples among the contributing catalogues.
- [Stern et al. 2010](https://arxiv.org/html/0907.3149): historical compilation and
  two new determinations relative to prior work.

## Release scope

This closes the source-level integration audit with an explicit limitation, not
with a claim of complete independence. The loader carries these limitations as
likelihood_notes, exported to comparison JSON and comparison_conventions.md.
The numerical covariance and existing priors remain unchanged by this review.

For a publication using CC, select and document a specific compilation and perform
leave-study-out/systematic-policy sensitivity analyses. These are new scientific
runs, separate from completing this software integration. Exact crossmatching and
construction of missing cross-covariances require galaxy catalogues or additional
author-provided information; they cannot be recovered from CC.dat's three columns.
