# Model 4 matched EOS study — 2026-11-14

User authorized three new runs for Q=3H delta rho_dm, with the same setup as
the completed Model 5 study. They run sequentially: cold_lambda (w=-1, w_dm=0),
cold_free_de (w free, w_dm=0), warm_lambda (w=-1, w_dm free). Delta is free in
all three. Reuse the completed non-interacting baselines; no duplicate baselines.

Datasets: PantheonPS + DESI_DR2 + CC. Priors: Omega_m=[0.1,0.5], H_0=[60,90],
M_abs=[-20.5,-18], delta=[-1,1], w=[-2,-0.33] when free, w_dm=[-0.15,0.15]
when free. Fixed Omega_b=0.048; model-derived CLASS r_d, Neff=3.044, total
neutrino mass 0.06 eV, three massive species. Unrestricted iw background
regime; existing numerical domains and high-H0 BBN limitations retained.
120 walkers, burn=1000, thin=1, cap=100000, DE/snooker 80/20, eight workers,
one numerical thread per worker, unchanged convergence policy.

20 targeted tests passed; six representative numerical background checks passed
with maximum scaled density error 1.25e-10 and finite CLASS r_d. This does not
certify every joint-prior point. No background equations or likelihood code
changed; the launcher now accepts Linear_IDE_4. The old Model 5 launcher is
preserved as run_logs/run_analysis_model5_completed_snapshot.py. Existing chain
manifests/results have not been relabelled. Source hashes change with the
launcher extension; analyse old chains with their original source identity.

Launcher: reproducibility/SAIP2026_IDE/scripts/run_model4_eos_study.py
Plans: reproducibility/SAIP2026_IDE/configs/model4_eos_cc_v1_plans.json
Log: run_logs/Model4_EOS_CC_v1.log
Detached session: model4_eos_cc_v1

## Comparison with arXiv:2512.05548v2

The paper uses beta=-3 delta relative to our convention. Completed Model 5
post-burn chains give corr(delta,w)=-0.994327 and corr(delta,w_dm)=+0.906781,
matching its stated correlation signs after conversion. Our fixed-EOS case
has no established coupling preference; freeing w does not establish w>-1.
The warm interacting case mildly prefers positive delta and positive w_dm.
Thus the degeneracy pattern agrees qualitatively, but not all claimed detection
or model-selection trends. Datasets and calibration differ (we derive r_d with
CLASS and fix Omega_b; the paper samples both).

The paper uses DIC=2*mean(chi2)-chi2_min. Applying that formula to our saved
polished minima and mean deviances gives deltas relative to LCDM of +1.159
(fixed EOS), +1.404 (free DE), -2.370 (free DM). These are a separately computed
comparison, not a replacement for native posterior-mean DIC. In particular the
free-DE native delta DIC=-9.472 with negative p_D should not be interpreted as
strong support. Paper: https://arxiv.org/pdf/2512.05548


## Future burn-in policy (2026-11-17)

The study launcher now defaults to `--burn 3000`; explicit overrides remain available. Historical runs retain their recorded burn. Use `--burn 1000` when resuming an existing 1000-burn study without a policy migration. The queued fresh Model 4 warm-DM case inherits 3000. See `MODEL4_BURN_REVIEW.md` for the snapshot sensitivity check. A fixed burn is not a guarantee of adequate settling for every future run.
