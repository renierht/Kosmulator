# Six matched EOS runs — prepared, not launched

Prepared 2026-11-12 on the integrated checkout. Interaction confirmed by Marcel:
Model 5, Q = 3 H delta rho_de, without a sign switch. Positive Q transfers DE
into DM. No existing chains or production checkout were changed.

## Cases

All cases sample Omega_m, H_0 and M_abs. Fixed parameters are removed from the
sample vector and counted accordingly in information criteria.

| Case / output component | Interaction | w_de | w_dm | k |
|---|---|---|---|---|
| lcdm | Q=0, delta fixed to zero | -1 | 0 | 3 |
| wcdm | Q=0, delta fixed to zero | free | 0 | 4 |
| warm_lambda | Q=0, delta fixed to zero | -1 | free | 4 |
| model5_lambda | Model 5, delta free | -1 | 0 | 4 |
| model5_free_de | Model 5, delta free | free | 0 | 5 |
| model5_warm_lambda | Model 5, delta free | -1 | free | 5 |

All six intentionally use the internal name Linear_IDE_5, including its
model-derived CLASS sound horizon. The first three set delta=0 exactly, rather
than sampling a very narrow coupling prior. Their physical labels are the case
names above, not six distinct interacting models. The native LCDM_v registry
entry does not support these free EOS parameters. Each case has a separate
output suffix and chain metadata. Reusing a fixed-coupling chain as a free one
is rejected. No analytic background or CLASS source equations were changed.

## Shared proposed settings

- Pantheon+SH0ES (PantheonPS), DESI DR2, and CC, as requested after reviewing
  arXiv:2512.05548. Use --include-cc for every case. Our CC compilation has
  36 points, not the paper's 32. Published pair correlations are included;
  optional additional Moresco SPS covariance remains disabled (CC_SYS_COV=None).
- Flat background, fixed Omega_b=0.048, N_eff=3.044, total neutrino mass 0.06 eV,
  three massive species, CLASS-derived r_d.
- Uniform priors Omega_m=[0.1,0.5], H_0=[60,90], M_abs=[-20.5,-18].
  Where sampled, w_de=[-2,-0.33], delta=[-1,1].
- **Selected free w_dm prior: [-0.15,0.15].** Explicitly requested by the user;
  retain CLASS-derived r_d and fixed Omega_b=0.048.
- Unrestricted background regime (`iw`): no optional positive-density,
  no-Big-Rip or doom-factor classification prior. Background/CLASS numerical
  validity still applies; this is not a perturbative stability claim. In
  particular, the interacting w_de=-1 cases are not assigned a doom-factor
  stability classification.
- 120 walkers, emcee DE/snooker mixture, burn=1000 fixed steps, thin=1,
  full-chain storage, cap=100000 steps, existing convergence policy unchanged:
  post-burn length/tau >=100, ESS >=2000, Rhat <=1.01, relative tau tolerance
  0.05, one consecutive passing check. Within-ensemble diagnostics do not
  replace independent-ensemble confirmation.
- Eight worker processes per run, one thread per numerical library; six cases
  sequential, not six competing pools. Initial jitter is 0.10 of prior width.
- New chains are necessary: this uses the newly integrated likelihood and
  distance policies. Existing sign-switching results remain separate.
- Existing fixed-Omega_b BBN-table restrictions are unchanged. No YHe fix or
  hidden H_0-prior adjustment is introduced.

The initial suggestion to reserve CC for a secondary analysis is superseded
by the user's request to include it. Omit --include-cc only for a separate
matched no-CC comparison; the CC runs have an additional _cc suffix.

## Interpretation and prior screen

This is a constant background EOS study. Allowing negative w_dm is a
phenomenological extension, not a direct thermal warm-particle mass constraint.
See https://arxiv.org/abs/1605.00649 for the generalized-dark-matter distinction
between background EOS and perturbation parameters.

For delta=0, rho_dm scales as a^[-3(1+w_dm)]. Above w_dm=1/3 it grows faster
than radiation toward the past. At Omega_m=0.3, H_0=70, w_de=-1 and delta=0
or 0.02, the 16-point CLASS screen found finite r_d at w_dm=-0.5,-0.1,0,0.1,
but radiation-initialization failures at 0.2,0.3,0.34,0.5. Failures below 1/3
may depend on the numerical start epoch; they are not a measured physical
exclusion. The screen is not a complete joint-prior domain audit. A broad
nominal rectangle with failed CLASS points would acquire an implicit cutoff.
No initialization settings have been changed to conceal this issue.

Recommendation: start the general-EOS comparison with [-0.1,0.1], subject to
user agreement, and inspect posterior boundaries before deciding on a broader
sensitivity run. The launcher's explicit prior argument prevents silently
making this choice. At the tested uncoupled point, r_d=144.962 Mpc for w_dm=0
and 64.974 Mpc for w_dm=0.1: even this proposed interval has a large effect.

## Validation

- 119 tests and 730 subtests passed in Kosmulator_test.
- Six parameter configurations checked: k=3,4,4,4,5,5; fixed values recorded,
  free/fixed chain compatibility protected; r_d not sampled.
- Six representative cases checked against independent solve_ivp integration
  of the conservation equations at 151 scale factors each. Maximum scaled
  density difference 1.25e-10; all six model-derived sound horizons finite.
- Cold delta=0 Model 5 versus native LCDM CLASS r_d relative difference 1.94e-9.
- Warm r_d changes explicitly verified, so the free parameter reaches CLASS.
- Plan-only driver exercised for all six cases. No MCMC chain was created.

Reports: validation/model5_eos_preflight.json and
validation/model5_wdm_domain.json. The latter stores actual CLASS exceptions.

## Commands for the selected prior

In WSL:

```bash
source /home/marcel/miniconda3/etc/profile.d/conda.sh
conda activate Kosmulator_test
source /mnt/c/Users/Marcel/PhD/Kosmulator_test/Clik/code/plc_3.0/plc-3.1/bin/clik_profile.sh
cd /mnt/c/Users/Marcel/.codex/visualizations/2026/10/13/01a1358d-d46e-7da0-8eda-70bb634653cf/Kosmulator_IDE_main_integration
```

The following is a **plan only**, using the selected
prior; it does not run sampling:

```bash
python reproducibility/SAIP2026_IDE/scripts/run_model5_eos_study.py \
  --run-name Model5_EOS_CC_v1 --w-dm-prior -0.15 0.15 --num-cores 8 --include-cc
```

After agreement, adding `--sample` runs the six cases sequentially. Select one
case using `--case NAME`; use the same settings/name and `--resume` to resume
it. Commands fail fast if a case fails. If a sequence stops after completed
cases, select the interrupted/remaining cases explicitly instead of rerunning
completed cases accidentally. Output suffix is `Model5_EOS_v1_CASE_iw`, with
internal model folder `Linear_IDE_5`, under MCMC_Chains, Plots/Saved_plots and
the native statistical-table output tree. Do not edit source between starting
and resuming: the launcher checks source and dataset fingerprints.

## Comparison paper: arXiv:2512.05548v2

The paper samples r_d, rather than deriving it with CLASS, and samples Omega_b.
It uses DESY5 rather than Pantheon+SH0ES, and includes TD and GRB. Its broad
w_dm prior therefore does not validate the same range with our CLASS-derived
sound horizon. Its coupling convention translates to beta=-3 delta.
Neither TD nor GRB has a dataset/likelihood route in this checkout.

The earlier proposal [-0.3,0.3] was not adopted: the measured CLASS failures at
0.2 and 0.3 lie within it. Before launching, choose between retaining the
CLASS-derived calibration and investigating its domain, or preparing a separate
free-r_d, late-time analysis. Switching to sampled r_d changes the inference,
parameter counts and scientific question; it is not a numerical workaround.

## Final user selection and endpoint screen (2026-11-12)

The user chose to continue the existing CLASS-derived-r_d analysis, add CC to
all six cases, and use w_dm=[-0.15,0.15]. No production sampling has started.
The six full plans, including source/data fingerprints, are stored in
reproducibility/SAIP2026_IDE/configs/model5_eos_cc_v1_plans.json.

validation/model5_wdm015_domain.json records 81 CLASS checks: Omega_m=0.1,0.3,0.5;
H0=60,75,90; delta=0,-0.1,0.1; w_dm=-0.15,0,0.15. 52 were finite.
For H0=60 and 75, 52/54 were finite, including every uncoupled case. The two
failures were Omega_m=0.1, delta=0.1, w_dm=0.15. An independent analytic
background screen with standard approximate radiation found negative E^2 in
both cases, consistent with an invalid expansion history, not merely evidence
for a numerical EOS cutoff. This diagnostic is not a replacement for CLASS.
All H0=90 checks failed: 25 BBN-table errors, three background step-size errors
(over all H0), and one radiation-initialization error account for all 29 failures.
The high-H0 BBN restriction already affects the cold controls. No helium,
initialization, baryon-density or likelihood settings were changed.

These are representative tests, not full joint-prior certification; delta
endpoints +/-1 and the free-DE range were not exhaustively screened here.
The nominal rectangular priors remain subject to existing background/CLASS
validity. Monitor rejected proposals and posterior boundaries in production.
The selected prior supersedes the earlier recommendations and pending-choice
language preserved above as investigation history.


## Future burn-in policy (2026-11-17)

The study launcher now defaults to `--burn 3000`; explicit overrides remain available. Historical runs retain their recorded burn. Use `--burn 1000` when resuming an existing 1000-burn study without a policy migration. The queued fresh Model 4 warm-DM case inherits 3000. See `MODEL4_BURN_REVIEW.md` for the snapshot sensitivity check. A fixed burn is not a guarantee of adequate settling for every future run.
