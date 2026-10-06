PROVISIONAL TABLE 5 FOR CO-AUTHOR REVIEW

Only the DIC plug-in convention is changed to the MLE/mode-based version.
N=1670 is the loaded measurement count: 1657 Pantheon+SH0ES plus 13 DESI DR2.
Reference: LCDM_v with model-derived CLASS r_d, k=3; IDE regimes k=5.
Burn=1000, thin=1, retaining every finite aligned sample.
Retained sample counts: LCDM 444000; iw 900000; +iw 756000; Siw 864000.

Inputs: Kosmulator_validation_model_comparison.json from the integrated four-chain
validation supplies Dbar and the successful LCDM/iw/Siw polished minima.
Pasted text(7).txt supplies +iw chi2=1470.850477016320 from the independently
validated positive-delta boundary sequence. The unstable exact-zero numerical
minimum from the original integrated +iw run is not used.

DIC_mode = 2*Dbar-chi2; pD_mode = Dbar-chi2. The posterior-mean DIC is included
in CSV/JSON as a labelled sensitivity comparison, not mixed into DIC_mode.
All differences are relative to the matching CLASS-derived LCDM reference.
No sigma/significance values are inferred from Delta pD.

Co-author agreement on DIC convention and burn-in, revised integrated boundary
search validation, and normal Kosmulator workflow validation remain pending.
No production code, chains, CLASS implementation, or GitHub files were changed.

Numerical spot check: +iw positive-boundary chi2 changes by 4.09e-5 between
delta=1e-8 and 1e-10, and its final neighbourhood range is 2.46e-5. Extra
digits are retained for reproducibility, not a claim of that numerical accuracy.
