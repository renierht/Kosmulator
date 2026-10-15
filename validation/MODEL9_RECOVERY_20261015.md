# Model 9 recovery, 2026-10-15

The Model9_unrestricted_20261014_iw experiment stalled at 18,683 saved steps.
The original parent was PID 5889; a native CLASS crash killed worker 5897.
A normal restart reproduced the crash. Both stalled parents were interrupted,
and the HDF5 chain plus metadata were backed up before recovery under
`Kosmulator_warm_dm/recovery/model9_20261015_230338/`.

A separate, single-thread gdb replay reproduced SIGSEGV in
`tools/evolver_ndf15.c:numjac`, while evaluating thermodynamics at t=-50.
All three unperturbed and perturbed derivatives were NaN. Comparisons with NaN
never initialized Rowmax, which was then used to index ydel_Fdel.

The fix checks finite input state/derivatives and perturbed derivatives before
Jacobian indexing. It returns a normal CLASS failure for invalid derivatives.
No model equation, prior, dataset, helium setting, precision tolerance, walker
count, burn, cap or convergence threshold was changed. The existing strict
CLASS likelihood handles reported failures; no approximate r_d fallback is used.

The old production build helper did not force relinking the Python extension
when the static C library changed. Its build_ext call now includes --force,
matching the existing development-branch build helper. The old cached backend
214a1a74 remains available; the relinked guarded backend is 0eadca07.

Validation:

- Direct native NaN test returns _FAILURE_ before indexing uninitialized memory:
  `gcc -I Class/IDE_background/include validation/test_numjac_nan.c Class/IDE_background/libclass.a -lm -lstdc++ -fopenmp -o /tmp/test_numjac_nan`
- `validation/validate_class_nan_guard.py` checks four successful reference
  points; maximum r_d difference was about 2.38e-7 Mpc (1.54e-9 relative).
- The old crashing point was h=0.7416782117591887, Omega_m=0.30705956939982576,
  w=-1.06468083280595, delta=-0.18805327362650148, w_dm=0, with paper calibration.
  The rebuilt binary returns finite r_d=57.69170758032131 Mpc on three independent
  cache-cleared calls. Thus it is not accurate to say this particular point is
  now always rejected: rebuilding changes the observed numerical behavior.
- All 120 saved checkpoint log probabilities were recomputed: maximum absolute
  difference 5.979964157631912e-6. This checks existing saved states, not every
  point in the prior. The successful computations are numerically consistent.
- All 56 tests in the production checkout passed.

This repairs a native-code failure and permits recovery; it does not establish
that every numerical failure throughout the prior is physically justified.
The earlier radiation-initialization and BBN-table limitations remain. Publication
inference still needs those domain issues assessed. A future repair changing
finite likelihoods or usable prior support materially must be evaluated separately
for whether fresh chains are needed.
