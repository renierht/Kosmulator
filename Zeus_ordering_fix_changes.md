# Zeus log-likelihood ordering fix (change record)

Date: 2026-10-07
Companion file: `Zeus_ordering_fix_changes.patch` (exact unified diff of every line changed in the three files below; apply with `git apply`, inspect with `git apply --stat`).

Line numbers refer to the files as they are after the change.

## 1. Purpose

The DESI DR2 + DESY5 statistics gave $\Delta{\rm DIC} = -965.8$ and $\Delta{\rm WAIC} \approx -1.3\times10^{7}$ against a DESI DR2 target of $\Delta{\rm DIC} = -9.3$. The DESY5 likelihood was not the cause (see section 6). Two problems with the zeus chain for LCDM were:

1. **A stuck walker.** Walker 18 stayed at the prior wall ($\Omega_m = 0.01$, $r_d \approx 660$ Mpc, $\chi^2 \approx 65{,}000$) until about step 990, past the 500-step burn-in. Its 494 post-burn samples inflated $\bar D$ (DIC) and the WAIC variance term. This change does **not** address that; see section 7.
2. **A bookkeeping bug (fixed here).** zeus 2.5.4 flattens log-probabilities walker by walker (`zeus/samples.py`, line 115: `reshape((-1,), order='F')`). Kosmulator flattens the saved samples step by step (`samples[burn:].reshape(-1, ndim)`, C order). Every zeus chain therefore stored a `log_like` array paired with the wrong parameter rows.

Consequences of the bug:
- $\bar D$ and so DIC were not affected, because a mean does not depend on order.
- The MLE parameter values taken from zeus chains (`argmax(log_like)` in `calculate_asymmetric_from_samples`) were wrong.
- The fault was hard to diagnose. In the LCDM chain the stuck walker's 494 values appeared as a block across all 32 walkers at steps 1125 - 1140. That block starts at flat index $18 \times 2000 = 36{,}000$.

A second, smaller problem: under `--overwrite`, each engine deleted only its own chain file. A stale `X_zeus.h5` therefore survived an emcee rerun, and a later stats-only run could load it again.

## 2. Summary of changes

| file | location | change |
|---|---|---|
| `Kosmulator_main/utils.py` | `AppendProgressCallback` (line 1662) | Also stores zeus' 2D log-probabilities as `log_prob_2d` (steps, walkers), in the same layout as `samples`. |
| `Kosmulator_main/utils.py` | new `zeus_flat_loglike` (lines 1731 - 1781) | Single reader that returns post-burn log-likelihoods flattened step by step (C order), matching the samples. |
| `Kosmulator_main/utils.py` | `load_or_run_chain`, cross-engine reuse I (lines 986 - 996) | Uses the new reader. |
| `Kosmulator_main/utils.py` | `load_or_run_chain`, post-run read-back (lines 1121 - 1142) | Uses the new reader for zeus, and prints which chain file the statistics used. |
| `Kosmulator_main/Kosmulator_MCMC.py` | `run_mcmc`, zeus overwrite (lines 841 - 848) | Overwrite removes both `X_zeus.h5` and `X.h5`. |
| `Kosmulator_main/Kosmulator_MCMC.py` | `run_mcmc`, end of zeus branch (lines 1030 - 1053) | Flat `log_prob` / `log_like` built from `log_prob_2d` (C order) and tagged `loglike_order = "C"`, no longer from `sampler.get_log_prob(flat=True)`. |
| `Kosmulator_main/Kosmulator_MCMC.py` | `run_mcmc`, emcee overwrite (lines 1065 - 1072) | Overwrite removes both `X.h5` and `X_zeus.h5`. |
| `Marcel_chain.py` | import (line 109), `load_chain` zeus branch (lines 130 - 137) | Uses the new reader. This also fixes a second burn-in being applied to an already post-burn array. |

The emcee paths are unchanged apart from the overwrite deletion. emcee flattens chain, `log_prob` and blobs consistently, step by step.

## 3. `Kosmulator_main/utils.py`

### 3.1 `AppendProgressCallback` (line 1662)

zeus calls every callback as `cb(iteration, get_chain(), get_log_prob())` (`zeus/ensemble.py`, line 432). Here `get_log_prob()` without `flat` is the (steps, walkers) array, aligned with the chain. The callback received it but threw it away. It now writes it to `log_prob_2d` beside `samples`.

**Before**
```python
class AppendProgressCallback:
    """
    Zeus callback that appends only new local samples to 'samples' dataset.
    """
    ...
        new_block = chain[self._prev_local:local_n]
        with h5py.File(self.filename, "a") as f:
            if "samples" not in f:
                ds = f.create_dataset(
                    "samples",
                    data=chain,
                    maxshape=(None, nwalker, ndim),
                    chunks=(1, nwalker, ndim),
                )
            else:
                ds = f["samples"]
                old_n = ds.shape[0]
                new_n = old_n + new_local
                ds.resize((new_n, nwalker, ndim))
                ds[old_n:new_n, :, :] = new_block

        self._prev_local = local_n
        return False
```

**After**
```python
class AppendProgressCallback:
    """
    Zeus callback that appends only new local samples to 'samples' dataset,
    and the matching log-probabilities to 'log_prob_2d' (steps, walkers), so
    that both share the same (step, walker) layout.
    """
    ...
        lp = None if log_prob is None else np.asarray(log_prob, dtype=float)
        if lp is not None and lp.shape != (local_n, nwalker):
            lp = None  # unexpected layout: store nothing rather than misaligned values

        new_block = chain[self._prev_local:local_n]
        with h5py.File(self.filename, "a") as f:
            if "samples" not in f:
                ds = f.create_dataset(
                    "samples",
                    data=chain,
                    maxshape=(None, nwalker, ndim),
                    chunks=(1, nwalker, ndim),
                )
                lp_block = lp
            else:
                ds = f["samples"]
                old_n = ds.shape[0]
                new_n = old_n + new_local
                ds.resize((new_n, nwalker, ndim))
                ds[old_n:new_n, :, :] = new_block
                lp_block = None if lp is None else lp[self._prev_local:local_n]

            if lp_block is not None:
                if "log_prob_2d" not in f:
                    lds = f.create_dataset(
                        "log_prob_2d",
                        shape=(0, nwalker),
                        maxshape=(None, nwalker),
                        chunks=(1, nwalker),
                        dtype="f8",
                    )
                else:
                    lds = f["log_prob_2d"]
                n_total = ds.shape[0]
                n_lp = lds.shape[0]
                start = n_total - lp_block.shape[0]
                lds.resize((n_total, nwalker))
                # Rows written before this file stored log-probs (e.g. a resumed
                # older chain) are marked missing with NaN.
                if start > n_lp:
                    lds[n_lp:start, :] = np.nan
                lds[start:n_total, :] = lp_block

        self._prev_local = local_n
        return False
```

Notes:
- The log-prob rows are always written to the same row indices as the samples just written, so the two datasets stay the same length.
- When an older chain without `log_prob_2d` is resumed, its earlier rows are filled with NaN. The reader (3.2) then refuses to use them, so no wrong values are produced.
- The write timing is unchanged: every `ncheck` iterations, plus the final flush in `run_mcmc`. Samples and log-probs therefore always cover the same steps.

### 3.2 New helper `zeus_flat_loglike` (lines 1731 - 1781)

This is the one place that turns a zeus file into a flat log-likelihood array matching `samples[burn:].reshape(-1, ndim)`. It is used by every reader in sections 3.3, 3.4, 4.2 and 5.

```python
def zeus_flat_loglike(
    h5f, burn: int, n_expected: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Post-burn log-likelihoods from an open Zeus chain file, flattened step-major
    (C order) so entry i pairs with row i of samples[burn:].reshape(-1, ndim).

    zeus' own flat output (get_log_prob(flat=True)) is walker-major (order='F')
    and must not be paired with the C-order flattened samples.

    Sources, in order of preference:
      1. 'log_prob_2d' (steps, walkers), written alongside 'samples';
      2. flat 'log_like' (or 'log_prob') saved in C order (attr loglike_order == "C");
      3. a legacy flat 'log_like' from before this fix (walker-major), re-ordered.
    Returns None if nothing usable is found or the length does not match.
    """
    if "samples" not in h5f:
        return None
    nst, nw = h5f["samples"].shape[:2]
    burn = int(burn) if 0 <= int(burn) < nst else 0
    n_post = (nst - burn) * nw

    out = None
    if "log_prob_2d" in h5f and h5f["log_prob_2d"].shape == (nst, nw):
        arr = np.asarray(h5f["log_prob_2d"][burn:], dtype=float)
        if np.isnan(arr).any():
            print(
                f"[WARNING] {h5f.filename}: log-probabilities missing for part of the "
                "post-burn chain; log_like not used (DIC/WAIC unavailable)."
            )
            return None
        out = arr.reshape(-1)
    else:
        key = "log_like" if "log_like" in h5f else ("log_prob" if "log_prob" in h5f else None)
        if key is None:
            return None
        flat = np.asarray(h5f[key], dtype=float).reshape(-1)
        if flat.size != n_post:
            return None
        if h5f.attrs.get("loglike_order", "") == "C":
            out = flat
        else:
            print(
                f"[WARNING] {h5f.filename}: legacy walker-major '{key}' re-ordered "
                "to match the samples (saved before the Zeus ordering fix)."
            )
            out = flat.reshape(nw, nst - burn).T.reshape(-1)

    if n_expected is not None and out.size != int(n_expected):
        return None
    return out
```

Notes:
- With `log_prob_2d` (new files) the burn-in can be changed between runs. The post-burn slice is taken from the 2D array, so it does not depend on the burn used when the chain was saved.
- Old files have no `log_prob_2d` and no `loglike_order` attribute. Their flat `log_like` is assumed to be zeus' walker-ordered output and is re-ordered, with a warning. This is only possible when the current burn matches the one used at save time. Otherwise the length check fails and `None` is returned.

### 3.3 `load_or_run_chain`: cross-engine reuse I (lines 986 - 996)

This path, used when emcee was requested but only a zeus chain exists, copied the zeus `log_like` across unchanged. The old note in the code already doubted the alignment.

**Before**
```python
        with h5py.File(zeus_chain, "r") as f:
            all_samples = f["samples"][:]    # (nsteps, nwalker, ndim)
            loglike = np.array(f["log_like"]) if "log_like" in f else (
                np.array(f["log_prob"]) if "log_prob" in f else None
            )

        burn = int(CONFIG_model.get("burn", 0) or 0)
        # Guard against silly values
        if burn >= all_samples.shape[0]:
            burn = 0

        flat = all_samples[burn:, :, :].reshape(-1, all_samples.shape[-1])
        """
        NOTE: loglike above is unburned/unflattened relative to `flat` in this
        # particular reuse path (Zeus stores log_prob already burn-discarded at
        # save time in run_mcmc, so this is likely already aligned — verify
        # shapes match len(flat) before trusting it; fall back to None if not.
        """
        if loglike is not None and loglike.shape[0] != flat.shape[0]:
            loglike = None
        return {"samples": flat, "loglike": loglike}
```

**After**
```python
        burn = int(CONFIG_model.get("burn", 0) or 0)
        with h5py.File(zeus_chain, "r") as f:
            all_samples = f["samples"][:]    # (nsteps, nwalker, ndim)
            # Guard against silly values
            if burn >= all_samples.shape[0]:
                burn = 0
            flat = all_samples[burn:, :, :].reshape(-1, all_samples.shape[-1])
            # Same step-major order as `flat` (None if unavailable)
            loglike = zeus_flat_loglike(f, burn, n_expected=flat.shape[0])

        return {"samples": flat, "loglike": loglike}
```

### 3.4 `load_or_run_chain`: post-run read-back (lines 1121 - 1142)

After a run, or a zeus fast load or resume, `log_like` is read back from the chain file for DIC and WAIC. This is the path that fed the misaligned LCDM values into the statistics.

**Before**
```python
        saved_path = zeus_chain if is_zeus_run else chain_path
        loglike = None
        if os.path.exists(saved_path):
            try:
                with h5py.File(saved_path, "r") as h5f:
                    if "log_like" in h5f:
                        candidate = np.asarray(h5f["log_like"], dtype=float).reshape(-1)
                        if candidate.size == flat_samples.shape[0]:
                            loglike = candidate
            except Exception as exc:
                print(f"[WARNING] Could not load saved log_like: {exc}")
        result = {"samples": flat_samples, "loglike": loglike}
```

**After**
```python
        saved_path = zeus_chain if is_zeus_run else chain_path
        loglike = None
        if os.path.exists(saved_path):
            try:
                with h5py.File(saved_path, "r") as h5f:
                    if is_zeus_run:
                        loglike = zeus_flat_loglike(
                            h5f,
                            int(CONFIG_model.get("burn", 0) or 0),
                            n_expected=flat_samples.shape[0],
                        )
                    elif "log_like" in h5f:
                        candidate = np.asarray(h5f["log_like"], dtype=float).reshape(-1)
                        if candidate.size == flat_samples.shape[0]:
                            loglike = candidate
            except Exception as exc:
                print(f"[WARNING] Could not load saved log_like: {exc}")
        print(
            f"[INFO] Chain used for statistics: {saved_path} "
            f"(log_like {'loaded' if loglike is not None else 'unavailable'})\n"
        )
        result = {"samples": flat_samples, "loglike": loglike}
```

The emcee branch (`elif`) is the original code. The new `[INFO]` line shows which file the statistics actually used, so a stale file being loaded is now visible in the output.

## 4. `Kosmulator_main/Kosmulator_MCMC.py`

### 4.1 Zeus branch, overwrite (lines 841 - 848)

**Before**
```python
        if do_overw and exists:
            try:
                os.remove(zeus_chain)
            except FileNotFoundError:
                pass
            exists = False
```

**After**
```python
        if do_overw:
            # Remove both engines' files so a stale emcee chain cannot be reused later
            for stale in (zeus_chain, chain_path):
                try:
                    os.remove(stale)
                except FileNotFoundError:
                    pass
            exists = False
```

### 4.2 Zeus branch, saving `log_prob` / `log_like` at the end of a run (lines 1030 - 1053)

This is the source of the bug. `sampler.get_log_prob(discard=burn, flat=True)` is walker-ordered, while the function returns `all_samples[burn:].reshape(-1, ndim)`, which is step-ordered.

**Before**
```python
        # Save Blobs from Zeus if available
        # Explicitly save log_prob and blobs from Zeus
        if saveChains:
            with h5py.File(zeus_chain, "a") as h5f:
                try:
                    # 1. Save log_prob
                    flat_log_prob = sampler.get_log_prob(discard=burn, flat=True)
                    if "log_prob" in h5f:
                        del h5f["log_prob"]
                    h5f.create_dataset("log_prob", data=flat_log_prob)

                    # 2. Flat top-hat priors everywhere → log_prob == log_like exactly
                    if "log_like" in h5f:
                        del h5f["log_like"]
                    h5f.create_dataset("log_like", data=flat_log_prob)
                except Exception as e:
                    print(f"\n[WARNING] Could not save 'log_prob'/'log_like' from Zeus: {e}")
```

**After**
```python
        # Save flat post-burn log_prob / log_like, aligned row-for-row with the
        # samples returned below. Built from 'log_prob_2d' in the file, not from
        # sampler.get_log_prob(flat=True), which zeus flattens walker-major.
        if saveChains:
            with h5py.File(zeus_chain, "a") as h5f:
                try:
                    flat_log_prob = utils.zeus_flat_loglike(
                        h5f, burn, n_expected=all_samples[burn:].shape[0] * nwalker
                    )
                    for key in ("log_prob", "log_like"):
                        if key in h5f:
                            del h5f[key]
                    if flat_log_prob is None:
                        print(
                            "\n[WARNING] Zeus log-probabilities incomplete; "
                            "'log_prob'/'log_like' not saved (DIC/WAIC unavailable)."
                        )
                    else:
                        h5f.create_dataset("log_prob", data=flat_log_prob)
                        # Flat top-hat priors everywhere → log_prob == log_like exactly
                        h5f.create_dataset("log_like", data=flat_log_prob)
                        h5f.attrs["loglike_order"] = "C"
                except Exception as e:
                    print(f"\n[WARNING] Could not save 'log_prob'/'log_like' from Zeus: {e}")
```

Another advantage: the old code only covered the steps run in the current session, so a resumed zeus chain had log-probs for part of the chain only. The new values come from the file, which covers every saved step.

### 4.3 Emcee branch, overwrite (lines 1065 - 1072)

**Before**
```python
        if saveChains:
            if overwrite and os.path.exists(chain_path):
                try:
                    os.remove(chain_path)
                except FileNotFoundError:
                    pass
            backend = emcee.backends.HDFBackend(chain_path)
```

**After**
```python
        if saveChains:
            if overwrite:
                # Remove both engines' files so a stale Zeus chain cannot be reused later
                for stale in (chain_path, chain_path.replace(".h5", "_zeus.h5")):
                    try:
                        os.remove(stale)
                    except FileNotFoundError:
                        pass
            backend = emcee.backends.HDFBackend(chain_path)
```

## 5. `Marcel_chain.py`

The import on line 109 now also brings in `zeus_flat_loglike`. The zeus branch of `load_chain` (lines 130 - 137):

**Before**
```python
        with h5py.File(zeus_path, "r") as f:
            all_samples = f["samples"][:]
            if "log_like" in f:
                loglike_raw = np.array(f["log_like"], dtype=float)
            elif "log_prob" in f:
                loglike_raw = np.array(f["log_prob"], dtype=float)
            else:
                loglike_raw = None

        flat_samples = all_samples[burn:].reshape(-1, all_samples.shape[-1])
        flat_loglike = loglike_raw[burn:].reshape(-1) if loglike_raw is not None else None
```

**After**
```python
        with h5py.File(zeus_path, "r") as f:
            all_samples = f["samples"][:]
            flat_samples = all_samples[burn:].reshape(-1, all_samples.shape[-1])
            # Post-burn log_like in the same step-major order as flat_samples
            flat_loglike = zeus_flat_loglike(f, burn, n_expected=flat_samples.shape[0])
```

The old line `loglike_raw[burn:]` removed `burn` entries from an array that was already post-burn. Its length was therefore always `burn` short of the samples, and it was also walker-ordered.

## 6. Validation

All scripts were run from the scratchpad with the project's `pykosm` environment. All three edited files compile (`py_compile`).

1. **Fresh zeus run** (toy 2D Gaussian, 8 walkers, 200 steps, burn 50, through `AppendProgressCallback` and `zeus_flat_loglike`): the stored log-likelihood equals the true log-posterior at every post-burn sample. zeus' own `get_log_prob(discard=burn, flat=True)` fails the same comparison, which reproduces the bug.
2. **Flat file in the new order** (no `log_prob_2d`, `loglike_order = "C"`): aligned.
3. **Old walker-ordered file:** re-ordered and aligned, with the warning printed.
4. **Resumed old file** (100 steps without log-probs, plus 100 new steps): `log_prob_2d` has 100 NaN rows, and the helper returns `None` with a warning instead of misaligned values.
5. **Real chain** (a copy of `LCDM_v/DESI_DR2_DESY5/DESI_DR2+DESY5_zeus.h5`): after re-ordering, the 488 samples in the stuck region ($\Omega_m < 0.1$ or $r_d > 200$ Mpc) have $\ln L \le -14{,}874$, and all other samples have $\ln L \ge -1{,}341$. Before the fix those samples showed $\ln L \approx -829$.

DESY5 checks from the same session, for reference:
- `Observations/DESY5.dat` and `DESY5_covsys_000.txt` are byte-identical (md5) to Cobaya's `sn_data/DESY5` files, last changed April 2024.
- Kosmulator's distance convention and covariance construction match Cobaya's `sn/desy5.py` and `sn/pantheonplus.py` (master).

With walker 18 excluded, or burn-in raised to 1100 or more, the existing chains give:
- $p_D({\rm LCDM}) \approx 2.9 - 3.0$;
- $\Delta{\rm DIC} \approx -9.1$ to $-9.2$, against a target of $-9.3$;
- $\Delta\chi^2 = -13.42$, against $-13.6$;
- $\sigma = 3.2$, against $3.3$.

### Not tested
- No full Kosmulator run with zeus was made after the change. The two `run_mcmc` edits (4.2, 4.3) and the overwrite deletion (4.1) were checked by reading them, not by running them. The helper and the callback, which hold the logic, were tested directly.
- `Marcel_chain.py` was compiled but not run.

## 7. Not changed (deliberately)

- **Stuck walkers and the initial spread of walkers** (item 3 of the proposal). Walkers still start at 10% of the prior width, because Nelder-Mead gives no Hessian (`optimise_initial_guess`, around line 455). For `r_d` with prior (0.01, 1000) that is about ±71 Mpc. A walker can still get stuck past the burn-in.
- **Guards in the statistics step** (item 4): the alignment spot-check, and the outlier warning for DIC and WAIC.
- **The old zeus chain file** `MCMC_Chains/DESI_DESY5_zHEL_32X2500/LCDM_v/DESI_DR2_DESY5/DESI_DR2+DESY5_zeus.h5` is still on disk. It now loads in the correct order, but it still contains the stuck walker. Delete it, or rerun with `--overwrite`, which now removes both files.
- **The final flush** in `run_mcmc` still only writes if `sampler.iteration` is a multiple of `ncheck` (existing behaviour of `AppendProgressCallback`). Samples and log-probs are affected equally, so alignment is not affected.

## 8. Other uncommitted change in `Post_processing.py` (same session, not part of this fix)

Sigma now uses the difference in the number of free parameters as the $\chi^2$ degrees of freedom (Wilks), not the difference in $p_D$. A negative $p_D$ difference had made sigma NaN.
- `"n_params": num_params` is stored in each result, and `reference_npar` holds the reference model's value.
- `stats['sigma'] = significance(stats['dChi'], d_npar) if d_npar > 0 else 0.0`.

The `rng = np.random.default_rng(42)` lines in the same file's diff (the fixed WAIC subsample seed) were not made in this session.
