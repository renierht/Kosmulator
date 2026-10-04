"""Posterior-based model comparison, independent of plotting summaries.

DIC uses the mean deviance and deviance at the posterior mean. AIC/AICc/BIC
use a separately polished likelihood, with the full sampled parameter count.
No CLASS implementation or precision setting is changed here.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

log = logging.getLogger("Kosmulator.Model_comparison")
DEFAULTS = {
    "polish": True, "starts": 3, "maxfev": 1500,
    "xatol": 1e-6, "fatol": 2e-5,
    "boundary_eps": (1e-4, 1e-6, 1e-8),
    "stored_chi2_tolerance": 1e-3,
    "delta_boundary_eps": (1e-6, 1e-8, 1e-10),
    "delta_boundary_chi2_tolerance": 5e-4,
}
SWITCHES = (
    "ALLOW_NEGATIVE_ENERGIES", "ALLOW_BIG_RIP",
    "ALLOW_DOOM_FACTOR_INSTABILITIES",
)


def normalise_key(key):
    return str(key).replace("PantheonP_SH0ES", "PantheonPS").replace(
        "Pantheon+SH0ES", "PantheonPS"
    ).replace("_", "+")


def observation_index(key, cfg):
    from Kosmulator_main import utils
    matches = [i for i, obs in enumerate(cfg["observations"])
               if normalise_key(key) in {
                   normalise_key("+".join(obs)),
                   normalise_key(utils.generate_label(obs, config_model=cfg,
                                                       obs_index=i)),
               }]
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely match observation group {key!r}")
    return matches[0]


def count_observations(data, cfg, index):
    """Count measurements actually selected by the likelihood loaders.

    CMB retains the existing effective-count convention, explicitly annotated;
    users can supply a dataset's n_data_points when an exact count is known.
    Gaussian BBN calibration likelihoods count as one measurement.
    """
    counts, notes = {}, []
    for obs, typ in zip(cfg["observations"][index],
                        cfg["observation_types"][index]):
        d = data[obs]
        if isinstance(d, dict) and "n_data_points" in d:
            n = int(d["n_data_points"])
        elif obs in ("PantheonP", "PantheonPS"):
            n = len(d["m_b_corr"])
        elif obs == "BAO":
            n = len(d["covd1"])
        elif obs in ("DESI_DR1", "DESI_DR2") or typ == "SNe":
            n = len(d["redshift"])
        elif typ in ("OHD", "CC", "f", "f_sigma_8"):
            n = len(d["type_data"])
        elif obs in ("BBN_prior", "BBN_PryMordial"):
            n = 2 if "cov" in d and "mu_Neff" in d else 1
        elif typ in ("BBN_DH", "BBN_DH_AlterBBN"):
            n = 1 if d.get("mode", "mean") == "mean" else len(d["systems"])
        elif obs == "CMB_lowl":
            n = 30
            notes.append("CMB_lowl N uses the existing effective-count convention")
        elif obs in ("CMB_hil", "CMB_hil_TT"):
            from Kosmulator_main import Statistical_packages as sp
            like = sp._get_hil_like() if obs == "CMB_hil" else sp._get_hilTT_like()
            lm = like.get_lmax()
            if isinstance(lm, dict):
                ls = [lm.get(k, lm.get(k.upper(), 0)) or 0
                      for k in ("tt", "ee", "bb", "te")]
            else:
                ls = list(lm) + [0] * 4
            n = max(int(ls[0])-1, 0)
            if obs == "CMB_hil":
                n += max(int(ls[1])-1, 0) + max(int(ls[3])-1, 0)
            notes.append(f"{obs} N uses the existing multipole-count convention")
        elif obs == "CMB_lensing":
            from Kosmulator_main import Statistical_packages as sp
            primary = any(x in {"CMB_hil", "CMB_hil_TT", "CMB_lowl"}
                          for x in cfg["observations"][index])
            sp.set_lensing_mode("raw" if primary else "cmbmarged")
            like = sp._get_lensing_like()
            if hasattr(like, "get_lensing_nbins"):
                n = int(like.get_lensing_nbins())
            elif hasattr(like, "get_lensing_bins"):
                n = len(like.get_lensing_bins())
            else:
                n = 8
                notes.append("CMB_lensing N uses the existing effective count of 8")
        else:
            raise ValueError(f"No measurement-count rule for {obs} ({typ}); "
                             "supply data[obs]['n_data_points']")
        if n <= 0:
            raise ValueError(f"Non-positive measurement count for {obs}: {n}")
        counts[obs] = n
    return sum(counts.values()), counts, notes


def _blob_values(blob):
    a = np.asarray(blob)
    if a.dtype.names:
        field = next((n for n in ("log_like", "loglike", "log_likelihood",
                                  "lnlike", "ll") if n in a.dtype.names), None)
        if field is None and len(a.dtype.names) == 1:
            field = a.dtype.names[0]
        if field is None:
            raise ValueError("Ambiguous structured likelihood blobs")
        a = a[field]
    return np.asarray(a, dtype=float).reshape(-1)


def saved_likelihood(samples, source, burn, cfg, index):
    """Read emcee likelihoods only when the complete retained chain matches.

    Posterior values without blobs are converted using the actual prior;
    they are never silently treated as likelihood values.
    """
    paths = [source] if isinstance(source, (str, Path)) else (source or [])
    if not any(Path(path).is_file() for path in paths):
        return None, None
    import emcee
    from Kosmulator_main import Kosmulator_MCMC as km
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        import h5py
        with h5py.File(path, "r") as f:
            if "mcmc" not in f:
                continue  # Older Zeus files contain samples, not likelihoods.
        backend = emcee.backends.HDFBackend(str(path), read_only=True)
        chain = backend.get_chain(discard=burn, flat=True)
        if chain.shape != samples.shape or not np.array_equal(
                chain, samples, equal_nan=True):
            continue
        if backend.has_blobs():
            ll = _blob_values(backend.get_blobs(discard=burn, flat=True))
            origin = "saved likelihood blobs"
        else:
            post = backend.get_log_prob(discard=burn, flat=True)
            prior = km.log_prior_all(samples, cfg, index)
            ll = np.full(len(post), np.nan)
            valid = np.isfinite(post) & np.isfinite(prior)
            ll[valid] = post[valid] - prior[valid]
            origin = "saved log posterior minus evaluated log prior"
        if len(ll) != len(samples):
            raise ValueError("Likelihood/sample alignment mismatch")
        return ll, f"{origin}: {path.resolve()}"
    return None, None


def posterior_statistics(samples, log_likelihood, evaluate):
    """Evaluate DIC at the full, unrounded retained-posterior mean."""
    samples = np.asarray(samples, dtype=float)
    ll = np.asarray(log_likelihood, dtype=float)
    if samples.ndim != 2 or ll.shape != (len(samples),):
        raise ValueError("Expected a 2-D chain and aligned 1-D likelihoods")
    finite = np.isfinite(ll) & np.all(np.isfinite(samples), axis=1)
    if not finite.any():
        raise ValueError("No finite retained posterior samples")
    selected, likelihood = samples[finite], ll[finite]
    mean = selected.mean(axis=0)
    dbar = float(np.mean(-2 * likelihood))
    dmean = -2 * float(evaluate(mean))
    if not np.isfinite(dmean):
        raise ValueError("Posterior mean has an invalid likelihood; DIC undefined")
    return selected, likelihood, {
        "D_bar": dbar, "D_at_mean": dmean, "p_D": dbar-dmean,
        "DIC": 2*dbar-dmean, "Posterior_mean": mean.tolist(),
        "Retained_samples": int(len(samples)),
        "Finite_samples": int(finite.sum()),
        "Excluded_samples": int((~finite).sum()),
    }


def polish_likelihood(samples, ll, evaluate, bounds, names, options,
                      positive_delta=False, stable_w=False):
    """Multi-start likelihood polishing with explicit IDE boundary checks."""
    bounds = np.asarray(bounds, dtype=float).copy()
    if positive_delta and "delta" in names:
        bounds[names.index("delta"), 0] = max(
            0., bounds[names.index("delta"), 0])
    if stable_w and "w" in names:
        bounds[names.index("w"), 1] = min(
            -1-options["boundary_eps"][-1], bounds[names.index("w"), 1])
    best = samples[int(np.argmax(ll))].copy()
    best_ll = float(evaluate(best))
    if not np.isfinite(best_ll):
        raise ValueError("Best stored likelihood point is invalid")
    runs = []
    if not options["polish"]:
        return best, best_ll, runs, "stored-chain likelihood point (unpolished)"

    delta_sequence = positive_delta and "delta" in names and not stable_w
    delta_eps = tuple(options.get("delta_boundary_eps", DEFAULTS["delta_boundary_eps"]))
    delta_tol = float(options.get("delta_boundary_chi2_tolerance",
                                  DEFAULTS["delta_boundary_chi2_tolerance"]))
    if delta_sequence:
        j = names.index("delta")
        delta_sequence = bounds[j, 0] <= 0 < bounds[j, 1]
    if delta_sequence:
        # The exact-zero CLASS branch has reproducible ULP-scale dips. Keep
        # the optimisation inside positive delta and approach zero explicitly.
        bounds[j, 0] = float(delta_eps[-1])
        if bounds[j, 1] < delta_eps[0]:
            raise ValueError("delta prior cannot contain the configured boundary sequence")
        best = np.clip(best, bounds[:, 0], bounds[:, 1])
        best_ll = float(evaluate(best))
        if not np.isfinite(best_ll):
            raise ValueError("Positive-delta starting point is invalid")

    def run(start, fixed=None, local_simplex=False, label="interior"):
        nonlocal best, best_ll
        fixed = fixed or {}
        free = [j for j in range(len(names)) if j not in fixed]
        full_start = np.clip(np.asarray(start, dtype=float), bounds[:, 0], bounds[:, 1])
        for j, value in fixed.items():
            full_start[j] = value
        def expand(x):
            full = full_start.copy()
            full[free] = x
            return full
        rejected = []
        def objective(x):
            nonlocal best, best_ll
            full = expand(x)
            try:
                value = float(evaluate(full))
            except Exception as exc:
                if not rejected:
                    rejected.append(f"{type(exc).__name__}: {exc}")
                return 1e100
            if np.isfinite(value) and value > best_ll:
                best, best_ll = full.copy(), value
            return -value if np.isfinite(value) else 1e100
        settings = {key: options[key] for key in ("xatol", "fatol", "maxfev")}
        settings.update(maxiter=options["maxfev"], adaptive=True, disp=False)
        x0 = full_start[free]
        if local_simplex:
            widths = samples.std(axis=0, ddof=1)
            simplex = np.tile(x0, (len(free)+1, 1))
            for k, j in enumerate(free):
                step = max(.15*widths[j], 1e-6*max(abs(full_start[j]), 1.))
                if label.endswith("positive boundary"):
                    # Recovered standalone boundary refinement simplex widths.
                    step = {"Omega_m": 2e-4, "w": 2e-3,
                            "H_0": .03, "M_abs": .002}.get(names[j], step)
                lo, hi = bounds[j]
                direction = -1 if full_start[j]-lo < hi-full_start[j] else 1
                trial = np.clip(full_start[j]+direction*step, lo, hi)
                if trial == full_start[j]:
                    trial = np.clip(full_start[j]-direction*step, lo, hi)
                simplex[k+1, k] = trial
            settings["initial_simplex"] = simplex
        result = minimize(objective, x0, method="Nelder-Mead",
                          bounds=[tuple(bounds[j]) for j in free], options=settings)
        terminal = expand(result.x)
        try:
            terminal_ll = float(evaluate(terminal))
        except Exception:
            terminal_ll = -np.inf
        if np.isfinite(terminal_ll) and terminal_ll > best_ll:
            best, best_ll = terminal.copy(), terminal_ll
        vertices, values = result.final_simplex
        xspread = np.max(np.abs(vertices[1:]-vertices[0]), axis=0)
        fspread = float(np.max(np.abs(values[1:]-values[0])))
        runs.append({
            "stage": label, "success": bool(result.success) and bool(np.isfinite(terminal_ll)),
            "optimizer_success": bool(result.success),
            "message": str(result.message), "nfev": int(result.nfev),
            "chi2": float(-2*terminal_ll),
            "parameters": dict(zip(names, terminal.tolist())),
            "simplex_parameter_spreads": dict(zip([names[j] for j in free],
                                                   xspread.tolist())),
            "simplex_objective_spread": fspread,
            "xatol": options["xatol"], "fatol": options["fatol"],
            "first_rejected_evaluation_exception": rejected[0] if rejected else None,
        })
        return terminal

    starts = [best.copy(), samples.mean(axis=0), np.median(samples, axis=0)]
    requested = int(options["starts"])
    if requested > 3:
        order = np.argsort(ll)[::-1]
        starts.extend(samples[j].copy() for j in order[1:requested-2])
    for i, start in enumerate(starts[:requested], 1):
        run(start, local_simplex=True, label=f"interior start {i}")
    note = "polished likelihood (interior search)"
    if delta_sequence:
        j = names.index("delta")
        interior, interior_ll = best.copy(), best_ll
        sequence_start = interior.copy()
        sequence = []
        for eps in delta_eps:
            sequence_start = run(sequence_start, {j: float(eps)},
                                 local_simplex=True,
                                 label=f"delta={eps:g} positive boundary")
            record = runs[-1]
            probes = [sequence_start.copy()]
            for k in range(len(names)):
                if k == j:
                    continue
                for target in (-np.inf, np.inf):
                    probe = sequence_start.copy()
                    probe[k] = np.nextafter(probe[k], target)
                    if bounds[k, 0] <= probe[k] <= bounds[k, 1]:
                        probes.append(probe)
                step = 1e-8*max(abs(sequence_start[k]), 1.)
                for sign in (-1, 1):
                    probe = sequence_start.copy()
                    probe[k] += sign*step
                    if bounds[k, 0] <= probe[k] <= bounds[k, 1]:
                        probes.append(probe)
            probe_chi2 = [-2*float(evaluate(probe)) for probe in probes]
            spread = float(np.ptp(probe_chi2)) if np.all(np.isfinite(probe_chi2)) else np.inf
            record["neighbourhood_chi2_range"] = spread
            record["boundary_chi2_tolerance"] = delta_tol
            record["boundary_epsilon"] = float(eps)
            sequence.append((sequence_start.copy(), probe_chi2[0], record))
        final, final_chi2, final_record = sequence[-1]
        competitive = final_chi2 <= -2*interior_ll + delta_tol
        if competitive:
            converged = (len(sequence) >= 2 and
                         abs(sequence[-1][1]-sequence[-2][1]) <= delta_tol and
                         all(r["success"] and r["neighbourhood_chi2_range"] <= delta_tol
                             for _, _, r in sequence) and
                         all(right[1] <= left[1]+delta_tol
                             for left, right in zip(sequence, sequence[1:])))
            final_record["boundary_sequence_validated"] = bool(converged)
            if not converged:
                diagnostic = [{"epsilon": r["boundary_epsilon"], "chi2": chi,
                               "success": r["success"],
                               "neighbourhood_range": r["neighbourhood_chi2_range"]}
                              for _, chi, r in sequence]
                raise ValueError("Positive-delta boundary sequence requires review; "
                                 "no polished model-comparison row accepted: "
                                 + json.dumps(diagnostic))
            # Select the validated terminal limit, rather than a transient
            # optimiser evaluation or a point on the numerically singular zero branch.
            best, best_ll = final.copy(), -0.5*final_chi2
            note = ("validated positive-delta boundary limit as delta approaches 0 "
                    f"from above (terminal delta={delta_eps[-1]:g})")
        else:
            best, best_ll = interior, interior_ll
            note = "interior MLE; positive-delta boundary sequence also checked"
    if stable_w and "w" in names:
        j = names.index("w")
        sequence_start = best.copy()
        for eps in options["boundary_eps"]:
            fixed = {j: -1-float(eps)}
            if all(bounds[k, 0] <= v <= bounds[k, 1] for k, v in fixed.items()):
                sequence_start = run(sequence_start, fixed,
                                     label=f"w=-1-{eps:g} boundary")
        note = ("boundary-limited likelihood supremum as w approaches -1 from below"
                if abs(best[j]+1) <= 2*options["boundary_eps"][-1]
                else "interior MLE; w->-1 boundary sequence also checked")
    return best, best_ll, runs, note


def analyse_group(samples, data, cfg, model_name, index, source=None, options=None):
    from Kosmulator_main import Kosmulator_MCMC as km
    import User_defined_modules as udm
    opts = dict(DEFAULTS)
    opts.update(cfg.get("postprocessing", {}))
    opts.update(options or {})
    if int(opts["starts"]) < 1 or int(opts["maxfev"]) < 1:
        raise ValueError("starts and maxfev must be positive")
    if opts["xatol"] <= 0 or opts["fatol"] <= 0:
        raise ValueError("Optimiser tolerances must be positive")
    eps = tuple(map(float, opts["boundary_eps"]))
    if not eps or any(e <= 0 for e in eps) or any(a <= b for a, b in zip(eps, eps[1:])):
        raise ValueError("boundary_eps must be a decreasing positive sequence")
    opts["boundary_eps"] = eps
    delta_eps = tuple(map(float, opts["delta_boundary_eps"]))
    if (len(delta_eps) < 2 or any(not np.isfinite(e) or e <= 0 for e in delta_eps)
            or any(a <= b for a, b in zip(delta_eps, delta_eps[1:]))):
        raise ValueError("delta_boundary_eps must be a decreasing finite positive sequence")
    if (not np.isfinite(opts["delta_boundary_chi2_tolerance"])
            or opts["delta_boundary_chi2_tolerance"] <= 0):
        raise ValueError("delta_boundary_chi2_tolerance must be finite and positive")
    opts["delta_boundary_eps"] = delta_eps
    names = list(cfg["parameters"][index])
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(names):
        raise ValueError("Retained samples do not match CONFIG parameter dimensions")
    n, counts, notes = count_observations(data, cfg, index)
    if n <= len(names):
        raise ValueError("Non-positive residual degrees of freedom")
    obs, types = cfg["observations"][index], cfg["observation_types"][index]
    func = udm.Get_model_function(model_name)
    original = {k: getattr(udm, k) for k in SWITCHES}
    flags = cfg.get("_postprocessing_switches", original)
    try:
        for key in SWITCHES:
            setattr(udm, key, flags[key])
        def evaluate(theta):
            arr = np.asarray(theta, dtype=float)[None, :]
            prior = float(np.asarray(km.log_prior_all(arr, cfg, index)).ravel()[0])
            if not np.isfinite(prior):
                return -np.inf
            value = float(np.asarray(km.log_likelihood_all(
                arr, data, cfg, func, model_name, obs, types, index)).ravel()[0])
            return value if np.isfinite(value) and abs(value) < 1e90 else -np.inf

        ll, origin = saved_likelihood(samples, source, int(cfg.get("burn", 0)), cfg, index)
        if ll is None:
            origin = "re-evaluated all retained samples (no aligned saved likelihoods)"
            log.warning("%s: %s; this can be expensive", model_name, origin)
            ll = np.full(len(samples), np.nan)
            for start in range(0, len(samples), 256):
                batch = samples[start:start+256]
                valid = np.all(np.isfinite(batch), axis=1)
                if valid.any():
                    pos = np.flatnonzero(valid)+start
                    ll[pos] = km.log_likelihood_all(batch[valid], data, cfg, func,
                                                   model_name, obs, types, index)
        selected, likelihood, result = posterior_statistics(samples, ll, evaluate)
        raw_idx = int(np.argmax(likelihood))
        raw_chi2 = -2*float(likelihood[raw_idx])
        check_chi2 = -2*evaluate(selected[raw_idx])
        if abs(check_chi2-raw_chi2) > opts["stored_chi2_tolerance"]:
            raise ValueError("Saved/current likelihood mismatch: "
                             f"delta chi2={check_chi2-raw_chi2:g}")
        # Automatic paper-regime boundaries apply only to the validated model.
        positive = model_name == "NonLinear_IDE_2" and not flags[SWITCHES[0]]
        stable = model_name == "NonLinear_IDE_2" and not flags[SWITCHES[2]]
        best, best_ll, runs, mle_note = polish_likelihood(
            selected, likelihood, evaluate,
            [cfg["prior_limits"][index][p] for p in names], names, opts,
            positive_delta=positive, stable_w=stable)
        k, chi2 = len(names), -2*best_ll
        aicc = chi2+2*k+2*k*(k+1)/(n-k-1) if n > k+1 else np.nan
        if n <= k+1:
            notes.append("AICc undefined because N <= k+1")
        result.update({
            "N": n, "k": k, "Data_counts": counts,
            "Log-Likelihood": best_ll, "Chi_squared": chi2,
            "Reduced_Chi_squared": chi2/(n-k),
            "Raw_chain_chi2": raw_chi2,
            "Stored_point_check_delta_chi2": check_chi2-raw_chi2,
            "AIC": chi2+2*k, "AICc": aicc,
            "BIC": chi2+k*np.log(n), "MLE_parameters": dict(zip(names, best.tolist())),
            "Parameter_names": names, "MLE_note": mle_note,
            "Optimizer_runs": runs, "Likelihood_source": origin,
            "Burn": int(cfg.get("burn", 0)), "Thin": 1,
            "Physicality_switches": dict(flags), "Postprocessing_options": opts,
        })
        if runs and not all(r["success"] for r in runs):
            notes.append("One or more optimiser runs did not report convergence; see diagnostics")
        if notes:
            result["Note"] = " | ".join(notes)
        return result
    finally:
        for key, value in original.items():
            setattr(udm, key, value)


def compare_models(results, reference):
    """Undefined/mismatched references produce NaN deltas, never fake zeros."""
    for model, groups in results.items():
        for key, result in groups.items():
            ref = next((v for rk, v in results.get(reference, {}).items()
                        if normalise_key(rk) == normalise_key(key)), None)
            matching = ref is not None and result["Data_counts"] == ref["Data_counts"]
            for statistic, delta in (("Chi_squared", "dChi_squared"),
                                     ("AIC", "dAIC"), ("AICc", "dAICc"),
                                     ("DIC", "dDIC"), ("BIC", "dBIC")):
                result[delta] = float(result[statistic]-ref[statistic]) if matching else np.nan
            result["Reference_model"] = reference
            if not matching:
                result["Note"] = (result.get("Note", "") +
                    " | Matching reference model/dataset unavailable; deltas undefined").strip(" |")
    return results


def analyse_models(all_samples, data, config, reference):
    results = {}
    for model, groups in all_samples.items():
        cfg = config[model]
        results[model] = {}
        for key, samples in groups.items():
            index = observation_index(key, cfg)
            source = cfg.get("_postprocessing_sources", {}).get(key)
            results[model][key] = analyse_group(samples, data, cfg, model, index, source)
    return compare_models(results, reference)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def export_comparison(results, folder):
    """Full-precision diagnostics, CSV, LaTeX table and data-driven IC figure."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model_comparison.json").write_text(
        json.dumps(_json_safe(results), indent=2, allow_nan=False), encoding="utf-8")
    fields = ["Model", "Observation", "N", "k", "p_D", "Chi_squared",
              "D_bar", "D_at_mean", "AIC", "AICc", "DIC", "BIC",
              "dChi_squared", "dAIC", "dAICc", "dDIC", "dBIC", "MLE_note"]
    with (folder / "model_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model, groups in results.items():
            for key, row in groups.items():
                writer.writerow({"Model": model, "Observation": key,
                                 **{name: row[name] for name in fields[2:]}})
    from Kosmulator_main.Plot_metadata import model_label, observation_label
    def escape(value):
        return str(value).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    lines = [r"\begin{tabular}{llrrrrrrrr}",
             r"Model & Observations & $N$ & $k$ & $p_D$ & $\Delta\chi^2$ & $\Delta$AIC & $\Delta$AICc & $\Delta$DIC & $\Delta$BIC \\",
             r"\hline"]
    for model, groups in results.items():
        for key, row in groups.items():
            values = [escape(model_label(model, flags=row.get("Physicality_switches"))),
                      escape(observation_label(key)), str(row["N"]), str(row["k"])]
            values += [f"{row[name]:.3f}" if np.isfinite(row[name]) else "--"
                       for name in ("p_D", "dChi_squared", "dAIC", "dAICc", "dDIC", "dBIC")]
            lines.append(" & ".join(values) + r" \\")
    lines.append(r"\end{tabular}")
    (folder / "model_comparison.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
    import matplotlib.pyplot as plt
    grouped = {}
    for model, groups in results.items():
        for key, row in groups.items():
            if model != row["Reference_model"] and all(np.isfinite(row[k]) for k in
                                                       ("dAIC", "dAICc", "dDIC", "dBIC")):
                grouped.setdefault(normalise_key(key), []).append((model, row))
    for number, (key, rows) in enumerate(grouped.items(), 1):
        fig, ax = plt.subplots(figsize=(max(6, len(rows)*2.5), 4.5))
        x, width = np.arange(len(rows)), .18
        for j, (stat, label, color) in enumerate(zip(
                ("dAIC", "dAICc", "dDIC", "dBIC"),
                ("AIC", "AICc", "DIC", "BIC"),
                ("#4C78A8", "#F58518", "#54A24B", "#8F63B8"))):
            bars = ax.bar(x+(j-1.5)*width, [r[1][stat] for r in rows], width,
                          label=r"$\Delta$"+label, color=color)
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.axhline(0, color="0.3", linewidth=.8)
        ax.set_xticks(x, [model_label(model, flags=row.get("Physicality_switches"), mathtext=True)
                         for model, row in rows])
        ax.set_ylabel("IC difference relative to " +
                      model_label(rows[0][1]["Reference_model"], mathtext=True))
        ax.set_title(observation_label(key, mathtext=True))
        ax.margins(y=.2)
        ax.legend(ncol=4, fontsize=9)
        fig.tight_layout()
        for extension in ("pdf", "png"):
            fig.savefig(folder / f"information_criteria_{number}.{extension}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
