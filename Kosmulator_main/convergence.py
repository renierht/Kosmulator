"""Convergence policy ported from fR_followup_fixes at 8de59c3.

The upstream single-ensemble diagnostics are retained, not interpreted as
independent-chain rank-normalized R-hat or bulk/tail ESS.
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
import numpy as np
from Kosmulator_main import constants as K

POLICY_VERSION = "fr-8de59c3-v1"


def save_report(monitor, status, path, model, group, engine):
    """Persist the actual verdict and settings, never infer it from chain length."""
    if path is None:
        return
    import h5py
    with h5py.File(path, "a") as handle:
        handle.attrs["converged"] = bool(status.get("converged", False))
        handle.attrs["convergence_policy"] = POLICY_VERSION
    def serial(value):
        if isinstance(value, np.ndarray): return serial(value.tolist())
        if isinstance(value, np.generic): return serial(value.item())
        if isinstance(value, dict): return {k: serial(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [serial(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value): return None
        return value
    report = dict(policy=POLICY_VERSION, model=model, group=group, engine=engine,
                  burn=monitor.burn, check_every=monitor.check_every,
                  consecutive=monitor.consecutive, rules=monitor.rules,
                  status=status, history=monitor.hist,
                  interpretation="Single-ensemble split-Rhat and approximate pooled ESS; not independent-chain rank Rhat or bulk/tail ESS.")
    Path(str(path)+".convergence.json").write_text(json.dumps(serial(report), indent=2)+"\n")
    Path(str(path)+".convergence.txt").write_text(monitor.message(status, model, group, engine)+"\n")

def _acf_1d(x: np.ndarray) -> Optional[np.ndarray]:
    """Normalised autocorrelation function of a 1-D series (FFT), or None if it is constant."""
    x = np.asarray(x, dtype=float)
    n = 1
    while n < len(x):
        n <<= 1
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    if not (acf[0] > 0):
        return None
    return acf / acf[0]


def autocorr_time_mk(chain, c: float = 5.0) -> np.ndarray:
    """
    Integrated autocorrelation time of every parameter of an ensemble chain
    (n_steps, n_walkers, n_dim), with zeus's default estimator ("mk";
    Karamanis, Beutler & Peacock 2021; zeus.autocorr.AutoCorrTime): the walkers' chains
    are joined end to end, the autocorrelation function is taken about the
    overall mean and summed up to Sokal's automated window (M >= c tau).

    Unlike the emcee estimator, which averages the autocorrelation functions of
    the walkers about each walker's own mean, this keeps the differences between
    the walkers' means, so a parameter whose walkers drift slowly is not
    reported as well mixed. Returns NaN for a parameter that does not move.
    """
    x = np.asarray(chain, dtype=float)
    n, w, d = x.shape
    taus = np.full(d, np.nan)
    for k in range(d):
        f = _acf_1d(x[:, :, k].T.reshape(-1))      # walker after walker
        if f is None:
            continue
        t = 2.0 * np.cumsum(f) - 1.0
        m = np.arange(len(t)) < c * t
        taus[k] = t[int(np.argmin(m))] if np.any(m) else t[-1]
    return taus


def split_rhat(chain) -> np.ndarray:
    """
    Split-Rhat of every parameter (Gelman et al. 2013, Bayesian Data Analysis,
    Sec. 11.4): each walker's chain is cut into two halves and the
    2 x n_walkers half-chains are compared (between- and within-chain variance).
    Values above ~1.01 mean that walkers, or the first and second halves of the
    chain, still sample different regions.
    """
    x = np.asarray(chain, dtype=float)
    n, w, d = x.shape
    h = n // 2
    if h < 2:
        return np.full(d, np.nan)
    parts = np.concatenate([x[:h], x[h:2 * h]], axis=1)          # (h, 2w, d)
    means = parts.mean(axis=0)
    W = parts.var(axis=0, ddof=1).mean(axis=0)
    B = h * means.var(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(((h - 1) / h * W + B / h) / W)


def convergence_rules(PLOT_SETTINGS: Optional[dict] = None, convergence: Optional[float] = None) -> Dict[str, float]:
    """Thresholds of the stopping rule (constants, overridable through PLOT_SETTINGS)."""
    ps = PLOT_SETTINGS or {}
    return {
        "tau_factor": float(ps.get("conv_tau_factor", K.CONV_TAU_FACTOR)),
        "ess_min": float(ps.get("conv_ess_min", K.CONV_ESS_MIN)),
        "rhat_max": float(ps.get("conv_rhat_max", K.CONV_RHAT_MAX)),
        "rtol": float(convergence if convergence is not None else ps.get("conv_tau_rtol", K.CONV_TAU_RTOL)),
    }


def convergence_status(post_chain, rules: Dict[str, float], tau_prev: Optional[float] = None,
                       param_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """All quantities of the stopping rule for a post-burn-in chain (n_post, n_walkers, n_dim)."""
    x = np.asarray(post_chain, dtype=float)
    n, w, d = x.shape
    names = list(param_names) if param_names else [f"p{i}" for i in range(d)]
    tau = autocorr_time_mk(x)
    rhat = split_rhat(x)
    j = int(np.nanargmax(tau)) if np.isfinite(tau).any() else 0
    jr = int(np.nanargmax(rhat)) if np.isfinite(rhat).any() else 0
    tau_max = float(tau[j]) if np.isfinite(tau[j]) else float("nan")
    rhat_max = float(rhat[jr]) if np.isfinite(rhat[jr]) else float("nan")
    n_over = n / tau_max if tau_max > 0 else float("nan")
    ess = n * w / tau_max if tau_max > 0 else float("nan")
    rel = (abs(tau_max - tau_prev) / tau_max
           if (tau_prev is not None and np.isfinite(tau_prev) and tau_max > 0) else float("nan"))
    crit = {
        "length": bool(np.isfinite(n_over) and n_over >= rules["tau_factor"]),
        "ess": bool(np.isfinite(ess) and ess >= rules["ess_min"]),
        "stable": bool(np.isfinite(rel) and rel < rules["rtol"]),
        "rhat": bool(np.isfinite(rhat_max) and rhat_max < rules["rhat_max"]),
    }
    # A stalled/invalid parameter must not disappear through nanargmax.
    # This guards undefined diagnostics without changing finite upstream rules.
    if not (np.isfinite(x).all() and np.isfinite(tau).all()
            and np.all(tau > 0) and np.isfinite(rhat).all()):
        crit = {key: False for key in crit}
    return {
        "n_post": int(n), "walkers": int(w), "tau": tau, "tau_max": tau_max,
        "tau_param": names[j] if j < len(names) else f"p{j}",
        "n_over_tau": n_over, "ess": ess, "rel_change": rel,
        "rhat": rhat, "rhat_max": rhat_max, "rhat_param": names[jr] if jr < len(names) else f"p{jr}",
        "criteria": crit, "ok": all(crit.values()),
    }


class ConvergenceMonitor:
    """
    Applies the stopping rule during a run (zeus callback or emcee loop) and
    keeps the history that convergence_plot draws.

    update(it, chain) takes the GLOBAL chain up to step `it` (steps, walkers,
    dim), including steps from a resumed file, and returns True when the run
    may stop.
    """

    def __init__(self, burn: int, rules: Dict[str, float], check_every: int = 100,
                 earliest_stop: Optional[int] = None, consecutive: int = 1,
                 param_names: Optional[List[str]] = None):
        self.burn = int(burn)
        self.rules = dict(rules)
        self.check_every = max(1, int(check_every))
        self.earliest_stop = int(earliest_stop if earliest_stop is not None else burn)
        self.consecutive = max(1, int(consecutive))
        self.param_names = list(param_names) if param_names else None
        self.hist: Dict[str, List[Any]] = {k: [] for k in
                                           ("it", "tau_max", "tau_param", "n_post", "ess", "rel", "rhat_max", "ok")}
        # Burn-in phase, for the plot only (never used to stop): the same numbers on the
        # second half of the chain so far, as zeus's own callbacks do (discard = 0.5).
        self.pre: Dict[str, List[Any]] = {k: [] for k in ("it", "tau_max", "rel", "rhat_max")}
        self.walkers = None
        self.streak = 0
        self.stopped_at: Optional[int] = None
        self.ended_at: Optional[int] = None          # set by final()
        self.last: Optional[Dict[str, Any]] = None

    def _evaluate_burnin(self, it: int, chain) -> None:
        """Burn-in diagnostics for the plot: chain[it//2 : it], once that holds >= check_every steps."""
        if it < 2 * self.check_every:
            return
        try:
            part = np.asarray(chain[it // 2:it], dtype=float)
            prev = self.pre["tau_max"][-1] if self.pre["tau_max"] else None
            st = convergence_status(part, self.rules, prev, self.param_names)
        except Exception:
            return
        for k, v in (("it", it), ("tau_max", st["tau_max"]), ("rel", st["rel_change"]),
                     ("rhat_max", st["rhat_max"])):
            self.pre[k].append(v)

    def _evaluate(self, it: int, chain, full_interval: bool = False) -> Optional[Dict[str, Any]]:
        it = int(it)
        n_post = it - self.burn
        if n_post <= 0:
            self._evaluate_burnin(it, chain)
            return None
        if n_post < max(self.check_every, 20):
            return None
        post = np.asarray(chain[self.burn:it], dtype=float)
        prev = self.hist["tau_max"][-1] if self.hist["tau_max"] else None
        if full_interval:
            # the end of a run between two checks: tau's stability is judged against a check at
            # least one interval earlier, as at a regular check (not one a few steps back)
            prev = next((t for i_, t in zip(reversed(self.hist["it"]), reversed(self.hist["tau_max"]))
                         if it - int(i_) >= self.check_every), None)
        st = convergence_status(post, self.rules, prev, self.param_names)
        st["it"] = it
        self.walkers = st["walkers"]
        for k, v in (("it", it), ("tau_max", st["tau_max"]), ("tau_param", st["tau_param"]),
                     ("n_post", st["n_post"]), ("ess", st["ess"]), ("rel", st["rel_change"]),
                     ("rhat_max", st["rhat_max"]), ("ok", st["ok"])):
            self.hist[k].append(v)
        self.last = st
        return st

    def update(self, it: int, chain) -> bool:
        st = self._evaluate(it, chain)
        if st is None:
            return False
        self.streak = self.streak + 1 if st["ok"] else 0
        if self.stopped_at is None and self.streak >= self.consecutive and int(it) >= self.earliest_stop:
            self.stopped_at = int(it)
            return True
        return False

    def final(self, it: int, chain) -> Dict[str, Any]:
        """Status at the end of the run; 'converged' is True if the rule stopped the run or holds now
        (also after the earliest stop, as during the run)."""
        self.ended_at = int(it)
        if self.last is None or self.last.get("it") != int(it):
            st = self._evaluate(it, chain, full_interval=True)
            if st is not None:
                self.streak = self.streak + 1 if st["ok"] else 0
        st = dict(self.last or {})
        st["converged"] = bool(self.stopped_at is not None
                               or (st.get("ok", False) and self.streak >= self.consecutive
                                   and int(it) >= self.earliest_stop))
        st["stopped_at"] = self.stopped_at
        return st

    def steps_needed(self, st: Dict[str, Any]) -> Optional[int]:
        """Rough number of extra steps for the length and ESS conditions at the current tau_max."""
        tau, w = st.get("tau_max"), st.get("walkers") or self.walkers
        if not (tau and np.isfinite(tau) and w):
            return None
        need = max(self.rules["tau_factor"] * tau, self.rules["ess_min"] * tau / w)
        return int(max(0, np.ceil(need - st.get("n_post", 0))))

    def message(self, st: Dict[str, Any], model_name: str, group: str, engine: str) -> str:
        """One-line verdict with the numbers behind each condition."""
        r = self.rules
        if not st or "tau_max" not in st:
            return (f"[{model_name} | {group}] {engine}: the chain after burn-in is too short to estimate "
                    "the autocorrelation time. Treat these results as preliminary; increase nsteps.")
        c = st["criteria"]

        def mark(ok):
            return "ok" if ok else "NOT MET"
        rel = st["rel_change"]
        parts = [
            f"tau_max = {st['tau_max']:.1f} ({st['tau_param']}), N_post/tau = {st['n_over_tau']:.0f} "
            f"(needs >= {r['tau_factor']:.0f}: {mark(c['length'])})",
            f"ESS = {st['ess']:.0f} (needs >= {r['ess_min']:.0f}: {mark(c['ess'])})",
            ("tau_max change since the previous check = "
             + ("n/a" if not np.isfinite(rel) else f"{100 * rel:.1f}%")
             + f" (needs < {100 * r['rtol']:.0f}%: {mark(c['stable'])})"),
            f"split-Rhat = {st['rhat_max']:.4f} ({st['rhat_param']}) (needs < {r['rhat_max']:.2f}: {mark(c['rhat'])})",
        ]
        if st.get("converged"):
            where = (f"stopped at step {st['stopped_at']}" if st.get("stopped_at")
                     else f"met the rule at the end ({st['it']} steps)")
            return f"[{model_name} | {group}] {engine} converged, {where}: " + "; ".join(parts) + "."
        more = self.steps_needed(st)
        extra = f" Roughly {more} more steps are needed at this tau_max." if more else ""
        if not c["rhat"]:
            extra += " Split-Rhat falls as the walkers mix, so a longer chain is needed."
        return (f"[{model_name} | {group}] {engine} reached {st['it']} steps without meeting the "
                f"convergence rule: " + "; ".join(parts) + "." + extra
                + " Treat these results as preliminary; increase nsteps.")


class ZeusConvergenceCallback:
    """
    zeus callback: every `ncheck` steps it writes the new samples (append_writer),
    applies the convergence rule to the global chain (earlier steps of a resumed
    run + this run) and redraws the convergence plot. Also handles the one-time
    CMB precision switch and the optional log-posterior printout.
    """

    def __init__(self, monitor: "ConvergenceMonitor", ncheck: int, done: int = 0, prefix_chain=None,
                 plot_func=None, append_writer=None, precision_switch_iter: Optional[int] = None,
                 fine_kwargs: Optional[dict] = None, debug: bool = False):
        self.monitor = monitor
        self.ncheck = max(1, int(ncheck))
        self.done = int(done)
        self.prefix = None if prefix_chain is None else np.asarray(prefix_chain, dtype=float)
        self.plot_func = plot_func
        self.append_writer = append_writer
        self.precision_switch_iter = precision_switch_iter
        self.fine_kwargs = fine_kwargs
        self.debug = debug
        self._switched = False

    def __call__(self, iteration, chain, log_prob):
        if (self.precision_switch_iter is not None and not self._switched
                and iteration >= int(self.precision_switch_iter)):
            try:
                from Kosmulator_main import Statistical_packages as SP
                SP.set_precision("fine", **(self.fine_kwargs or {}))
                self._switched = True
            except Exception as e:
                if self.debug:
                    print(f"[Zeus] precision switch failed: {e}")

        if getattr(K, "print_loglike", False) and (iteration % max(1, getattr(K, "print_loglike_every", 1)) == 0):
            valid_lp = log_prob[np.isfinite(log_prob)] if log_prob is not None else np.array([])
            if valid_lp.size > 0:
                print(f"[Zeus Step {iteration}] Log-Post: Max={np.max(valid_lp):.4f} | "
                      f"Mean={np.mean(valid_lp):.4f}", flush=True)

        if iteration % self.ncheck != 0:
            return False
        if callable(self.append_writer):
            try:
                self.append_writer(iteration, chain, log_prob)
            except Exception:
                if self.debug:
                    print("[Zeus] append_writer raised; ignored.")
        full = chain if self.prefix is None else np.concatenate([self.prefix, chain], axis=0)
        stop = self.monitor.update(self.done + int(iteration), full)
        if callable(self.plot_func):
            try:
                self.plot_func(self.monitor)
            except Exception:
                if self.debug:
                    print("[Zeus] plot_func raised; ignored.")
        return stop


def make_zeus_callbacks(monitor: "ConvergenceMonitor", ncheck: int, done: int = 0, prefix_chain=None,
                        plot_func=None, append_writer=None, precision_switch_iter: Optional[int] = None,
                        fine_kwargs: Optional[dict] = None, debug: bool = False):
    """The zeus callback list used by Kosmulator (one ZeusConvergenceCallback)."""
    return [ZeusConvergenceCallback(monitor, ncheck, done, prefix_chain, plot_func, append_writer,
                                    precision_switch_iter, fine_kwargs, debug)]
