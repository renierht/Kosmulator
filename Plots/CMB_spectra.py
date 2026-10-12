"""
CMB band-power figure of the best-fit plots (item 57).

For a group with ACT, SPT, the Planck + ACT (+ SPT) combinations or a CMB lensing likelihood,
one figure with a panel per spectrum (TT, TE, EE and the lensing power spectrum), each with
the band powers of every CMB likelihood of the group (Planck high- and low-l, ACT DR6,
SPT-3G D1, ACT DR6 + Planck PR4, SPT-3G MUSE and Planck 2018 lensing), the median model, its
68% band from posterior draws, and below it the residuals in units of each band power's error.
The binned model values are the likelihoods' own (ground_cmb.band_powers: windows,
calibrations, lensing corrections), so (data - model)/error is what each likelihood sees,
up to the correlations between band powers.

Saved as best_fits/<group>/cmb_spectra.png next to bestfit.png.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from Kosmulator_main import constants as K

SPEC_ORDER = ("TT", "TE", "EE", "PP")
#: Colours by experiment (first match on the band powers' label)
EXPERIMENT_COLOURS = (("Planck low-l", "#17becf"), ("Planck PR4", "#9467bd"), ("Planck", "#1f77b4"),
                      ("ACT", "#ff7f0e"), ("SPT", "#2ca02c"))
#: Tags drawn in this figure (the best-fit figure keeps the Planck-only groups)
FIGURE_TAGS = tuple(K.GROUND_CMB_TAGS) + tuple(K.LENSING_TAGS)


def wants_figure(group: Sequence[str]) -> bool:
    """True if the group has a CMB likelihood this figure draws (ground-based or lensing)."""
    return any(str(t) in FIGURE_TAGS for t in group or ())


def _components(group: Sequence[str]) -> List[str]:
    """The group's CMB likelihoods in order, combinations expanded, each once."""
    out: List[str] = []
    for t in group or ():
        for c in K.cmb_components(str(t)):
            if c.startswith("CMB_") and c not in out:
                out.append(c)
    return out


def _colour(label: str) -> str:
    for key, col in EXPERIMENT_COLOURS:
        if label.startswith(key):
            return col
    return "#7f7f7f"


def _curves(cl: Dict[str, np.ndarray], ell_tt: np.ndarray, ell_pp: np.ndarray) -> Dict[str, np.ndarray]:
    """Unbinned model curves: D_l [muK^2] for TT, TE, EE; 10^7 [L(L+1)]^2 C_L^phiphi / 2 pi for PP."""
    from Kosmulator_main import ground_cmb as GC
    t2 = (K.T_CMB_DEFAULT * 1e6) ** 2
    out = {}
    for s in ("TT", "TE", "EE"):
        c = np.asarray(cl[s.lower()], float)
        e = ell_tt[ell_tt < c.size]
        out[s] = e * (e + 1) / (2 * np.pi) * c[e] * t2
    pp = np.asarray(cl["pp"], float)
    e = ell_pp[ell_pp < pp.size]
    out["PP"] = GC.PP_PLOT_SCALE * (e * (e + 1.0)) ** 2 / (2 * np.pi) * pp[e]
    return out


def cmb_spectra_figure(model_name: str, obs_key: str, group: Sequence[str], params_med: Dict[str, float],
                       band_draws: Optional[Sequence[Dict[str, float]]], PLOT_SETTINGS: dict,
                       get_cls: Callable[[Dict[str, float]], Optional[Dict[str, np.ndarray]]],
                       model_label: Optional[str] = None):
    """
    Draw the figure; returns it (the caller saves and closes it), or None when nothing can be
    drawn. get_cls(p): the lensed CLASS spectra of point p (dimensionless C_l from l = 0,
    keys tt, te, ee, bb, pp), the run's single CLASS run per point.
    """
    from Kosmulator_main import ground_cmb as GC
    cl = get_cls(params_med)
    if not isinstance(cl, dict):
        return None
    entries = []
    for tag in _components(group):
        try:
            entries += GC.band_powers(tag, cl, params_med, group)
        except Exception as e:                      # one missing data set must not stop the plots
            print(f"[CMB spectra] {tag}: band powers not drawn ({e})")
    specs = [s for s in SPEC_ORDER if any(e["spec"] == s for e in entries)]
    if not specs:
        return None

    lmax_cl = len(np.asarray(cl["tt"])) - 1
    ell_hi = max(float(np.max(e["ell"])) for e in entries if e["spec"] != "PP") if any(
        e["spec"] != "PP" for e in entries) else 2500.0
    ell_tt = np.unique(np.round(np.geomspace(2, min(lmax_cl, 1.1 * ell_hi), 600)).astype(int))
    ell_pp = np.unique(np.round(np.geomspace(2, min(lmax_cl, 3100), 400)).astype(int))
    med = _curves(cl, ell_tt, ell_pp)

    # 68% band of the model curves from the posterior draws (their spectra are computed ahead
    # through the worker pool by the best-fit plots, item 52)
    band = None
    if band_draws:
        rows = []
        for p in band_draws:
            c = get_cls(p)
            if isinstance(c, dict):
                cv = _curves(c, ell_tt, ell_pp)
                rows.append(np.concatenate([cv[s] for s in specs]))
        if len(rows) >= max(10, len(band_draws) // 2):
            rows = np.vstack(rows)
            band = (np.percentile(rows, 16, axis=0), np.percentile(rows, 84, axis=0))

    label = model_label or model_name
    fig = plt.figure(figsize=(10.0, 3.6 * len(specs)))
    outer = fig.add_gridspec(nrows=len(specs), ncols=1, hspace=0.32, top=1.0 - 0.05 / len(specs))
    off = 0
    legend: Dict[str, object] = {}
    for i, s in enumerate(specs):
        sub = outer[i].subgridspec(nrows=2, ncols=1, height_ratios=(3.0, 1.1), hspace=0.0)
        ax = fig.add_subplot(sub[0, 0])
        axr = fig.add_subplot(sub[1, 0], sharex=ax)
        x = ell_pp if s == "PP" else ell_tt
        y = med[s]
        n = y.size
        if band is not None:
            ax.fill_between(x[:n], band[0][off:off + n], band[1][off:off + n], color="k", alpha=0.15,
                            linewidth=0, zorder=5, label="68% band")
        off += n
        ax.plot(x[:n], y, color="r", lw=1.6, zorder=6, label=f"{label} (median)")
        axr.axhline(0.0, color="0.4", lw=0.9)
        for h in (-2.0, 2.0):
            axr.axhline(h, color="0.7", lw=0.7, ls=":")
        xs = []
        for e in (e for e in entries if e["spec"] == s):
            col = _colour(e["label"])
            yerr = e.get("err_lohi", e["err"])
            ax.errorbar(e["ell"], e["y"], yerr=yerr, fmt="o", ms=2.6, lw=0.8, color=col, ecolor=col,
                        capsize=0, alpha=0.9, zorder=3, label=e["label"])
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (np.asarray(e["y"]) - np.asarray(e["model"])) / np.asarray(e["err"])
            axr.plot(e["ell"], r, "o", ms=2.4, color=col, alpha=0.9)
            xs.append(np.asarray(e["ell"], float))
        allx = np.concatenate(xs)
        ax.set_xscale("log")
        ax.set_xlim(max(1.5, 0.85 * allx.min()), 1.15 * allx.max())
        if s != "PP":
            # y range from the model: the noisiest band powers (Planck EE at l > 1000) would
            # otherwise set it; points outside are still in the residual panel
            inx = (x[:n] >= ax.get_xlim()[0]) & (x[:n] <= ax.get_xlim()[1])
            lo, hi = float(np.min(y[inx])), float(np.max(y[inx]))
            pad = 0.25 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(r"$10^7\,[L(L+1)]^2 C_L^{\phi\phi}/2\pi$" if s == "PP" else rf"$D_\ell^{{{s}}}$ [$\mu$K$^2$]")
        axr.set_ylabel(r"$\Delta/\sigma$")
        axr.set_ylim(-5, 5)
        axr.set_yticks((-3, 0, 3))
        ax.tick_params(labelbottom=False)
        axr.set_xlabel(r"$L$" if s == "PP" else r"Multipole $\ell$")
        for h, l in zip(*ax.get_legend_handles_labels()):
            legend.setdefault(l, h)
    # One legend for the figure: the same experiments and model in every panel
    fig.legend(list(legend.values()), list(legend.keys()), loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=min(4, len(legend)), fontsize=8.5, frameon=False)
    return fig
