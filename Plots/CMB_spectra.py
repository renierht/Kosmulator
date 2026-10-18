"""
CMB band-power figure of the best-fit plots (item 57; every CMB group since CT-34).

For every group with CMB data (Planck-only groups too), one figure with a panel per spectrum (TT, TE, EE and the lensing power spectrum), each with
the band powers of every CMB likelihood of the group (Planck high- and low-l, ACT DR6,
SPT-3G D1, ACT DR6 + Planck PR4, SPT-3G MUSE and Planck 2018 lensing), the median model, its
68% band from posterior draws, and below it the residuals in units of each band power's error.
The binned model values are the likelihoods' own (ground_cmb.band_powers: windows,
calibrations, lensing corrections). The residuals are (data - model)/error for the Gaussian
likelihoods (the correlations between band powers are not shown), the signed square root of
each multipole's chi^2 for the tabulated low-l EE (SimAll, SRoll2), and against the error on
the model's side for Planck's low-l TT band powers. Residuals beyond the panel are drawn at
its edge.

Colours (CT-41): one per likelihood of the group, in order, from the run's palette as in bestfit.png
(constants.OBS_COLOR_ORDER, without colours close to the model's), the model in constants.MODEL_COLOR,
its band with PLOT_SETTINGS["model_band_alpha"]. The y-range of TT, TE and EE covers the model and
the band powers whose error bar is smaller than the model's range there (CT-34 c).

Saved as best_fits/<group>/cmb_spectra.png; bestfit.png keeps the group's other data.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from Kosmulator_main import constants as K

SPEC_ORDER = ("TT", "TE", "EE", "PP")
#: Tags drawn in this figure: every CMB likelihood (CT-34; before, the Planck-only groups were in
#: bestfit.png)
FIGURE_TAGS = tuple(K.CMB_TAGS)


def wants_figure(group: Sequence[str]) -> bool:
    """True if the group has a CMB likelihood."""
    return any(str(t) in FIGURE_TAGS or str(t).startswith("CMB_") for t in group or ())


def _components(group: Sequence[str]) -> List[str]:
    """The group's CMB likelihoods in order, combinations expanded, each once."""
    out: List[str] = []
    for t in group or ():
        for c in K.cmb_components(str(t)):
            if c.startswith("CMB_") and c not in out:
                out.append(c)
    return out


def _data_colours(labels) -> Dict[str, str]:
    """One colour per likelihood of the group, in order, from the run's palette as in bestfit.png
    (constants.OBS_COLOR_ORDER), without colours close to the model's (constants.MODEL_COLOR)."""
    import matplotlib.colors as mc
    m = np.array(mc.to_rgb(K.MODEL_COLOR))
    # (and none too light to see on white, e.g. yellow)
    lum = lambda c: float(np.dot(mc.to_rgb(c), (0.299, 0.587, 0.114)))
    pal = [c for c in K.OBS_COLOR_ORDER
           if np.linalg.norm(np.array(mc.to_rgb(c)) - m) > 0.35 and lum(c) < 0.8] or ["b"]
    out: Dict[str, str] = {}
    for lab in labels:
        out.setdefault(lab, pal[len(out) % len(pal)])
    return out


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
    cl = get_cls(params_med)
    if not isinstance(cl, dict):
        print(f"[CMB spectra] {model_name} {obs_key}: no CLASS spectra at the median; figure not drawn")
        return None
    from Kosmulator_main import ground_cmb as GC
    entries = []
    for tag in _components(group):
        try:
            entries += GC.band_powers(tag, cl, params_med, group)
        except Exception as e:                      # one missing data set must not stop the plots
            need = (" (Planck's high-l band powers are read from Observations/plik_lite_v22_TTTEEE.clik, "
                    "also when the run uses full plik)") if tag in ("CMB_hil", "CMB_hil_TT") else ""
            print(f"[CMB spectra] {tag}: band powers not drawn ({e}){need}")
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
    try:
        _panels(fig, specs, entries, med, band, ell_tt, ell_pp, label,
                band_alpha=float(PLOT_SETTINGS.get("model_band_alpha", 0.15)))
    except Exception:
        plt.close(fig)                       # the caller only reports the error
        raise
    return fig


def _panels(fig, specs, entries, med, band, ell_tt, ell_pp, label, band_alpha=0.15) -> None:
    colours = _data_colours([e["label"] for e in entries])
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
            ax.fill_between(x[:n], band[0][off:off + n], band[1][off:off + n], color=K.MODEL_COLOR,
                            alpha=band_alpha, linewidth=0, zorder=5, label="68% band")
        off += n
        ax.plot(x[:n], y, color=K.MODEL_COLOR, lw=1.6, zorder=6, label=f"{label} (median)")
        axr.axhline(0.0, color="0.4", lw=0.9)
        for h in (-2.0, 2.0):
            axr.axhline(h, color="0.7", lw=0.7, ls=":")
        xs, pts = [], []
        for e in (e for e in entries if e["spec"] == s):
            col = colours[e["label"]]
            yerr = e.get("err_lohi", e["err"])
            ax.errorbar(e["ell"], e["y"], yerr=yerr, fmt="o", ms=2.6, lw=0.8, color=col, ecolor=col,
                        capsize=0, alpha=0.9, zorder=3, label=e["label"])
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.asarray(e["resid"], float) if "resid" in e else \
                    (np.asarray(e["y"]) - np.asarray(e["model"])) / np.asarray(e["err"])
            axr.plot(e["ell"], np.clip(r, -4.8, 4.8), "o", ms=2.4, color=col, alpha=0.9)
            xs.append(np.asarray(e["ell"], float))
            _err = np.asarray(e["err"], float)
            pts.append((np.asarray(e["ell"], float), np.asarray(e["y"], float),
                        _err if _err.ndim == 1 else np.max(_err, axis=0)))
        allx = np.concatenate(xs)
        ax.set_xscale("log")
        ax.set_xlim(max(1.5, 0.85 * allx.min()), 1.15 * allx.max())
        if s != "PP":
            # y range from the model, extended to the band powers whose error bar is smaller than the
            # model's range (CT-34 c: SRoll2 at l = 2 and low-l TT at l = 2, 5 were cut off); the noisiest
            # ones (Planck EE at l > 1000) still do not set it and are in the residual panel
            inx = (x[:n] >= ax.get_xlim()[0]) & (x[:n] <= ax.get_xlim()[1])
            lo, hi = float(np.min(y[inx])), float(np.max(y[inx]))
            span = hi - lo
            for ex, ey, ee in pts:
                ok = np.isfinite(ey) & np.isfinite(ee) & (ee < span)
                if ok.any():
                    lo = min(lo, float(np.min(ey[ok] - ee[ok])))
                    hi = max(hi, float(np.max(ey[ok] + ee[ok])))
                low = np.isfinite(ey) & (ex <= 30)          # the low-l points themselves, always
                if low.any():
                    lo, hi = min(lo, float(np.min(ey[low]))), max(hi, float(np.max(ey[low])))
            pad = 0.08 * (hi - lo) if (hi - lo) > span else 0.25 * span
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
