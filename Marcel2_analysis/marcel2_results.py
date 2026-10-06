"""
Information criteria and parameter constraints for the Marcel_2 chains.

Reads only flat_samples/*.npz (made by marcel2_chains.py --export). No Kosmulator
imports. Writes results/ic_table.csv, results/parameters.csv and results/summary.md.

Conventions
-----------
log_prob equals the stored blobs in every chain, so it is treated as ln L (flat priors).
D = -2 ln L.
DIC (Kosmulator/Spiegelhalter-MAP): D_hat = -2 max(ln L), pD = Dbar - D_hat, DIC = D_hat + 2 pD.
DIC (Gelman): pD_V = Var(D)/2, DIC = Dbar + pD_V. Independent cross-check of pD.
AIC = 2k - 2 lnLmax ;  AICc adds 2k(k+1)/(N-k-1) ;  BIC = k ln N - 2 lnLmax.
k = number of sampled parameters (ndim). N is an ASSUMED data-point count, see N_DATA.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

HERE = Path(__file__).parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

# N = 1701 (PantheonP.dat rows, includes SH0ES calibrators) + 13 (DESI_DR2.txt data rows),
# following Kosmulator's len(m_b_corr) and len(redshift). UNVERIFIED against a Kosmulator run.
N_DATA = 1701 + 13

# Chains too short to analyse (4 and 149 steps) and with a different dataset.
SKIP = ("BBN_PryMordial",)

# Reference model for each comparison group (same dataset, same prior family).
# IDE_CLASS_Included has no LCDM counterpart, so it is compared to its own Unrestricted variant.
def group_and_model(stem):
    p = stem.split("__")
    if p[0] == "IDE_CLASS_Included":
        return "CLASS_included", p[1]
    return p[1], p[2]                      # e.g. ("Free", "LCDM_v")


def significance(dchi, r):
    """Equivalent Gaussian significance of dchi2 for r extra degrees of freedom.
    Same as Kosmulator's Post_processing.significance (r rounded to an integer), except:
    abs(r) is used, so the sign of the comparison does not matter, and r = 0 (or NaN)
    returns NaN rather than 0, because chi2 with zero degrees of freedom is undefined."""
    r = np.round(abs(r))
    if np.isnan(dchi) or np.isnan(r) or r < 1:
        return np.nan
    p = chi2.sf(abs(dchi), df=r)
    return np.round(norm.isf(p / 2), 1)


def analyse(path):
    d = np.load(path)
    s, lp, names = d["samples"], d["log_prob"], [str(n) for n in d["names"]]
    k = s.shape[1]                         # = len(names); ndim is not stored in the npz
    D = -2.0 * lp
    Dbar, Dhat = D.mean(), D.min()
    pD_map = Dbar - Dhat
    pD_var = D.var(ddof=1) / 2.0
    lnL = lp.max()
    aic = 2 * k - 2 * lnL
    aicc = aic + 2 * k * (k + 1) / (N_DATA - k - 1)
    bic = k * np.log(N_DATA) - 2 * lnL
    row = dict(k=k, n_samples=len(lp), chi2_min=Dhat, Dbar=Dbar,
               pD_MAP=pD_map, pD_Gelman=pD_var, DIC=Dhat + 2 * pD_map,
               DIC_Gelman=Dbar + pD_var, AIC=aic, AICc=aicc, BIC=bic)
    q = np.percentile(s, [16, 50, 84], axis=0)
    par = [dict(parameter=n, median=q[1, i], minus=q[1, i] - q[0, i], plus=q[2, i] - q[1, i])
           for i, n in enumerate(names)]
    return row, par


rows, pars = [], []
for f in sorted((HERE / "flat_samples").glob("*.npz")):
    if any(t in f.stem for t in SKIP):
        continue
    group, model = group_and_model(f.stem)
    r, p = analyse(f)
    rows.append(dict(group=group, model=model, **r))
    pars += [dict(group=group, model=model, **x) for x in p]

ic = pd.DataFrame(rows)
# Differences relative to the reference model of each group.
ref = {"CLASS_included": "Unrestricted (iwCDM)"}
for g, sub in ic.groupby("group"):
    r = ref.get(g, "LCDM_v")
    base = sub[sub.model == r].iloc[0]
    for c in ("DIC", "DIC_Gelman", "AIC", "AICc", "BIC"):
        ic.loc[sub.index, "d" + c] = sub[c] - base[c]
    ic.loc[sub.index, "dchi2"] = sub["chi2_min"] - base["chi2_min"]
    ic.loc[sub.index, "dk"] = sub["k"] - base["k"]
    ic.loc[sub.index, "dpD"] = sub["pD_MAP"] - base["pD_MAP"]
    ic.loc[sub.index, "reference"] = r
ic["sigma_pD"] = [significance(a, b) for a, b in zip(ic.dchi2, ic.dpD)]   # Kosmulator definition
ic["sigma_k"] = [significance(a, b) for a, b in zip(ic.dchi2, ic.dk)]     # Wilks, r = delta k
ic = ic.round(3)
par = pd.DataFrame(pars).round(4)
ic.to_csv(OUT / "ic_table.csv", index=False)
par.to_csv(OUT / "parameters.csv", index=False)

pd.set_option("display.width", 250, "display.max_columns", 30)
cols = ["group", "model", "k", "chi2_min", "dchi2", "dk", "dpD", "sigma_pD", "sigma_k",
        "pD_MAP", "DIC", "dDIC", "dDIC_Gelman", "dAIC", "dAICc", "dBIC"]
txt = ic[cols].to_string(index=False)
print(txt)
print()
print(par.to_string(index=False))
(OUT / "summary.md").write_text(
    f"N_DATA assumed = {N_DATA}\n\n```\n{txt}\n```\n\n```\n{par.to_string(index=False)}\n```\n")


# ---------------------------------------------------------------- LaTeX tables
def tex(s):
    return str(s).replace("_", r"\_")


def num(x, d=2):
    return "n/a" if pd.isna(x) else f"{x:.{d}f}"


GROUP_TITLE = {"CLASS_included": "CLASS included (reference: Unrestricted)",
               "Free": r"Free $r_d$, Free (reference: $\Lambda$CDM)",
               "Positive": r"Free $r_d$, Positive (reference: $\Lambda$CDM)",
               "Positive_stable": r"Free $r_d$, Positive stable (reference: $\Lambda$CDM)"}

lines = [r"\begin{table}[htbp]", r"\centering", r"\small", r"\begin{tabular}{lcrrrrrrrr}", r"\toprule",
         r"Model & $k$ & $\chi^2_{\min}$ & $\Delta\chi^2$ & $p_D$ & DIC & $\Delta$DIC & $\Delta$AIC & $\Delta$BIC & $\sigma$ \\",
         r"\midrule"]
for g, sub in ic.groupby("group", sort=False):
    lines.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{GROUP_TITLE.get(g, tex(g))}}}}} \\")
    for _, r in sub.iterrows():
        is_ref = r.model == r.reference
        lines.append(" & ".join([tex(r.model), f"{int(r.k)}", num(r.chi2_min, 1),
                                 "ref." if is_ref else num(r.dchi2, 2), num(r.pD_MAP, 2), num(r.DIC, 1),
                                 "ref." if is_ref else num(r.dDIC, 2), "ref." if is_ref else num(r.dAIC, 2),
                                 "ref." if is_ref else num(r.dBIC, 2),
                                 "ref." if is_ref else num(r.sigma_pD, 1)]) + r" \\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines += [r"\end{tabular}",
          r"\caption{Information criteria and $\Delta\chi^2$ significance for the Marcel\_2 chains "
          rf"(DESI DR2 + Pantheon+\,\&\,SH0ES), assuming $N={N_DATA}$ data points. $\chi^2_{{\min}}=-2\ln\mathcal{{L}}_{{\max}}$, "
          r"$p_D=\bar D-\hat D$, and $\sigma$ is the equivalent Gaussian significance of $\Delta\chi^2$ for "
          r"$r=\Delta p_D$ (rounded) degrees of freedom. Differences are relative to the stated reference model.}",
          r"\label{tab:marcel2_ic}", r"\end{table}"]
(OUT / "ic_table.tex").write_text("\n".join(lines) + "\n")

PLAB = {"Omega_m": r"$\Omega_m$", "w": r"$w$", "delta": r"$\delta$", "H_0": r"$H_0$", "r_d": r"$r_d$", "M": r"$M$"}
plist = ["Omega_m", "w", "delta", "H_0", "r_d", "M"]
dec = {"Omega_m": 3, "w": 3, "delta": 3, "H_0": 2, "r_d": 1, "M": 3}
pl = [r"\begin{table}[htbp]", r"\centering", r"\small", r"\resizebox{\textwidth}{!}{%",
      r"\begin{tabular}{l" + "c" * len(plist) + "}", r"\toprule",
      "Model & " + " & ".join(PLAB[p] for p in plist) + r" \\", r"\midrule"]
for (g, m), sub in par.groupby(["group", "model"], sort=False):
    cells = []
    for p in plist:
        x = sub[sub.parameter == p]
        if x.empty:
            cells.append("n/a")
        else:
            x = x.iloc[0]; d = dec[p]
            cells.append(rf"${x['median']:.{d}f}^{{+{x['plus']:.{d}f}}}_{{-{x['minus']:.{d}f}}}$")
    pl.append(tex(f"{g}: {m}") + " & " + " & ".join(cells) + r" \\")
pl += [r"\bottomrule", r"\end{tabular}}",
       r"\caption{Posterior medians with 68\% (16th - 84th percentile) intervals for the Marcel\_2 chains. "
       r"Parameter names are inferred from the chain columns; $H_0$ in km\,s$^{-1}$\,Mpc$^{-1}$, $r_d$ in Mpc.}",
       r"\label{tab:marcel2_params}", r"\end{table}"]
(OUT / "parameters_table.tex").write_text("\n".join(pl) + "\n")
print("LaTeX written to", OUT)
