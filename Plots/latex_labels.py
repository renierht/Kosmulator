"""
latex_labels.py: robust LaTeX labels for parameters, models and datasets.

Kosmulator lets users name parameters and models freely (``Omega_m``,
``w0``, ``alpha_JLA``, ``ps_A_100_100``, ``MyModel_v``, ...). This module
turns any such name into a label that compiles with real LaTeX
(``--latex_enabled``) and with Matplotlib's mathtext, following the usual
cosmology conventions:

* Greek names become symbols wherever they stand: ``Omega`` -> \\Omega,
  ``sigma8`` -> \\sigma_8, ``Omegam`` -> \\Omega_{\\rm m}.
* A single letter or Greek symbol with trailing digits gets them as a
  subscript: ``w0`` -> w_0, ``H0`` -> H_0, ``f1`` -> f_1.
* Text after ``_`` is a subscript; descriptive words are upright:
  ``Omega_m`` -> \\Omega_{\\rm m}, ``r_d`` -> r_{\\rm d},
  ``alpha_JLA`` -> \\alpha_{\\rm JLA}, ``H_0`` -> H_0.
* With several ``_`` the first single letter or Greek name is the symbol,
  words before it become a superscript and the rest a subscript:
  ``ps_A_100_100`` -> A^{\\rm ps}_{100,100}.
* ``...h^2`` and ``^n`` powers are kept: ``Omega_bh^2`` -> \\Omega_{\\rm b}h^2.
* Everything else is set upright with LaTeX special characters escaped.

Each label is test-compiled once (LaTeX when ``text.usetex`` is on, else
mathtext); a label that fails falls back to an escaped upright version, so a
strange name can no longer stop the plotting stage.

Users can add exact labels to ``PARAM_LATEX_OVERRIDES`` (parameters) or
``PLOT_SETTINGS["model_latex_names"]`` (models).
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Symbol tables
# ---------------------------------------------------------------------------

_GREEK_LOWER = [
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta",
    "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi",
    "varpi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon", "phi",
    "varphi", "chi", "psi", "omega",
]
_GREEK_UPPER_CMD = ["Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma",
                    "Upsilon", "Phi", "Psi", "Omega"]
# Capitals without a LaTeX command look like Latin letters (set upright)
_GREEK_UPPER_LATIN = {"Alpha": "A", "Beta": "B", "Epsilon": "E", "Zeta": "Z",
                      "Eta": "H", "Iota": "I", "Kappa": "K", "Mu": "M", "Nu": "N",
                      "Omicron": "O", "Rho": "P", "Tau": "T", "Chi": "X"}

GREEK: Dict[str, str] = {g: "\\" + g for g in _GREEK_LOWER + _GREEK_UPPER_CMD}
GREEK.update({k: r"\mathrm{%s}" % v for k, v in _GREEK_UPPER_LATIN.items()})
GREEK["ell"] = r"\ell"
GREEK["omicron"] = "o"

_UNICODE_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho",
    "σ": "sigma", "ς": "varsigma", "τ": "tau", "υ": "upsilon", "φ": "phi",
    "χ": "chi", "ψ": "psi", "ω": "omega", "Γ": "Gamma", "Δ": "Delta",
    "Θ": "Theta", "Λ": "Lambda", "Ξ": "Xi", "Π": "Pi", "Σ": "Sigma",
    "Υ": "Upsilon", "Φ": "Phi", "Ψ": "Psi", "Ω": "Omega", "ℓ": "ell",
}
_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

#: Exact labels for common cosmology names whose structure cannot be guessed
KNOWN_PARAMS: Dict[str, str] = {
    "Omega_bh^2": r"\Omega_{\mathrm{b}} h^2",
    "Omega_ch^2": r"\Omega_{\mathrm{c}} h^2",
    "Omega_dh^2": r"\Omega_{\mathrm{d}} h^2",
    "omegabh2": r"\Omega_{\mathrm{b}} h^2",
    "omegach2": r"\Omega_{\mathrm{c}} h^2",
    "ln10^10_As": r"\ln(10^{10} A_{\mathrm{s}})",
    "logA": r"\ln(10^{10} A_{\mathrm{s}})",
    "100theta_s": r"100\,\theta_{\mathrm{s}}",
    "theta_s": r"\theta_{\mathrm{s}}",
    "A_planck": r"A_{\mathrm{Planck}}",
    "wa": r"w_a",
    "hrd": r"h\,r_{\mathrm{d}}",
    "Neff": r"N_{\mathrm{eff}}",
    "As": r"A_{\mathrm{s}}",
    "ns": r"n_{\mathrm{s}}",
    "S8": r"S_8",
    "mnu": r"m_\nu",
    "Mb": r"M_B",
    "M_B": r"M_B",
    "eps": r"\epsilon",
}

#: Known cores of model names (before "CDM"), e.g. "wowa" in "wowaCDM"
_MODEL_PREFIX = {
    "": "", "L": r"\Lambda", "Lambda": r"\Lambda", "w": "w", "wowa": "w_0w_a",
    "w0wa": "w_0w_a", "fR": "f(R)", "fT": "f(T)", "fQ": "f(Q)",
}

_TEXT_ESC = {"\\": r"\textbackslash{}", "_": r"\_", "&": r"\&", "%": r"\%",
             "$": r"\$", "#": r"\#", "{": r"\{", "}": r"\}",
             "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
_MATH_ESC = {"\\": r"\backslash ", "_": r"\_", "&": r"\&", "%": r"\%",
             "$": r"\$", "#": r"\#", "{": r"\{", "}": r"\}", "~": r"\sim ",
             "^": r"\wedge ", " ": r"\ "}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _ascii(s: str) -> str:
    """Unicode Greek to names, sub/superscript digits to digits, drop other non-ASCII."""
    s = s.translate(_SUBSCRIPT_DIGITS).translate(_SUPERSCRIPT_DIGITS)
    out = []
    for ch in s:
        if ch in _UNICODE_GREEK:
            out.append(_UNICODE_GREEK[ch])
        elif ord(ch) < 128:
            out.append(ch)
        else:
            out.append(unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode())
    return "".join(out)


def _mathrm(text: str) -> str:
    """Upright text inside math mode, specials escaped for LaTeX or mathtext."""
    if _usetex():
        body = "".join(_MATH_ESC.get(c, c) for c in text)
    else:
        # mathtext knows neither \& nor \#: keep & as is, drop #
        body = "".join("" if c == "#" else ("&" if c == "&" else _MATH_ESC.get(c, c)) for c in text)
    return r"\mathrm{" + body + "}"


def tex_escape_text(text: str) -> str:
    """Escape a plain string for LaTeX text mode (outside $...$)."""
    return "".join(_TEXT_ESC.get(c, c) for c in str(text))


def _greek_split(tok: str):
    """(greek_name, remainder) if tok starts with a Greek name, else (None, tok)."""
    for g in sorted(GREEK, key=len, reverse=True):
        if tok.startswith(g):
            return g, tok[len(g):]
    return None, tok


def _symbol(tok: str) -> str:
    """LaTeX for a base symbol token (no underscores)."""
    if tok in KNOWN_PARAMS:
        return KNOWN_PARAMS[tok]
    if tok in GREEK:
        return GREEK[tok]
    m = re.fullmatch(r"(\d+)([A-Za-z].*)", tok)                 # 100theta -> 100\,\theta
    if m:
        return m.group(1) + r"\," + _symbol(m.group(2))
    m = re.fullmatch(r"([A-Za-z]+?)(\d+)", tok)                 # w0, sigma8, H0, gal545
    if m:
        head, digits = m.groups()
        if len(head) == 1 or head in GREEK:
            return f"{_symbol(head)}_{{{digits}}}"
        return _mathrm(tok)
    if re.fullmatch(r"[A-Za-z]", tok):
        return tok
    parts, rest_ = [], tok                                      # alphabeta -> \alpha\beta
    while rest_:
        g, rest_new = _greek_split(rest_)
        if not g:
            break
        parts.append(GREEK[g])
        rest_ = rest_new
    if parts and not rest_ and len(parts) > 1:
        return " ".join(parts)
    g, rest = _greek_split(tok)                                 # Omegam, omegab, Omegade
    if g and rest and len(g) >= 3 and re.fullmatch(r"[A-Za-z0-9]{1,2}", rest):
        sub = rest if rest.isdigit() else _mathrm(rest)
        return f"{GREEK[g]}_{{{sub}}}"
    return _mathrm(tok)


def _is_symbol_token(tok: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]", tok)) or tok in GREEK or bool(
        re.fullmatch(r"([A-Za-z]|%s)\d+" % "|".join(GREEK), tok))


def _script(tok: str) -> str:
    """LaTeX for a token used in a sub- or superscript."""
    if tok.isdigit():
        return tok
    if tok in GREEK:
        return GREEK[tok]
    m = re.fullmatch(r"([A-Za-z]+?)(\d+)", tok)
    if m and m.group(1) in GREEK:                               # sigma8 inside f_sigma8
        return f"{GREEK[m.group(1)]}_{{{m.group(2)}}}"
    return _mathrm(tok)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def _param_core(name: str, overrides: Optional[Dict[str, str]] = None) -> str:
    overrides = overrides or {}
    s = name.strip()
    if s in overrides:
        return overrides[s]
    if len(s) > 1 and s.startswith("$") and s.endswith("$"):
        return s[1:-1]                                          # already LaTeX
    if "\\" in s and _compiles(f"${s}$", _usetex()):
        return s                                                # user wrote LaTeX
    s = _ascii(s).replace("{", "").replace("}", "")
    if s in overrides:
        return overrides[s]
    if s in KNOWN_PARAMS:
        return KNOWN_PARAMS[s]
    if not s:
        return _mathrm("?")

    # trailing note in brackets: "Omega_m (derived)"
    m = re.fullmatch(r"(.+?)\s+(\(.*\))", s)
    if m:
        return _param_core(m.group(1), overrides) + r"\ " + _mathrm(m.group(2))

    # ... h^2 (physical densities) and general powers x^n
    m = re.fullmatch(r"(.+_[A-Za-z]+)h\^?2", s)
    if m:
        return _param_core(m.group(1), overrides) + r" h^2"
    if "^" in s:
        body, power = s.rsplit("^", 1)
        if body and re.fullmatch(r"[A-Za-z0-9.+-]+", power):
            p = power if re.fullmatch(r"[0-9.+-]+", power) else _mathrm(power)
            inner = _param_core(body, overrides)
            return f"{{{inner}}}^{{{p}}}"

    toks = [t for t in s.split("_") if t != ""]
    # Greek name followed by a number is one unit: sigma_8 -> \sigma_8
    merged = []
    for t in toks:
        if merged and t.isdigit() and merged[-1] in GREEK:
            merged[-1] = merged[-1] + t
        else:
            merged.append(t)
    toks = merged

    if len(toks) == 1:
        return _symbol(toks[0])

    # log10_X, log_X, ln_X
    if toks[0] in ("log10", "log", "ln"):
        fn = {"log10": r"\log_{10}", "log": r"\log", "ln": r"\ln"}[toks[0]]
        return fn + " " + _param_core("_".join(toks[1:]), overrides)

    if len(toks) == 2:
        base, sup, sub = toks[0], [], [toks[1]]
    else:
        idx = next((i for i, t in enumerate(toks) if _is_symbol_token(t)), 0)
        base, sup, sub = toks[idx], toks[:idx], toks[idx + 1:]

    # Amplitude-style names (A_cib_217, A_sbpx_100_100_TT): for a single capital
    # letter followed by both words and numbers, words go up, numbers down.
    if (re.fullmatch(r"[A-Z]", base) and not sup
            and any(t.isdigit() for t in sub) and any(not t.isdigit() for t in sub)):
        sup = [t for t in sub if not t.isdigit()]
        sub = [t for t in sub if t.isdigit()]

    out = _symbol(base)
    if (sub or sup) and ("_" in out or "^" in out):
        out = "{" + out + "}"                                   # sigma8_z0 -> {\sigma_8}_{z0}
    if sub:
        out = f"{out}_{{{','.join(_script(t) for t in sub)}}}"
    if sup:
        out = f"{out}^{{{','.join(_script(t) for t in sup)}}}"
    return out


@lru_cache(maxsize=4096)
def _compiles(label: str, usetex: bool) -> bool:
    """True if Matplotlib can render the label (LaTeX or mathtext)."""
    try:
        if usetex:
            from matplotlib.texmanager import TexManager
            TexManager().get_text_width_height_descent(label, 10)
        else:
            from matplotlib.mathtext import MathTextParser
            MathTextParser("path").parse(label)
        return True
    except Exception:
        return False


def _usetex() -> bool:
    try:
        import matplotlib
        return bool(matplotlib.rcParams.get("text.usetex", False))
    except Exception:
        return False


def param_latex(name: str, overrides: Optional[Dict[str, str]] = None,
                check: bool = True) -> str:
    """
    Math-mode LaTeX (without the surrounding $) for a parameter name.

    ``overrides`` maps exact names to labels and wins over every rule.
    With ``check`` the label is test-compiled once and replaced by an escaped
    upright version if it does not compile.
    """
    label = _param_core(str(name), overrides)
    if not check:
        return label
    tex = _usetex()
    if _compiles(f"${label}$", tex):
        return label
    fallback = _mathrm(_ascii(str(name)))
    return fallback if _compiles(f"${fallback}$", tex) else r"\mathrm{param}"


# ---------------------------------------------------------------------------
# Models and datasets
# ---------------------------------------------------------------------------

def model_label(name: str, latex_on: bool = True,
                overrides: Optional[Dict[str, str]] = None) -> str:
    """
    Display label for a model name, e.g. LCDM_v -> $\\Lambda$CDM,
    wowaCDM_v -> $w_0w_a$CDM, f1CDM_nv -> $f_1$CDM (nv),
    NonLinear_IDE_2 -> NonLinear IDE 2. A trailing ``_v`` (the vectorised
    implementation, Kosmulator's default) is dropped; other suffixes are shown
    in brackets. Without LaTeX the name is returned unchanged.
    """
    name = str(name)
    if not latex_on:
        return name
    overrides = overrides or {}
    if name in overrides:
        s = overrides[name]
        return s if "$" in s else f"${s}$"
    if "$" in name:
        return name

    toks = [t for t in _ascii(name).split("_") if t]
    if not toks:
        return tex_escape_text(name) if _usetex() else name
    core, rest = toks[0], [t for t in toks[1:] if t != "v"]

    m = re.fullmatch(r"(.*)CDM", core)
    if m and (m.group(1) in _MODEL_PREFIX or re.fullmatch(r"[A-Za-z]\d?", m.group(1) or "")):
        pre = _MODEL_PREFIX.get(m.group(1))
        if pre is None:
            pre = _symbol(m.group(1))
        label = (f"${pre}$CDM" if pre else "CDM")
        if rest:
            label += " (" + " ".join(tex_escape_text(t) for t in rest) + ")"
    else:
        label = " ".join(tex_escape_text(t) for t in [core] + rest)

    if _compiles(label, _usetex()):
        return label
    return tex_escape_text(name)


def text_label(text: str) -> str:
    """Plain text made safe for the active renderer (escaped when usetex is on)."""
    text = str(text)
    if "$" in text:
        return text if _compiles(text, _usetex()) else tex_escape_text(text.replace("$", ""))
    return tex_escape_text(text) if _usetex() else text
