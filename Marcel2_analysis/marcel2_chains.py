"""
Standalone loader for the Marcel_2 emcee chains.

Does NOT import Kosmulator and never writes to MCMC_Chains: HDF5 files are
opened read-only. It only turns the raw .h5 files into analysis-ready arrays.

Usage (from Python / notebook):
    from marcel2_chains import load_all, summary_table
    chains = load_all()
    print(summary_table(chains))
    c = chains["IDE_free_rd/Free/NonLinear_IDE_2"]
    samples = c.flat(burn=0.3, thin="auto")   # (N, ndim), burn-in removed, thinned
    c.names                                   # parameter names

Command line:
    python marcel2_chains.py            # prints the summary table
    python marcel2_chains.py --export   # also writes flat samples to ./flat_samples/*.npz
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "MCMC_Chains" / "Marcel_2"

# The .h5 files do not store parameter names. These are INFERRED from the column
# medians (Omega_m ~ 0.3, H_0 ~ 73, r_d ~ 137, M ~ -19.2, w ~ -1) and the
# Kosmulator model registry. Please confirm against the run configs.
# Keyed by (model-folder keyword, ndim).
NAMES_BY_NDIM = {
    ("LCDM_v", 4): ["Omega_m", "H_0", "r_d", "M"],
    ("LCDM_v", 3): ["Omega_m", "omega_b", "H_0"],           # BBN+CC+DESI+Pantheon (inferred)
    ("NonLinear_IDE_2", 6): ["Omega_m", "w", "delta", "H_0", "r_d", "M"],
    ("IDE_CLASS_Included", 5): ["Omega_m", "w", "delta", "H_0", "M"],
}
# BBN + PantheonP_SH0ES LCDM has ndim=4 but a different set (inferred, unverified).
BBN_SHOES_NAMES = ["Omega_m", "omega_b", "H_0", "M"]


@dataclass
class Chain:
    key: str                     # path relative to Marcel_2, without file name
    path: Path
    raw: np.ndarray              # (n_steps, n_walkers, ndim), truncated to filled steps
    log_prob: np.ndarray         # (n_steps, n_walkers)
    blobs: np.ndarray            # (n_steps, n_walkers); meaning unverified, see note below
    names: list[str]
    converged: bool | None       # 'converged' flag written by Kosmulator, None if absent
    accepted: np.ndarray = field(repr=False, default=None)

    @property
    def n_steps(self): return self.raw.shape[0]
    @property
    def n_walkers(self): return self.raw.shape[1]
    @property
    def ndim(self): return self.raw.shape[2]

    def tau(self):
        """Integrated autocorrelation time per parameter (emcee); NaN if chain too short."""
        import emcee
        try:
            return emcee.autocorr.integrated_time(self.raw, quiet=True)
        except Exception:
            return np.full(self.ndim, np.nan)

    def acceptance(self):
        """Mean acceptance fraction over the full run."""
        return float(np.mean(self.accepted / max(self.n_steps, 1))) if self.accepted is not None else np.nan

    def flat(self, burn=0.3, thin="auto", with_logprob=False):
        """Flattened samples. burn: fraction (<1) or integer number of steps.
        thin: integer, or 'auto' = half the largest autocorrelation time (min 1)."""
        nb = int(burn * self.n_steps) if burn < 1 else int(burn)
        if thin == "auto":
            t = np.nanmax(self.tau())
            thin = max(int(0.5 * t), 1) if np.isfinite(t) else 1
        s = self.raw[nb::thin].reshape(-1, self.ndim)
        if with_logprob:
            return s, self.log_prob[nb::thin].reshape(-1)
        return s


def _guess_names(key: str, ndim: int, path: Path):
    if "BBN" in path.name and ndim == 4:
        return BBN_SHOES_NAMES
    for (kw, nd), nm in NAMES_BY_NDIM.items():
        if nd == ndim and kw in key:
            return nm
    return [f"p{i}" for i in range(ndim)]


def load_chain(path: Path) -> Chain:
    with h5py.File(path, "r") as h:
        g = h["mcmc"]
        # Arrays are pre-allocated (e.g. 100000 rows); 'iteration' is the number actually filled.
        n = int(g.attrs["iteration"])
        raw = g["chain"][:n]
        lp = g["log_prob"][:n]
        bl = g["blobs"][:n] if "blobs" in g else np.full(lp.shape, np.nan)
        acc = g["accepted"][:]
        conv = h.attrs.get("converged", None)
    key = str(path.parent.relative_to(ROOT))
    # collapse the redundant data-combination folder for the key, keep it as a suffix
    names = _guess_names(key, raw.shape[2], path)
    return Chain(key=key, path=path, raw=raw, log_prob=lp, blobs=bl, names=names,
                 converged=None if conv is None else bool(conv), accepted=acc)


def load_all(root: Path = ROOT) -> dict[str, Chain]:
    out = {}
    for p in sorted(root.rglob("*.h5")):          # ignores the *.h5:Zone.Identifier files
        c = load_chain(p)
        out[c.key + "/" + p.stem] = c
    return out


def summary_table(chains: dict[str, Chain], burn=0.3):
    import pandas as pd
    rows = []
    for k, c in chains.items():
        tau = c.tau()
        tmax = np.nanmax(tau) if np.isfinite(tau).any() else np.nan
        rows.append({
            "chain": k, "steps": c.n_steps, "walkers": c.n_walkers, "ndim": c.ndim,
            "converged_flag": c.converged, "acc_frac": round(c.acceptance(), 3),
            "tau_max": round(float(tmax), 1) if np.isfinite(tmax) else np.nan,
            "steps/tau": round(c.n_steps / tmax, 1) if np.isfinite(tmax) else np.nan,
            "max_logprob": round(float(c.log_prob.max()), 2),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="write flat samples to ./flat_samples/")
    a = ap.parse_args()
    ch = load_all()
    import pandas as pd
    pd.set_option("display.width", 250, "display.max_colwidth", 90)
    print(summary_table(ch).to_string(index=False))
    if a.export:
        out = Path(__file__).parent / "flat_samples"
        out.mkdir(exist_ok=True)
        for k, c in ch.items():
            s, lp = c.flat(with_logprob=True)
            np.savez(out / (k.replace("/", "__") + ".npz"), samples=s, log_prob=lp, names=c.names)
        print("exported to", out)
