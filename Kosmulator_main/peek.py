"""Read-only emcee inspection from a private copy of a saved chain."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path
import h5py
import numpy as np
from Kosmulator_main.convergence import split_rhat


def inspect_chain(path, burn):
    """Copy before opening HDF5; omit the last potentially incomplete iteration."""
    path = Path(path)
    if burn < 0:
        raise ValueError("Burn must be nonnegative")
    with tempfile.TemporaryDirectory(prefix="kosmulator-peek-") as folder:
        copied = Path(folder) / "snapshot.h5"
        shutil.copyfile(path, copied)
        with h5py.File(copied, "r") as handle:
            group = handle["mcmc"]
            saved = int(group.attrs["iteration"])
            end = min(saved, group["chain"].shape[0]) - 1
            if end <= burn:
                raise ValueError("Insufficient saved steps after burn and final-step exclusion")
            chain = group["chain"][burn:end]
    metadata = Path(str(path) + ".metadata.json")
    names = json.loads(metadata.read_text()).get("parameters") if metadata.exists() else None
    if names is None:
        names = [f"parameter_{i}" for i in range(chain.shape[-1])]
    if len(names) != chain.shape[-1] or not np.isfinite(chain).all():
        raise ValueError("Parameter metadata or sampled values are invalid")
    quantiles = np.percentile(chain.reshape(-1, chain.shape[-1]), [16, 50, 84], axis=0)
    rhat = split_rhat(chain)
    return dict(chain=str(path.resolve()), saved_steps=saved, inspected_end_exclusive=end,
                burn=burn, retained_steps=len(chain), walkers=chain.shape[1],
                parameters={name: dict(p16=float(q[0]), median=float(q[1]), p84=float(q[2]),
                                       split_rhat=float(r) if np.isfinite(r) else None)
                            for name, q, r in zip(names, quantiles.T, rhat)},
                note="Within-ensemble diagnostics; no stopping-policy or independent-ensemble claim")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--burn", type=int)
    args = parser.parse_args(argv)
    burn = args.burn
    if burn is None:
        manifests = [parent / "run_settings.json" for parent in args.chain.parents]
        manifest = next((p for p in manifests if p.is_file()), None)
        if manifest is None:
            parser.error("Legacy chain has no run_settings.json; provide its actual --burn")
        burn = int(json.loads(manifest.read_text())["burn"])
    print(json.dumps(inspect_chain(args.chain, burn), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
