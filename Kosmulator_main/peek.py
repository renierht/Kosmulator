"""Read-only emcee inspection from a private copy of a saved chain."""
import argparse
from contextlib import redirect_stdout
import sys
import json
import shutil
import tempfile
from pathlib import Path
import h5py
import numpy as np
from Kosmulator_main.convergence import split_rhat


def inspect_chain(path, burn, *, plot=False, plot_samples=10000):
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
    plot_path = None
    if plot:
        if plot_samples < 100:
            raise ValueError("plot_samples must be at least 100")
        with redirect_stdout(sys.stderr):
            plot_path = plot_corner(chain, names, path, end, plot_samples)
    quantiles = np.percentile(chain.reshape(-1, chain.shape[-1]), [16, 50, 84], axis=0)
    rhat = split_rhat(chain)
    return dict(chain=str(path.resolve()), corner_plot=plot_path, saved_steps=saved, inspected_end_exclusive=end,
                burn=burn, retained_steps=len(chain), walkers=chain.shape[1],
                parameters={name: dict(p16=float(q[0]), median=float(q[1]), p84=float(q[2]),
                                       split_rhat=float(r) if np.isfinite(r) else None)
                            for name, q, r in zip(names, quantiles.T, rhat)},
                note="Within-ensemble diagnostics; no stopping-policy or independent-ensemble claim")



def plot_corner(chain, names, source, end, limit):
    """Use GetDist and Kosmulator labels, with a deterministic display-only sample."""
    import matplotlib
    matplotlib.use("Agg")
    from getdist import MCSamples, plots
    from Plots.Plot_functions import greek_Symbols
    root = Path(__file__).resolve().parents[1] / "Plots" / "Saved_plots" / "Peek"
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f"{source.stem}_step{end}_", dir=root))
    flat = chain.reshape(-1, chain.shape[-1])
    rng = np.random.default_rng(173)
    selected = rng.choice(len(flat), size=min(limit, len(flat)), replace=False)
    samples = MCSamples(samples=flat[selected], names=names, labels=greek_Symbols(names))
    graph = plots.get_subplot_plotter()
    try:
        graph.triangle_plot([samples], filled=True)
        target = folder / "corner.png"
        graph.export(str(target))
    finally:
        import matplotlib.pyplot as plt
        plt.close(graph.fig)
    (folder / "snapshot.json").write_text(json.dumps(dict(
        source=str(source.resolve()), end_exclusive=end,
        retained_steps=len(chain), burn=end-len(chain), plotted_draws=len(selected), seed=173,
        note="Display subsample only; reported quantiles use all retained samples"
    ), indent=2))
    return str(target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--burn", type=int)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-samples", type=int, default=10000)
    args = parser.parse_args(argv)
    burn = args.burn
    if burn is None:
        manifests = [parent / "run_settings.json" for parent in args.chain.parents]
        manifest = next((p for p in manifests if p.is_file()), None)
        if manifest is None:
            parser.error("Legacy chain has no run_settings.json; provide its actual --burn")
        burn = int(json.loads(manifest.read_text())["burn"])
    print(json.dumps(inspect_chain(args.chain, burn, plot=args.plot, plot_samples=args.plot_samples), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
