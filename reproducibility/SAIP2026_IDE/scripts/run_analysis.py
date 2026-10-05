#!/usr/bin/env python3
"""Generate independent SAIP IDE chains, or analyse the user's saved chains."""
import argparse
import json
import os
from pathlib import Path
import re
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=("iw", "plus_iw", "siw"), required=True)
    parser.add_argument("--run-name", required=True, help="Unique name for this run")
    parser.add_argument("--mode", choices=("sample", "analyse"), default="sample")
    parser.add_argument("--chains", type=Path, help="JSON mapping model names to saved HDF5 paths")
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--derived-rd-samples", type=int, default=6000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", action="store_true", help="Print settings without importing Kosmulator or running CLASS")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("--run-name must contain only letters, numbers, underscores or hyphens")
    if args.max_steps <= 1000 or args.derived_rd_samples <= 0:
        parser.error("Require --max-steps > 1000 and --derived-rd-samples > 0")
    if args.mode == "analyse" and (args.chains is None or args.resume):
        parser.error("Analysis requires --chains and cannot resume sampling")
    if args.mode == "sample" and args.chains is not None:
        parser.error("--chains is only for --mode analyse")
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "reproducibility/SAIP2026_IDE/configs/paper.json").read_text())
    paths = None
    if args.chains:
        mapping = json.loads(args.chains.read_text())
        if set(mapping) != {"LCDM_v", "NonLinear_IDE_2"}:
            parser.error("--chains must specify LCDM_v and NonLinear_IDE_2")
        paths = {}
        for model, value in mapping.items():
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = args.chains.resolve().parent / path
            path = path.resolve()
            if not path.is_file():
                parser.error("Missing chain: " + str(path))
            paths[model] = str(path)
    suffix = args.run_name + "_" + args.regime
    settings = {
        "mode": args.mode, "regime": args.regime, "output_suffix": suffix,
        "walkers": config["walkers"], "burn": config["burn"], "thin": config["thin"],
        "max_steps": args.max_steps, "convergence_setting": 0.01,
        "derived_rd_samples": args.derived_rd_samples,
        "class_settings": config["class_settings"], "priors": config["prior_limits"],
        "regime_switches": config["regime_switches"][args.regime],
        "chain_paths": paths, "dic_convention": "posterior-mean",
    }
    print(json.dumps(settings, indent=2), flush=True)
    if args.plan:
        return
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(root))
    os.chdir(root)
    import Kosmulator as kc
    import User_defined_modules as udm
    from Kosmulator_main import constants as K
    kc.model_names = ["LCDM_v", "NonLinear_IDE_2"]
    kc.true_model = config["reference_model"]
    kc.observations = config["observations"]
    kc.nwalkers, kc.burn = config["walkers"], config["burn"]
    kc.nsteps, kc.convergence = args.max_steps, 0.01
    kc.prior_limits = dict(kc.prior_limits, **{k: tuple(v) for k, v in config["prior_limits"].items()})
    for key, value in config["class_settings"].items():
        setattr(K, key, value)
    for key, value in zip(
        ("ALLOW_NEGATIVE_ENERGIES", "ALLOW_BIG_RIP", "ALLOW_DOOM_FACTOR_INSTABILITIES"),
        config["regime_switches"][args.regime],
    ):
        setattr(udm, key, value)
    sys.argv = [str(root / "Kosmulator.py"), "--force_emcee", "--num_cores", "1",
                "--plot_table", "--output_suffix", suffix]
    if args.resume:
        sys.argv.append("--resume")
    options = {
        "postprocessing_options": dict(config["postprocessing"]),
        "plot_settings_overrides": {
            "derived_rd_samples": args.derived_rd_samples,
            "derived_rd_seed": config["derived_rd_seed"],
        },
    }
    if paths is not None:
        options["existing_chain_paths"] = paths
    kc.main(workflow_options=options)

if __name__ == "__main__":
    main()

