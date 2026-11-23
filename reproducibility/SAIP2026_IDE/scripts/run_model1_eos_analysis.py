#!/usr/bin/env python3
"""Generate independent SAIP IDE chains, or analyse the user's saved chains."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=("iw", "plus_iw", "siw"))
    parser.add_argument("--run-name", required=True, help="Unique name for this run")
    parser.add_argument("--mode", choices=("sample", "analyse"), default="sample")
    parser.add_argument("--chains", type=Path, help="JSON mapping model names to saved HDF5 paths")
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--derived-rd-samples", type=int, default=6000)
    parser.add_argument("--init-jitter-frac", type=float, default=0.10, help="Initial scatter as fraction of prior width; no change to sampling priors")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--emcee-moves", choices=("stretch", "de_snooker"), default="de_snooker")
    parser.add_argument("--num-cores", type=int, default=1)
    parser.add_argument("--ide-model", choices=("Linear_IDE_1", "Linear_IDE_2", "Linear_IDE_3", "Linear_IDE_4", "Linear_IDE_5", "NonLinear_IDE_2", "SignSwitch_IDE_1", "SignSwitch_IDE_2"), default="NonLinear_IDE_2")
    parser.add_argument("--only-model", choices=("lcdm", "ide", "cpl"), help="Run one model in a separate output suffix")
    parser.add_argument("--burn", type=int, default=1000, help="Fixed burn-in steps, as in the upstream convergence policy")
    parser.add_argument("--tau-consecutive", type=int, default=1)
    parser.add_argument("--dataset", choices=("paper", "dovekie"), default="paper")
    parser.add_argument("--include-cc", action="store_true", help="Add the local 36-point correlated CC compilation")
    parser.add_argument("--plan", action="store_true", help="Print settings without importing Kosmulator or running CLASS")
    parser.add_argument("--delta-prior", nargs=2, type=float, metavar=("LOW", "HIGH"))
    parser.add_argument("--omega-m-prior", nargs=2, type=float, metavar=("LOW", "HIGH"))
    parser.add_argument("--eos-case", choices=("cold_lambda", "cold_free_de", "warm_lambda"), default="cold_free_de")
    parser.add_argument("--uncoupled", action="store_true", help="Fix delta=0 in the selected linear IDE model; remove it from sampling")
    parser.add_argument("--w-dm-prior", nargs=2, type=float, metavar=("LOW", "HIGH"))
    parser.add_argument("--delta-dm-prior", nargs=2, type=float, required=True)
    parser.add_argument("--delta-de-prior", nargs=2, type=float, required=True)
    args = parser.parse_args()
    if args.ide_model != "Linear_IDE_1" or args.only_model != "ide" or args.uncoupled or args.delta_prior is not None:
        parser.error("This dedicated driver requires coupled Model 1 and independent delta-dm/de priors")
    free_w = args.eos_case == "cold_free_de"
    free_w_dm = args.eos_case == "warm_lambda"
    if (args.eos_case != "cold_free_de" or args.uncoupled or args.w_dm_prior is not None) and (args.only_model != "ide" or args.ide_model not in ("Linear_IDE_1", "Linear_IDE_2", "Linear_IDE_3", "Linear_IDE_4", "Linear_IDE_5")):
        parser.error("New EOS cases / uncoupled mode require --only-model ide and a supported linear IDE model")
    if free_w_dm != (args.w_dm_prior is not None):
        parser.error("Specify --w-dm-prior exactly when --eos-case warm_lambda is selected")
    if args.uncoupled and args.delta_prior is not None:
        parser.error("A fixed zero coupling cannot also have a sampling prior")
    if args.ide_model in ("Linear_IDE_1", "Linear_IDE_2", "Linear_IDE_3", "Linear_IDE_4", "Linear_IDE_5") and args.regime != "iw":
        parser.error("This linear IDE study uses --regime iw (background-only unrestricted regime)")

    if args.regime is None:
        if args.only_model != "cpl":
            parser.error("--regime is required for LCDM/IDE runs")
        args.regime = "iw"
    if not 0 < args.init_jitter_frac <= 1:
        parser.error("--init-jitter-frac must be in (0, 1]")
    if args.num_cores < 1:
        parser.error("--num-cores must be positive")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("--run-name must contain only letters, numbers, underscores or hyphens")
    if args.burn < 0 or args.max_steps <= args.burn or args.derived_rd_samples <= 0 or args.tau_consecutive < 1:
        parser.error("Require max-steps > burn >= 0, positive derived-rd-samples and tau-consecutive")
    if args.only_model != "cpl" and args.ide_model in ("SignSwitch_IDE_1", "SignSwitch_IDE_2") and args.regime == "siw":
        parser.error("Models 9–10 have no validated doom-factor stability classification")
    if args.mode == "analyse" and (args.chains is None or args.resume):
        parser.error("Analysis requires --chains and cannot resume sampling")
    if args.mode == "sample" and args.chains is not None:
        parser.error("--chains is only for --mode analyse")
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "reproducibility/SAIP2026_IDE/configs/paper.json").read_text())
    if args.dataset == "dovekie":
        config["observations"] = [["DES_Dovekie", "DESI_DR2"]]
        config["prior_limits"].pop("M_abs")
    if args.include_cc:
        config["observations"][0].append("CC")
    if args.only_model == "cpl":
        config["prior_limits"] = {k: v for k, v in config["prior_limits"].items() if k not in ("w", "delta")}
        config["prior_limits"].update(w0=[-3., 1.], wa=[-3., 2.])
    for key, bounds in (("delta_dm", args.delta_dm_prior), ("delta_de", args.delta_de_prior), ("delta", args.delta_prior), ("Omega_m", args.omega_m_prior), ("w_dm", args.w_dm_prior)):
        if bounds is not None:
            if not all(math.isfinite(v) for v in bounds) or bounds[0] >= bounds[1]:
                parser.error("Prior bounds must be finite and increasing")
            if key == "delta" and args.only_model != "ide":
                parser.error("--delta-prior requires --only-model ide")
            if key == "Omega_m" and not (0 < bounds[0] < bounds[1] < 1):
                parser.error("Require 0 < Omega_m lower < upper < 1")
            config["prior_limits"][key] = bounds
    if args.ide_model in ("Linear_IDE_1", "Linear_IDE_2", "Linear_IDE_3", "Linear_IDE_4", "Linear_IDE_5") and args.only_model == "ide":
        if not free_w:
            config["prior_limits"].pop("w", None)
        if args.uncoupled:
            config["prior_limits"].pop("delta", None)
    config["prior_limits"].pop("delta", None)
    fixed_parameters = {args.ide_model: {"delta": 0.0}} if args.uncoupled else None
    paths = None
    selected_models = (["w0waCDM"] if args.only_model == "cpl" else
                       ["LCDM_v"] if args.only_model == "lcdm" else
                       [args.ide_model] if args.only_model == "ide" else ["LCDM_v", args.ide_model])
    if args.chains:
        mapping = json.loads(args.chains.read_text())
        if set(mapping) != set(selected_models):
            parser.error("--chains must specify exactly the selected models")
        paths = {}
        for model, value in mapping.items():
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = args.chains.resolve().parent / path
            path = path.resolve()
            if not path.is_file():
                parser.error("Missing chain: " + str(path))
            paths[model] = str(path)
    suffix = args.run_name + ("_cpl" if args.only_model == "cpl" else "_" + args.regime)
    if args.dataset == "dovekie":
        suffix += "_dovekie"
    if args.include_cc:
        suffix += "_cc"
    settings = {
        "distance_policy": "dense-grid-dz0.01-v1",
        "likelihood_policy": "pantheon-zhd-zhel-v1",
        "emcee_moves": args.emcee_moves,
        "observations": config["observations"],
        "mode": args.mode, "regime": args.regime, "output_suffix": suffix,
        "models": selected_models, "free_w": free_w, "free_w_dm": free_w_dm,
        "walkers": config["walkers"], "burn": args.burn, "thin": config["thin"],
        "max_steps": args.max_steps, "convergence_setting": 0.05,
        "convergence_policy": {"tau_factor": 100, "ess_min": 2000, "rhat_max": 1.01, "consecutive": args.tau_consecutive},
        "init_jitter_frac": args.init_jitter_frac, "derived_rd_samples": args.derived_rd_samples,
        "class_settings": config["class_settings"], "priors": config["prior_limits"],
        "regime_switches": config["regime_switches"][args.regime],
        "chain_paths": paths, "dic_convention": "posterior-mean",
    }
    if fixed_parameters is not None:
        settings["fixed_parameters_by_model"] = fixed_parameters
    if args.only_model == "cpl":
        settings.pop("free_w")
        settings.update(free_w0=True, free_wa=True)
        settings["coupled_prior"] = "w0+wa<0"
        settings["regime"] = "not applicable"
        settings["regime_switches"] = None
    sys.path.insert(0, str(root))
    from Kosmulator_main.run_manifest import paper_identity
    settings.update(paper_identity(root, settings["observations"]))
    # Separate driver preserves historical launcher hashes and resume identities.
    import hashlib
    settings["source_hashes"][str(Path(__file__).resolve().relative_to(root))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(settings, indent=2), flush=True)
    if args.plan:
        return
    sys.path.insert(0, str(root))
    from Kosmulator_main.run_lock import run_lock
    if args.mode == "analyse":
        from Kosmulator_main.run_manifest import require_paper_chain_settings
        try:
            for model, chain_path in paths.items():
                require_paper_chain_settings(chain_path, settings, model)
        except ValueError as exc:
            parser.error(str(exc))
    with run_lock(root / "MCMC_Chains" / suffix):
        if args.mode == "sample":
            from Kosmulator_main.run_manifest import validate_or_record
            try:
                validate_or_record(root / "MCMC_Chains" / suffix, settings)
            except ValueError as exc:
                parser.error(str(exc))
        os.environ.setdefault("MPLBACKEND", "Agg")
        sys.path.insert(0, str(root))
        os.chdir(root)
        import Kosmulator as kc
        import User_defined_modules as udm
        from Kosmulator_main import constants as K
        K.EMCEE_MOVES = args.emcee_moves
        kc.model_names = selected_models
        udm.FREE_W, udm.FREE_W_DM = free_w, free_w_dm
        kc.true_model = config["reference_model"] if "LCDM_v" in selected_models else selected_models[0]
        kc.observations = config["observations"]
        kc.nwalkers, kc.burn = config["walkers"], args.burn
        kc.nsteps, kc.convergence = args.max_steps, 0.05
        kc.prior_limits = dict(kc.prior_limits, **{k: tuple(v) for k, v in config["prior_limits"].items()})
        for key, value in config["class_settings"].items():
            setattr(K, key, value)
        for key, value in zip(
            ("ALLOW_NEGATIVE_ENERGIES", "ALLOW_BIG_RIP", "ALLOW_DOOM_FACTOR_INSTABILITIES"),
            config["regime_switches"][args.regime],
        ):
            setattr(udm, key, value)
        sys.argv = [str(root / "Kosmulator.py"), "--force_emcee", "--num_cores", str(args.num_cores),
                    "--plot_table", "--output_suffix", suffix,
                    "--tau-consecutive", str(args.tau_consecutive)]
        if args.resume:
            sys.argv.append("--resume")
        options = {
            "postprocessing_options": dict(config["postprocessing"]),
            "plot_settings_overrides": {
                "init_jitter_frac": args.init_jitter_frac, "derived_rd_samples": args.derived_rd_samples,
                "derived_rd_seed": config["derived_rd_seed"],
            },
        }
        if fixed_parameters is not None:
            options["fixed_parameters_by_model"] = fixed_parameters
        if paths is not None:
            options["existing_chain_paths"] = paths
        kc.main(workflow_options=options)

if __name__ == "__main__":
    main()

