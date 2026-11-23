"""Record and validate paper-launcher settings under the run directory lock."""
import json
import os
from pathlib import Path

SCIENTIFIC_KEYS = (
    "likelihood_policy", "dataset_hashes", "distance_policy", "launcher", "runtime_constants", "cli_settings", "source_hashes", "true_values",
    "observations", "models", "priors", "class_settings", "regime_switches",
    "free_w", "free_w_dm", "free_w0", "free_wa", "coupled_prior", "fixed_parameters_by_model",
    "emcee_moves", "walkers", "burn", "thin", "convergence_policy", "convergence_setting",
)


def validate_or_record(directory, settings):
    """Call while holding the run lock, before importing or starting the sampler."""
    directory = Path(directory)
    path = directory / "run_settings.json"
    if path.exists():
        previous = json.loads(path.read_text())
        previous.setdefault("emcee_moves", "stretch")
        settings = dict(settings)
        settings.setdefault("emcee_moves", "stretch")
        changed = [key for key in SCIENTIFIC_KEYS
                   if previous.get(key) != settings.get(key)]
        if changed:
            raise ValueError("Saved run settings differ: " + ", ".join(changed)
                             + ". Use a fresh run name.")
        return
    if any(directory.rglob("*.h5")):
        raise ValueError(
            "Existing chains lack a complete run_settings.json. Continue them "
            "with their original launcher until their provenance is verified."
        )
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(settings, indent=2) + "\n")
    os.replace(temporary, path)


def generic_settings(namespace, constants, model_module, cli, root):
    """Conservative direct-launcher provenance; serialize before comparing."""
    import hashlib
    settings = dict(
        likelihood_policy="pantheon-zhd-zhel-v1", distance_policy="dense-grid-dz0.01-v1", launcher="generic", models=namespace["model_names"],
        observations=namespace["observations"], priors=namespace["prior_limits"],
        true_values=namespace["true_values"], walkers=namespace["nwalkers"],
        burn=namespace["burn"], convergence_setting=namespace["convergence"],
        emcee_moves=constants.EMCEE_MOVES,
        free_w=getattr(model_module, "FREE_W", None),
        free_w_dm=getattr(model_module, "FREE_W_DM", None),
        regime_switches=[getattr(model_module, key, None) for key in
                        ("ALLOW_NEGATIVE_ENERGIES", "ALLOW_BIG_RIP",
                         "ALLOW_DOOM_FACTOR_INSTABILITIES")],
    )
    settings["runtime_constants"] = {}
    for key, value in vars(constants).items():
        if key.isupper():
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError):
                continue
            settings["runtime_constants"][key] = value
    operational = {"num_cores", "use_mpi", "latex_enabled", "plot_table",
                   "output_suffix", "resume", "overwrite"}
    settings["cli_settings"] = {key: value for key, value in vars(cli).items()
                                if key not in operational}
    settings["dataset_hashes"] = {}
    if any("CC" in group for group in namespace["observations"]):
        base = Path(constants.OBSERVATIONS_BASE)
        for filename in ("CC.dat", "CC_corr.txt"):
            path = base / filename
            settings["dataset_hashes"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    settings["source_hashes"] = {}
    files = [Path(root) / "User_defined_modules.py"]
    files += sorted((Path(root) / "Kosmulator_main").glob("*.py"))
    files += sorted((Path(root) / "Class/IDE_background/source").glob("*.c"))
    files += sorted((Path(root) / "Class/IDE_background/include").glob("*.h"))
    for path in files:
        settings["source_hashes"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(json.dumps(settings, allow_nan=False))


def require_chain_distance_policy(chain_path, expected, likelihood_policy=None):
    """Reject likelihood-based reanalysis of chains from a different integrator."""
    chain_path = Path(chain_path).resolve()
    for directory in chain_path.parents:
        manifest = directory / "run_settings.json"
        if manifest.is_file():
            previous = json.loads(manifest.read_text())
            if (previous.get("distance_policy") == expected and
                    (likelihood_policy is None or previous.get("likelihood_policy") == likelihood_policy)):
                return
            break
    raise ValueError(
        f"Chain {chain_path} has no matching distance/likelihood policy. "
        "Analyse legacy chains with their original checkout; corrected-distance "
        "likelihood statistics require fresh chains."
    )


def paper_identity(root, observations):
    """Content identity for the paper launcher's supported data and source inputs."""
    import hashlib
    root = Path(root)
    datasets = {
        'CC': ['CC.dat', 'CC_corr.txt'],
        'PantheonPS': ['PantheonP.dat', 'PantheonP.cov'],
        'PantheonP': ['PantheonP.dat', 'PantheonP.cov'],
        'DESI_DR2': ['DESI_DR2_synced.txt', 'DESI_DR2_covtot.txt'],
        'DES_Dovekie': ['DES_Dovekie/DES-Dovekie_HD.csv', 'DES_Dovekie/STAT+SYS.npz'],
    }
    requested = {name for group in observations for name in group}
    unsupported = requested - datasets.keys()
    if unsupported:
        raise ValueError('Paper provenance has no dataset mapping for: ' + ', '.join(sorted(unsupported)))
    def digest(path):
        value = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024*1024), b''):
                value.update(block)
        return value.hexdigest()
    files = [root/'User_defined_modules.py', root/'Kosmulator.py',
             root/'reproducibility/SAIP2026_IDE/scripts/run_analysis.py']
    files += sorted((root/'Kosmulator_main').glob('*.py'))
    files += sorted((root/'Class/IDE_background/source').glob('*.c'))
    files += sorted((root/'Class/IDE_background/include').glob('*.h'))
    return dict(source_hashes={str(p.relative_to(root)):digest(p) for p in files},
                dataset_hashes={name:digest(root/'Observations'/name)
                                for name in sorted({path for obs in requested for path in datasets[obs]})})


def require_paper_chain_settings(chain_path, settings, model):
    """Analysis can change burn, but cannot silently change the sampled posterior."""
    keys = ('distance_policy', 'likelihood_policy', 'source_hashes', 'dataset_hashes',
            'observations', 'priors', 'class_settings', 'regime_switches',
            'free_w', 'free_w_dm', 'free_w0', 'free_wa', 'coupled_prior', 'fixed_parameters_by_model')
    for folder in Path(chain_path).resolve().parents:
        manifest = folder/'run_settings.json'
        if manifest.is_file():
            previous = json.loads(manifest.read_text())
            changed = [key for key in keys if previous.get(key) != settings.get(key)]
            if model not in previous.get('models', []):
                changed.append('models')
            if changed:
                raise ValueError('Chain analysis settings differ: '+', '.join(changed)+
                                 '. Use its original checkout/settings; do not relabel the manifest.')
            return
    raise ValueError('Chain has no run manifest; use its original analysis checkout.')
