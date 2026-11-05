"""Record and validate paper-launcher settings under the run directory lock."""
import json
import os
from pathlib import Path

SCIENTIFIC_KEYS = (
    "observations", "models", "priors", "class_settings", "regime_switches",
    "free_w", "free_w_dm", "free_w0", "free_wa", "coupled_prior",
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
