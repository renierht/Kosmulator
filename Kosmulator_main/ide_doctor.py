"""Read-only checks for the IDE background workflow and shared CLASS cache."""
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from Kosmulator_main import check_installation


def check_ide_environment(root=None):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    report = check_installation(check_class=False, check_clik=True,
                                check_clik_env=False, check_planck_assets=False)
    report["scope"] = "IDE background environment; no likelihood or physics validation"
    report["conda_prefix"] = os.environ.get("CONDA_PREFIX")
    report["platform"] = platform.platform()
    report["threads"] = {key: os.environ.get(key) for key in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    report["mpi_launcher"] = shutil.which("mpiexec") or shutil.which("mpirun")
    # CLIK is currently imported by Class_run even for background workflows.
    report["clik_note"] = "Current CLASS adapter imports CLIK; Planck likelihood data are optional here."
    original = Path.cwd()
    try:
        os.chdir(root)
        from Kosmulator_main import Class_run as CR
        models = sorted(CR.IDE_CLASS_MODELS)
        mapping = {name: CR.class_backend_name(name) for name in models}
        report["backend_mapping"] = mapping
        backend = "IDE_background"
        model_dir = os.path.join("./Class", backend)
        if not Path(model_dir).is_dir():
            raise FileNotFoundError(model_dir)
        signature = CR._sig_hash(CR._full_signature(model_dir))
        artifact = CR._find_cached_so(backend, signature)
        report["class_signature"] = signature
        report["class_artifact"] = str(Path(artifact).resolve()) if artifact else None
        ok = all(value == backend for value in mapping.values()) and artifact is not None
        report["items"].append(dict(name="Shared IDE CLASS routing and matching cache",
                                    ok=ok, detail=report["class_artifact"] or "No matching cached extension",
                                    hint="Build the shared IDE backend explicitly with the normal CLASS loader."))
        report["ok"] = report["ok"] and ok
    except Exception as exc:
        report["items"].append(dict(name="IDE CLASS cache check", ok=False,
                                    detail=f"{type(exc).__name__}: {exc}",
                                    hint="Activate the normal Kosmulator_test environment and CLIK profile."))
        report["ok"] = False
    finally:
        os.chdir(original)
    return report


def main():
    report = check_ide_environment()
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
