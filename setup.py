from __future__ import annotations

from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).resolve().parent


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default


def read_version() -> str:
    """
    Single-source version: Kosmulator_main/__init__.py defines __version__.
    """
    ns = {}
    init_path = ROOT / "Kosmulator_main" / "__init__.py"
    exec(init_path.read_text(encoding="utf-8"), ns)
    return ns.get("__version__", "0.0.0")


def parse_requirements(req_path: Path) -> list[str]:
    """
    Keep this intentionally lightweight.
    - Reads requirements.txt as-is (you control it).
    - Skips blanks and comments.
    """
    if not req_path.exists():
        return []
    reqs: list[str] = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs


# Optional groups: pip install -e ".[cmb]", ".[mpi]" or ".[all]"
EXTRAS = {
    # Planck through clipy (pure-Python clik), ACT DR6 / SPT-3G D1 through candl, ACT DR6 lensing
    "cmb": ["clipy-like>=0.15", "candl-like>=2.0.3", "act_dr6_lenslike"],
    # MPI runs; mpi4py needs an MPI library (conda: mpich; Ubuntu: sudo apt install mpich)
    "mpi": ["mpi4py", "schwimmbad"],
}
EXTRAS["all"] = sorted({req for reqs in EXTRAS.values() for req in reqs})


setup(
    name="kosmulator",
    version=read_version(),
    description="Kosmulator: cosmological inference framework with modular likelihoods and sampler backends.",
    long_description=read_text(ROOT / "README.md", default="Kosmulator: cosmological inference framework."),
    long_description_content_type="text/markdown",
    author="Renier Hough",
    # url="https://github.com/renierht/Kosmulator",
    packages=find_packages(include=["Kosmulator_main", "Kosmulator_main.*"]),
    include_package_data=False,
    zip_safe=False,
    python_requires=">=3.9,<3.13",
    install_requires=parse_requirements(ROOT / "requirements.txt"),
    extras_require=EXTRAS,
    entry_points={
        "console_scripts": [
            # Checks the installation; --setup also builds CLASS and makes a test run
            "kosmulator-doctor=Kosmulator_main:main_doctor",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ],
)

