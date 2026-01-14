from __future__ import annotations

from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).resolve().parent


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default


def parse_requirements(req_path: Path) -> list[str]:
    """
    Parse a requirements.txt style file.
    - Ignores comments and blank lines
    - Skips editable installs and direct URLs (keep those in requirements.txt only)
    """
    if not req_path.exists():
        return []

    reqs: list[str] = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-e ", "git+", "http://", "https://")):
            # Keep these in requirements.txt; avoid breaking pip install .
            continue
        reqs.append(line)
    return reqs


# Basic metadata
PACKAGE_NAME = "kosmulator"
DESCRIPTION = "Kosmulator: cosmological inference framework with modular likelihoods and sampler backends."
LONG_DESCRIPTION = read_text(ROOT / "README.md", default=DESCRIPTION)

# If you want a dynamic version later, add __version__ to Kosmulator_main/__init__.py
# For now, keep it simple and explicit:
def read_version() -> str:
    ns = {}
    exec((ROOT / "Kosmulator_main" / "__init__.py").read_text(encoding="utf-8"), ns)
    return ns.get("__version__", "2.0.0")

VERSION = read_version()

install_requires = parse_requirements(ROOT / "requirements.txt")

# Find packages inside Kosmulator_main
packages = find_packages(where=".", include=["Kosmulator_main", "Kosmulator_main.*"])

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Renier Hough",
    # url="https://github.com/<your-user>/<your-repo>",  # optional
    packages=packages,
    include_package_data=True,  # respects MANIFEST.in if you add one later
    python_requires=">=3.9,<3.13",
    install_requires=install_requires,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",  # change if different
        "Operating System :: OS Independent",
    ],
    # Optional: only add this if you *actually* have a stable main() entry point.
    # entry_points={
    #     "console_scripts": [
    #         "kosmulator=Kosmulator_main.Kosmulator:main",
    #     ]
    # },
)

