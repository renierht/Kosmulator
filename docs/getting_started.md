# Getting Started

## Requirements

Kosmulator runs on **Linux**, **macOS** and **Windows through WSL2**. Some dependencies are always needed, others only for particular data sets. The conda environment (`environment.yml`) installs all of them.

**Core (every run)**

- Python 3.9 - 3.12 (3.11 recommended)
- NumPy, SciPy, Matplotlib, h5py, Pandas, tqdm, threadpoolctl
- GetDist (corner plots and marginal statistics)
- emcee and zeus (the MCMC samplers)
- Cython, setuptools and a C/C++ compiler with `make`, for the CLASS builds

**CMB data**

- CLASS (`classy`), built by Kosmulator itself, per model, from `Class/<model>`
- clipy (or the compiled Planck CLIK) for the Planck likelihoods, candl for ACT DR6 and SPT-3G, act_dr6_lenslike for the ACT DR6 lensing likelihood

**Optional**

- mpi4py and schwimmbad, for MPI runs
- AlterBBN, for live BBN predictions beyond the grid Kosmulator ships
- LaTeX, for publication-quality plot labels (`--latex_enabled`)

## Installation

The recommended route is a conda environment, which brings everything Kosmulator needs, including the C compiler it uses to build CLASS, so nothing needs `sudo`.

::::{admonition} Windows users: clone into your Linux home folder
:class: tip
On WSL, clone Kosmulator into your Linux home folder (`~/`), not into `/mnt/c` or `/mnt/d`. Reading and writing files on the Windows drives from WSL is much slower, and an MCMC run reads and writes a lot.
::::

:::::{tab-set}
::::{tab-item} conda (recommended)

Install [Miniforge](https://github.com/conda-forge/miniforge) if you do not have conda yet. On macOS the Xcode Command Line Tools are needed as well (`xcode-select --install`). Then:

```bash
git clone --depth 1 https://github.com/renierht/Kosmulator.git
cd Kosmulator
conda env create -f environment.yml
conda activate kosmulator
kosmulator-doctor --setup
```
::::
::::{tab-item} pip

Use a clean virtual environment with Python 3.9 - 3.12 (3.11 recommended) and a C/C++ compiler (Ubuntu or WSL: `sudo apt install build-essential python3-dev python3-venv`; macOS: `xcode-select --install`):

```bash
git clone --depth 1 https://github.com/renierht/Kosmulator.git
cd Kosmulator
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
kosmulator-doctor --setup
```

`pip install -e .` installs the core only, which is enough for the late-time data. The optional groups are:

| Extra | Adds | Needed for |
|---|---|---|
| `.[cmb]` | clipy, candl, act_dr6_lenslike | Planck, ACT DR6, SPT-3G and CMB lensing likelihoods |
| `.[mpi]` | mpi4py, schwimmbad | MPI runs (`--use_mpi`); needs an MPI library, e.g. `sudo apt install mpich` |
| `.[all]` | both of the above | everything |
::::
:::::

## Checking the installation

`kosmulator-doctor --setup` is run once after installing. It

- checks the Python packages and the compiler,
- builds CLASS for `LCDM_v` and `rd_Class` (a few minutes, once),
- loads every CMB likelihood code with its data,
- tests MPI and LaTeX, and
- makes a short test run.

Anything missing is listed with the command that adds it. Afterwards, `kosmulator-doctor` without `--setup` gives a quick check at any time.

::::{dropdown} All `kosmulator-doctor` options

| Option | What it does |
|---|---|
| `--setup` | Build CLASS for `LCDM_v` and `rd_Class`, load every CMB likelihood code and its data, test MPI and LaTeX, and make a short test run. |
| `--build-class MODEL` | Build `Class/MODEL` now (repeatable), for example a model with its own CLASS, so that compiler errors show before a run. |
| `--no-test-run` | With `--setup`: skip the test run. |
| `--keep-test` | With `--setup`: keep the test run's outputs. |
| `--verbose-build` | Show the compiler output instead of writing it to `Class/<model>/build.log`. |
::::

## What you do not have to install

You do not have to install **CLASS** or the **Planck likelihood code** yourself.

- Kosmulator builds its own CLASS for each model from `Class/<model>` the first time that model meets CMB data, and caches the build.
- The Planck, ACT DR6, SPT-3G and lensing data are in `Observations/`. The Planck likelihoods are read by [clipy](https://github.com/benabed/clipy), the pure-Python implementation of CLIK by K. Benabed and L. Balkenhol, and ACT and SPT by [candl](https://github.com/Lbalkenhol/candl). If the compiled CLIK is installed, Kosmulator uses it instead; the run log records which one was used.
- The Planck PR4 CamSpec data (`CMB_hil_PR4`, about 0.5 GB) are not in git. The first run that uses them downloads them, and says why and from where.

::::{dropdown} SPT-3G data missing or changed

If the SPT-3G D1 files are ever missing or changed in your copy, this downloads and checks them again:

```bash
python -m Kosmulator_main.get_spt_data
```

For the MUSE lensing file add `--muse-zip` with the zip from <https://pole.uchicago.edu/public/data/ge25/> (see `Observations/candl/README.md` and `Observations/spt3g_muse_lensing/README.md`).
::::

::::{dropdown} LaTeX for the plot labels (optional)

Kosmulator can use Matplotlib's LaTeX rendering for publication-quality labels (`--latex_enabled`). Check that LaTeX is available with `latex --version`. Without it, simply leave `--latex_enabled` out; the plots are then drawn with Matplotlib's own mathtext.

- **Windows**: install [MiKTeX](https://miktex.org/howto/install-miktex) and enable the automatic installation of missing packages.
- **macOS**: `brew install mactex`
- **Linux / WSL**: `sudo apt install texlive-latex-recommended texlive-fonts-recommended cm-super dvipng` (or `texlive-full`)

An error such as `RuntimeError: latex was not able to process the following string` usually means a LaTeX package (for example `type1ec.sty`) is missing.
::::

## Updating

From the Kosmulator folder:

```bash
git pull
conda env update -f environment.yml --prune   # conda installs
pip install -e ".[all]"                        # pip installs
kosmulator-doctor
```

CLASS is rebuilt automatically when its sources change.

## Optional extras

The compiled Planck CLIK (instead of clipy) and live AlterBBN calls are optional extras for advanced users; they are described in [Advanced installation](reference/advanced_installation.md).
