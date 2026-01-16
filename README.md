# Kosmulator: A Python framework for cosmological inference with MCMC

Kosmulator is a Python package utilising **Zeus** and **EMCEE (The MCMC Hammer)** to perform efficient, vectorised Markov Chain Monte Carlo (MCMC) simulations for studying **modified gravity** and **alternative cosmological models**.

The package is designed to be **modular**, **flexible**, and **user-friendly**, allowing researchers to easily configure inference runs, combine multiple sets of cosmological observations, and produce high-quality statistical summaries and visualisations.

---

## Features

- **Flexible Cosmological Inference Framework**  
  Run Bayesian inference pipelines for ΛCDM, modified gravity, and alternative cosmologies using a modular, model-agnostic architecture.

- **Multiple MCMC Backends**  
  Supports both **Zeus** and **EMCEE**, with automatic or user-controlled sampler selection and convergence diagnostics.

- **Vectorised Likelihood Evaluation**  
  Efficient vectorised likelihood computations for fast sampling across large parameter spaces and combined datasets.

- **Wide Range of Observational Data**  
  Built-in support for:
  - Type Ia Supernovae (JLA, Pantheon, Pantheon+, Union3, DES-Y5)
  - Baryon Acoustic Oscillations (BAO)
  - DESI (DR2)
  - Cosmic Chronometers / OHD
  - Growth of structure (fσ₈)
  - Cosmic Microwave Background (Planck 2015 & 2018 via CLIK)
  - Big Bang Nucleosynthesis (D/H, including AlterBBN-based likelihoods)

- **Automatic Parameter Injection & Dataset Awareness**  
  Parameters (including nuisance parameters) are automatically added or fixed based on selected observational datasets.

- **Sound Horizon (r_d) Policy Management**  
  Centralised handling of the sound horizon with support for fixed, free, CLASS-derived, or BBN-calibrated treatments depending on data combinations.

- **CLASS Integration with Caching**  
  Seamless integration with **CLASS**, including model-specific binary caching for fast reuse across runs and multiprocessing environments.

- **Publication-Ready Output & Visualisation**  
  Generates corner plots, best-fit theory comparisons, autocorrelation diagnostics, and LaTeX-ready statistical tables with a consistent directory structure.

- **Chain Reuse and Resume Capability**  
  Reload, resume, or extend existing MCMC chains for reproducibility and efficient experimentation.

- **Environment Diagnostics**  
  Includes a built-in `kosmulator-doctor` command to verify Python dependencies, CLASS, Planck CLIK likelihoods, and optional AlterBBN support.

---

## Requirements

Kosmulator is written in Python and supports a modular backend system. Some dependencies are always required, 
while others are required only when specific observational datasets are used.

#### Core Python Dependencies (required for all runs)

- **Python ≥ 3.9**
- **NumPy**
- **SciPy**
- **Matplotlib**
- **h5py**
- **Pandas**
- **GetDist** (corner plots and statistical visualisation)
- **EMCEE** (MCMC sampler)
- **Zeus** (vectorised MCMC sampler)

#### CMB-Specific Dependencies (required only when CMB likelihoods are used)

- **CLASS (classy)**  
  Used to compute background and perturbation quantities and CMB power spectra.
- **Planck CLIK likelihoods**  
  Required for Planck CMB likelihoods (TT, TTTEEE, low-ℓ, lensing).  
  The corresponding `.clik` likelihood directories must be available locally.
- **Astropy** 
- **Cython** 

#### Optional Dependencies

- **AlterBBN**  
  Optional backend for Big Bang Nucleosynthesis (BBN) D/H likelihoods.  
  Kosmulator supports approximate BBN likelihoods without AlterBBN, as well as physically motivated predictions using live AlterBBN calls or precomputed grids.

- **LaTeX (TeX Live / MiKTeX)**  
  Strongly recommended for publication-quality plots and LaTeX-ready tables. See Latex installation section additional information

---

## Installation
Clone the repository and install Kosmulator in a clean Python environment (recommended):
```bash
git clone https://github.com/renierht/Kosmulator.git
cd Kosmulator
pip install -e .
```
To verify your installation and check optional backends (CLASS, Planck CLIK, AlterBBN), run:
```bash
kosmulator-doctor
```
Note: For a full "Kitchen Sink" installation including CLASS and Planck CLIK, please see the Advanced Setup section.

---


## Quick Test
Run Kosmulator in your terminal.
``` bash
python Kosmulator.py 
```
If it ran successfully, it has been installed correctly!

## Configure Kosmulator Guide

#### Step 1: Define a New Model
Modify User_defined_modules.py to register your model parameters, and background expansion.
	- Implement E(z) for a new background model (flat or not).
	- (Optionally) define additional sanity restrictions for your model's free parameters.
	- Register the model name + parameter list in the model registry.
	- (Optional) expose CMB Cl wrappers for that model. Needed to fit to CMB observation

#### Step 2: Select Datasets 
In `Kosmulator.py`, select observation datasets. Observations are specified as lists of likelihood groups.
```python
# Run 1: JLA only | Run 2: OHD only | Run 3: Joint CC, OHD, and Pantheon
observations = [ ["JLA"], ["OHD"], ["CC", "OHD", "Pantheon"] ]
```

#### Step 3: Select model and configure MCMC Run
In `Kosmulator.py`, configure the MCMC sampler and specify the model names you want to analyse:

Specify the model names you want to analyse:
```python
# Models implemented in User_defined_modules.py
model_names: List[str] = ["Your_model_name"] 

true_model: str = "LCDM_v"  # Against which model you want to test it

# Sampler settings
nwalkers: int = 16
nsteps: int = 500
burn: int = 10
convergence: float = 0.01  # How accurate do you want the auto-correlator to be before stopping the run
```

#### Step 4. Execute you MCMC simulation
Run the script in the terminal:
```bash
python Kosmulator.py
```

---

## Project Structure

```plaintext
Kosmulator/
├── Kosmulator.py                # Main user-facing entry point (run inference)
├── User_defined_modules.py      # User-defined cosmological and gravity models
│
├── Kosmulator_main/             # Core inference engine
│   ├── __init__.py              # Package initialisation and versioning
│   ├── constants.py             # Global constants and runtime policies
│   ├── Config.py                # Dataset loading and parameter injection
│   ├── MCMC_setup.py            # High-level MCMC orchestration
│   ├── Kosmulator_MCMC.py       # Sampler execution (EMCEE / Zeus)
│   ├── Statistical_packages.py  # Likelihoods and statistical backends
│   ├── Class_run.py             # CLASS integration and caching
│   ├── rd_helpers.py            # Sound-horizon (r_d) handling
│   ├── Post_processing.py       # Post-processing and statistical summaries
│   └── utils.py                 # Shared utilities and CLI helpers
│
├── Class/                       # Local CLASS builds (per-model)
│   ├── LCDM_v/                  # CLASS source and build for ΛCDM
│   └── f1CDM_v/                 # CLASS source and build for modified gravity
│
├── AlterBBN_files/              # AlterBBN wrapper and interface code
│   ├── kosmo_bbn.c              # C interface for AlterBBN
│   └── alterbbn_ctypes.py       # Python ctypes wrapper
│
├── Observations/                # Observational datasets and likelihood files
│   ├── *.dat / *.txt            # Late-time cosmology data
│   ├── *.clik                  # Planck CMB likelihood directories
│   └── BBN/                     # Precomputed BBN grids
│
├── Plots/                       # Plotting and visualisation
│   ├── Plots.py                 # Plot orchestration (corner, best-fit, etc.)
│   ├── Plot_functions.py        # Plotting helper functions
│   └── Saved_plots/             # Generated plots
│
├── MCMC_Chains/                 # Stored MCMC chains
├── Statistical_analysis_tables/ # Statistical summaries and LaTeX-ready tables
│
├── setup.py                     # Packaging and installation
├── pyproject.toml               # Modern Python build configuration
├── LICENSE                      # Project license
└── README.md                    # Project documentation
```
		
## Command-Line Arguments to personalise your MCMC run
Kosmulator exposes a small set of CLI flags to control parallelism, sampler behaviour, diagnostics, and plotting.
You can view them any time with:

```bash
python Kosmulator.py --help
```

#### General/ Output

| Argument | Type | Default | Description |
|---|---|---|---|
| `--output_suffix` | str | `Test_run` | Suffix for output directories and files (chains, plots, tables). |
| `--overwrite` | flag | `False` | Delete any existing `.h5` chains and run MCMC from scratch. |
| `--resume` | flag | `False` | Resume incomplete chains (instead of only loading existing results). |
| `--init-log` | choice | `terse` | Initialisation logging style: `terse`, `normal`, or `verbose`. |

#### Parallelism

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num_cores` | int | `8` | Number of CPU cores to use for multiprocessing. |
| `--use_mpi` | flag | `False` | Force use of an MPI pool (if MPI is available). |

#### Sampler / Engine Control

| Argument | Type | Default | Description |
|---|---|---|---|
| `--engine-mode` | choice | `mixed` | Sampler strategy: `mixed`, `single`, or `fastest`. |
| `--force_zeus` | flag | `False` | Force the Zeus sampler. |
| `--force_emcee` | flag | `False` | Force the emcee sampler (ignore Zeus even if available). |
| `--force_vectorisation` | flag | `False` | Treat all models as vectorised (overrides detection). |
| `--disable_vectorisation` | flag | `False` | Disable vectorised likelihood evaluation even if available (forces scalar evaluation). |

#### Convergence / Autocorrelation  
*(mainly affects Zeus early-stop behaviour)*

| Argument | Type | Default | Description |
|---|---|---|---|
| `--tau-consecutive` (alias: `--consecutive-required`) | int | `3` | For Zeus early-stop: require this many consecutive callback checks with \|Δτ\|/τ < target. |
| `--autocorr-check-every` | int | `100` | Check autocorrelation every `N` iterations. |
| `--autocorr-buffer` | int | `None` | Extra iterations after burn-in before convergence checks start. If not set, Kosmulator uses `max(1000, burn/5)` as a default buffer. |

#### Plotting / Presentation

| Argument | Type | Default | Description |
|---|---|---|---|
| `--latex_enabled` | flag | `False` | Enable LaTeX rendering in plots. |
| `--plot_table` | flag | `False` | Generate parameter-table plots. |
| `--corner-show-all-cmb-params` | flag | `False` | Corner plot: show all CMB parameters (including nuisance). Default behaviour shows only key cosmological parameters. |
| `--corner-table-full` | flag | `False` | Corner plot top table: keep the full parameter list (including CMB nuisances). |

#### Likelihood Debugging

| Argument | Type | Default | Description |
|---|---|---|---|
| `--print_loglike [N]` | int (optional) | disabled | Print likelihood diagnostics (components + TOTAL) for one walker. If passed without `N`, defaults to `1` (prints every call). If `N` is provided, prints every `N`th likelihood call. |

---

## Multiprocessing run examples
With MPI (recommended on clusters)
```bash
mpiexec -n <num_cores> python Kosmulator.py --use_mpi --output_suffix "Your_Project_Name" --latex_enabled --overwrite --plot_table
```
With Python multiprocessing (local / workstation)
```bash
python Kosmulator.py --num_cores <num_cores> --output_suffix "Your_Project_Name" --latex_enabled --overwrite --plot_table
```
Using nohup (run in background after closing terminal)
```bash
nohup mpiexec -n <num_cores> python Kosmulator.py --use_mpi --output_suffix "Your_Project_Name" --latex_enabled --overwrite --plot_table > kosmulator_run.log 2>&1 &
```

---
## References

### Original MCMC code which developed into Kosmulator

1. **Original Kosmulator implementation**  
   Hough, R. T., Abebe, A., & Ferreira, S. E. S. (2020).  
   *Viability tests of f(R)-gravity models with Supernovae Type Ia data*.  
   European Physical Journal C, 80(8), 787.  
   https://doi.org/10.1140/epjc/s10052-020-8342-7

### MCMC Samplers

1. **EMCEE**  
   Foreman-Mackey, D., Hogg, D. W., Lang, D., et al. (2013).  
   *emcee: The MCMC Hammer*.  
   PASP, 125(925), 306.  
   https://doi.org/10.1086/670067

2. **Zeus**  
   Karamanis, M., Beutler, F., Peacock, J. A., et al. (2021).  
   *zeus: A Python implementation of ensemble slice sampling for efficient Bayesian parameter inference*.  
   MNRAS, 508(3), 3549–3561.  
   https://doi.org/10.1093/mnras/stab2485

---

### Type Ia Supernovae
#### JLA
1. Hicken, M., Challis, P., Jha, S., et al. (2009).  
   *CfA3: 185 Type Ia Supernova Light Curves from the CfA*.  
   ApJ, 700, 331.  
   https://doi.org/10.1088/0004-637X/700/1/331

2. Neill, J. D., Sullivan, M., Howell, D. A., et al. (2009).  
   *The Local Hosts of Type Ia Supernovae*.  
   ApJ, 707, 1449.  
   https://doi.org/10.1088/0004-637X/707/2/1449

3. Conley, A., Guy, J., Sullivan, M., et al. (2010).  
   *Supernova Constraints and Systematic Uncertainties from the First Three Years of the SNLS*.  
   ApJS, 192, 1.  
   https://doi.org/10.1088/0067-0049/192/1/1

#### Pantheon
- **TODO:** Add Pantheon (Scolnic et al. 2018) reference

#### Pantheon+
1. Brout, D., Scolnic, D., Popovic, B., et al. (2022).  
   *The Pantheon+ Analysis: Cosmological Constraints*.  
   ApJ, 938(2), 110.  
   https://doi.org/10.3847/1538-4357/ac8e04

#### Other SN Compilations
- **TODO:** Union3
- **TODO:** DES-Y5 Supernova sample

---

### Expansion Rate Measurements
#### Cosmic Chronometers (CC)
1. Moresco, M., Jimenez, R., Verde, L., et al. (2020).  
   *Setting the Stage for Cosmic Chronometers II*.  
   ApJ, 898(1), 82.  
   https://doi.org/10.3847/1538-4357/ab9eb0
   
2. Loubser, S. I., Alabi, A. B., Hilton, M., Ma, Y.-Z., Tang, X., Hatamkhani, N.,  
   Cress, C., Skelton, R. E., & Nkosi, S. A. (2025).  
   *An independent estimate of H(z) at z = 0.5 from the stellar ages of brightest cluster galaxies*.  
   Monthly Notices of the Royal Astronomical Society, **540**(4), 3135–3149.  
   https://doi.org/10.1093/mnras/staf915
   
3. Loubser, S. I. (2025).  
   *Measuring the expansion history of the Universe with DESI cosmic chronometers*.  
   Monthly Notices of the Royal Astronomical Society, **544**(4), 3064–3075.  
   https://doi.org/10.1093/mnras/staf1939
   
4. Wang, Z.-F., Lei, L., & Fan, Y.-Z. (2026).  
   *New H(z) measurement at redshift z = 0.12 with DESI Data Release 1*.  
   arXiv:2601.07345.  
   https://doi.org/10.48550/arXiv.2601.07345

#### Observational Hubble Data (OHD)
- **TODO:** Add canonical OHD compilation reference(s)

---

### Large-Scale Structure
#### Baryon Acoustic Oscillations (BAO) / DESI
1. Adame, A. G., Aguilar, J., Ahlen, S., et al. (2024).  
   *DESI 2024 VI: Cosmological Constraints from Baryon Acoustic Oscillations*.  
   arXiv:2404.03002  
   https://arxiv.org/abs/2404.03002

- **TODO:** Add pre-DESI BAO compilation references if desired (e.g. 6dF, SDSS, BOSS)

#### Growth of Structure (fσ₈)
1. Skara, F., & Perivolaropoulos, L. (2020).  
   *Tension of the EG statistic and RSD data with Planck ΛCDM*.  
   Phys. Rev. D, 101(6), 063521.

#### σ₈ Constraints
1. Perenon, L., Bel, J., Maartens, R., et al. (2019).  
   *Optimising Growth of Structure Constraints on Modified Gravity*.  
   JCAP, 2019(06), 020.  
   https://doi.org/10.1088/1475-7516/2019/06/020

---

### Early-time observations
#### Cosmic Microwave Background (CMB — Planck)

Kosmulator uses the standard **Planck likelihood datasets** distributed by the
**European Space Agency (ESA)** and accessed via the Planck **CLIK** likelihood
library.

1. Planck Collaboration (2018).  
   *Planck 2018 results. VI. Cosmological parameters*.  
   Astronomy & Astrophysics, **641**, A6.  
   https://doi.org/10.1051/0004-6361/201833910

#### Big Bang Nucleosynthesis (BBN)
1. Cooke, R. J., Pettini, M., Jorgenson, R. A., Murphy, M. T., & Steidel, C. C. (2014).  
   *Precision Measures of the Primordial Abundance of Deuterium*.  
   The Astrophysical Journal, **781**(1), 31.  
   https://doi.org/10.1088/0004-637X/781/1/31

---

## Citation
Kosmulator is an actively developed research framework.
At present, there is no dedicated software paper describing the current version of Kosmulator.

If you use Kosmulator in your work, please cite the original paper:

Hough, R. T., Abebe, A., & Ferreira, S. E. S. (2020). EPJC, 80(8), 787.
https://doi.org/10.1140/epjc/s10052-020-8342-7

A dedicated Kosmulator software reference (conference proceeding) is in preparation and will be added here once publicly available.

## Contributions
Contributions are welcome.

If you would like to contribute, please fork the repository and submit a pull request.
Bug fixes, documentation improvements, new observational datasets, and extensions to cosmological or modified-gravity models are encouraged.

If you prefer, you may also contact the author directly with tested code or proposed improvements for inclusion in the main repository.

## Contact
For questions or feedback, please contact:

- Renier Hough - [25026097@mynwu.ac.za] 

- Robert Rugg - [31770312@mynwu.ac.za]

---


## Installation process example
### LaTeX Dependencies for Plot Rendering

Kosmulator uses **Matplotlib’s LaTeX rendering** to generate publication-quality plots and tables.  
To enable this functionality, a working LaTeX installation is required.

Check that LaTeX is available on your system by running:
```bash
latex --version
```

If LaTeX is not installed (or if required packages are missing), you may encounter errors such as:
RuntimeError: latex was not able to process the following string ... (your system is missing some required LaTeX packages, e.g. type1ec.sty)

Follow os installation or update below:
- **Windows**: [MiKTeX Installation Guide](https://miktex.org/howto/install-miktex, ensure that you enable the option for automatic installation of missing packages)
- **macOS**: Install MacTeX via Homebrew:
  ```bash
  brew install mactex
  ```
- **Linux**: Install TeX Live:
  ```bash
  sudo apt install texlive-full
  sudo apt install texlive-latex-recommended texlive-fonts-recommended (if missing packages are required)
  ```
LaTeX rendering is optional. If you prefer not to install a full LaTeX distribution, simply leave LaTeX disabled (default), i.e. do not pass --latex_enabled

---

### Advanced Kosmulator Installation: Kosmulator, CLIK, CLASS, and AlterBBN (Full Setup)

This section provides a **complete, reproducible installation example** for running
Kosmulator with **CLASS**, **Planck CMB likelihoods (CLIK / plc_3.1)**, and **optional AlterBBN** support.

This is a *kitchen-sink* setup intended for advanced users who require:
- CLASS with Python bindings (`classy`)
- Planck likelihoods via CLIK
- Full CMB analyses within Kosmulator
- Optional high-accuracy BBN predictions via AlterBBN

> **Platform assumed:** Ubuntu / WSL  
> Paths shown use `/mnt/d/`; adjust paths as needed for your system.  
> macOS installation should be similar, with only different gcc and gfortran compilers.

---

#### Assumptions

You want:
1. CLASS built and importable via Python (`import classy`)
2. Planck CLIK likelihoods working (`import clik`)
3. Kosmulator linked to local `.clik` likelihood directories
4. *(Optional)* AlterBBN support for the `BBN_DH_AlterBBN` likelihood

---

#### 0) Create a Clean Workspace

```bash
mkdir -p /mnt/d/Kosmulator_test
cd /mnt/d/Kosmulator_test
```

#### 1) Clone Required Repositories
##### 1.1 Clone Kosmulator, CLASS, and AlterBBN (optional)
```bash
git clone https://github.com/renierht/Kosmulator.git
git clone https://github.com/lesgourg/class_public.git CLASS
git clone https://github.com/espensem/AlterBBN.git
```

##### 1.2 Download Planck Likelihood Code and Data
```bash
mkdir -p /mnt/d/Kosmulator_test/Clik
cd /mnt/d/Kosmulator_test/Clik

wget -O COM_Likelihood_Code-v3.0_R3.10.tar.gz \
  "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Code-v3.0_R3.10.tar.gz"

wget -O COM_Likelihood_Data-baseline_R3.00.tar.gz \
  "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Data-baseline_R3.00.tar.gz"

tar -xzf COM_Likelihood_Data-baseline_R3.00.tar.gz
tar -xzf COM_Likelihood_Code-v3.0_R3.10.tar.gz
```

#### 2) Create and Populate a Conda Environment
```bash
cd /mnt/d/Kosmulator_test

conda create -n Kosmulator_test python=3.11 -y  # We recommned using Python 3.11!
conda activate Kosmulator_test

conda install -c conda-forge -y \
  numpy scipy matplotlib h5py pandas \
  emcee zeus-mcmc cython astropy

pip install getdist
```

#### 3) Install System Build Dependencies (Ubuntu / WSL) - #MacOS will most likely differ here!
```bash
sudo apt update
sudo apt install -y \
  build-essential gfortran python3-dev \
  libcfitsio-dev pkg-config cmake
  
sudo apt install -y ripgrep (optional)
```

#### 4) Build and Install CLASS
```bash
cd /mnt/d/Kosmulator_test/CLASS
make clean
make -j
python -m pip install .
```

##### 4.1 Quick CLASS Sanity Test
```bash
python - <<'PY'
from classy import Class
cosmo = Class()
cosmo.set({
    "h": 0.67,
    "omega_b": 0.02237,
    "omega_cdm": 0.12,
    "A_s": 2.1e-9,
    "n_s": 0.965,
    "tau_reio": 0.054,
    "output": "tCl,pCl,lCl,mPk",
    "l_max_scalars": 2000,
})
cosmo.compute()
print("OK: age =", cosmo.age(), "sigma8 =", cosmo.sigma8())
cosmo.struct_cleanup()
cosmo.empty()
PY
```

#### 5) Build and Install Planck CLIK
NOTE: this is the plc_3.1 tree produced by the Planck Likelihood Code tarball.

```bash
cd /mnt/d/Kosmulator_test/Clik/code/plc_3.0/plc-3.1
python waf configure --install_all_deps
python waf install
```

##### 5.1 Persist CLIK Environment Variables
This ensures every time you use `conda activate Kosmulator_test`, CLIKROOT and the necessary PYTHONPATH/LD_LIBRARY_PATH are set correctly. 
May differ based on your OS distribution
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

cat > $CONDA_PREFIX/etc/conda/activate.d/clik.sh <<'SH'
export CLIKROOT=/mnt/d/Kosmulator_test/Clik/code/plc_3.0/plc-3.1
source "$CLIKROOT/bin/clik_profile.sh"

python - <<'PY'
import importlib.machinery as m
import os, pathlib, sys

suf = m.EXTENSION_SUFFIXES[0]

# Candidate roots where "clik/" might live
candidates = []

# 1) Directly from CLIKROOT (most reliable in your layout)
clikroot = os.environ.get("CLIKROOT")
if clikroot:
    candidates.append(pathlib.Path(clikroot) / "lib/python/site-packages")

# 2) Any PYTHONPATH entries added by clik_profile.sh
pp = os.environ.get("PYTHONPATH", "")
for entry in pp.split(":"):
    if entry.strip():
        candidates.append(pathlib.Path(entry.strip()))

# 3) As a last resort: whatever is already on sys.path
for entry in sys.path:
    if entry:
        candidates.append(pathlib.Path(entry))

# Find the first directory that actually contains clik/
pkg = None
for base in candidates:
    p = base / "clik"
    if p.exists() and p.is_dir():
        pkg = p
        break

# Don't break activation if we can't find it (just exit quietly)
if pkg is None:
    raise SystemExit(0)

for base in ["lkl", "lkl_lensing"]:
    src = pkg / base
    dst = pkg / f"{base}{suf}"
    if src.exists():
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src.name, dst)
        except FileExistsError:
            pass
PY
SH
```

##### 5.2 Reactivate and Quick CLIK test:
This is just to check if the CLIK installation worked and can persist after switching conda environment off.
```bash
conda deactivate
conda activate Kosmulator_test

python - <<'PY'
import clik
import clik.lkl, clik.lkl_lensing
print("OK: CLIK imported successfully")
PY
```

#### 6) Optional: AlterBBN Installation
Note:
  AlterBBN is required ONLY for the BBN_DH_AlterBBN likelihood.
  Other BBN-related options (e.g. BBN_DH approx, BBN_PryMordial) do NOT require AlterBBN.
  Kosmulator does NOT use AlterBBN's standalone executable (primary.x).


```bash
cd /mnt/d/Kosmulator_test/

cp Kosmulator/AlterBBN_files/kosmo_bbn.c AlterBBN/
cp Kosmulator/AlterBBN_files/alterbbn_ctypes.py AlterBBN/

cd /mnt/d/Kosmulator_test/AlterBBN
make

file kosmo_bbn.c | grep -qi "C source" || { echo "ERROR: kosmo_bbn.c is not C source"; exit 1; }

mkdir -p build
gcc -O3 -fPIC -shared -o build/libkosmo_bbn.so kosmo_bbn.c -Isrc -Lsrc -lbbn -lm

export KOSMO_BBN_LIB="/mnt/d/Kosmulator_test/AlterBBN/build/libkosmo_bbn.so"
```

##### 6.1 Quick AlterBBN Sanity Test:
```bash
python - <<'PY'
from alterbbn_ctypes import run_bbn
print("D/H =", run_bbn(0.022, 3.046, 879.4)['D_H'])
PY
```

##### 6.2 Allow AlterBBN to persist through deactivation
```bash
echo 'export KOSMO_BBN_LIB="/mnt/d/Kosmulator_test/AlterBBN/build/libkosmo_bbn.so"' >> ~/.bashrc
source ~/.bashrc
```
P.S. Remember to reactivate environment

##### 6.3 Optional: AlterBBN standalone executable
```bash
make primary.c 
./primary.x
```


#### 7) Final Sanity Checks. Use if installation fail. AI can help to identify the issues with the printouts
##### 7.1 Confirm you’re running the right Python + classy + clik
```bash
which python
python -c "import sys; print(sys.executable)"
python -c "import classy; import clik; print('classy:', classy.__file__); print('clik:', clik.__file__)"
python -c "import os; print('CLIKROOT=', os.environ.get('CLIKROOT')); print('PYTHONPATH=', os.environ.get('PYTHONPATH','')[:200],'...')"
```

##### 7.2 Confirm clik submodules resolve
```bash
python - <<'PY'
import clik
import clik.lkl, clik.lkl_lensing
print("clik OK:", clik.__file__)
print("lkl OK:", clik.lkl.__file__)
print("lkl_lensing OK:", clik.lkl_lensing.__file__)
PY
```

##### 7.3 Which CLASS does Kosmulator use (from within repo)
```bash
cd /mnt/d/Kosmulator_test/Kosmulator
python - <<'PY'
import classy, sys
print("Python:", sys.executable)
print("classy:", classy.__file__)
PY
```

##### 7.4 Grep/ripgrep searches to confirm bindings and likelihood paths
```bash
cd /mnt/d/Kosmulator_test/Kosmulator

rg -n "import\s+clik|pyclik|clik\.|libclik|ctypes|cffi" Kosmulator_main Observations User_defined_modules.py
rg -n "\.clik|clik_dir|clik_path|plik_rd12|simall_100x143" Kosmulator_main Observations
rg -n "CLIKROOT|LD_LIBRARY_PATH|DYLD_LIBRARY_PATH|PATH=" .
```
Grep fallback form (if rg is missing):
```bash
grep -RIn --exclude-dir=.git "CLIKROOT\|plik\|commander\|simall\|smica\|plc_3\.0\|baseline\|low_l\|hi_l\|lensing\|\.clik" .
```

##### 7.5 Locate clik install + inspect shared library dependencies
```bash
python -c "import clik, inspect; print('clik file:', clik.__file__)"
python -c "import clik, os; print(clik.__file__)"

python - <<'PY'
import clik, pathlib, subprocess, sys
p = pathlib.Path(clik.__file__).resolve().parent
print("clik package dir:", p)
sos = sorted(p.rglob("*.so"))
print("Found .so files:")
for s in sos:
    print(" ", s)

if not sos:
    print("No .so files found under", p)
    sys.exit(1)

so = sos[0]
print("\nRunning ldd on:", so)
subprocess.run(["ldd", str(so)])
PY
```

##### 7.6 Does clik actually load the Planck .clik likelihoods you ship in Kosmulator?
```bash
python - <<'PY'
import os, clik

base = "/mnt/d/Kosmulator_test/Kosmulator/Observations"
hil  = os.path.join(base, "plik_rd12_HM_v22b_TTTEEE.clik")
lowl = os.path.join(base, "simall_100x143_offlike5_EE_Aplanck_B.clik")

print("Trying high-l clik:", hil)
c = clik.clik(hil)
print("  OK: high-l loaded")

print("Trying low-l clik:", lowl)
c2 = clik.clik(lowl)
print("  OK: low-l loaded")
PY
```

### Notes / common pitfalls
- If `import clik` works but loading a likelihood fails, it’s usually:
  * a missing dependency of libclik.so (check with `ldd`), or
  * incorrect CLIKROOT / PYTHONPATH / LD_LIBRARY_PATH, or
  * the .clik directory path you’re loading isn’t readable / incomplete.
