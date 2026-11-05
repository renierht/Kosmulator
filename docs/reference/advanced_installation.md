# Advanced installation

::::{admonition} Most users do not need this page
:class: note
The standard [installation](../getting_started.md) covers CLASS (Kosmulator builds its own, per model) and the Planck likelihoods (clipy, with the data in `Observations/`). This page is for two optional extras: the **compiled Planck CLIK** in place of clipy, and **live AlterBBN** calls for `BBN_DH_AlterBBN`.
::::

The commands assume Ubuntu or WSL and a conda environment called `kosmulator` (from `environment.yml`); macOS is similar, with its own compilers. They build everything in `~/Kosmulator_extras`, next to a Kosmulator clone in `~/Kosmulator`.

## Compiled Planck CLIK

Kosmulator uses CLIK instead of clipy when `import clik` works; the run log records which one was used. clipy's low-$\ell$ EE likelihood (`CMB_lowl`) needs NumPy 2.0 or newer, so CLIK is the route for an environment that must keep an older NumPy.

::::{dropdown} 1. System packages

```bash
sudo apt update
sudo apt install -y build-essential gfortran python3-dev libcfitsio-dev pkg-config cmake
```
::::

::::{dropdown} 2. Download the Planck likelihood code

```bash
mkdir -p ~/Kosmulator_extras/Clik
cd ~/Kosmulator_extras/Clik

wget -O COM_Likelihood_Code-v3.0_R3.10.tar.gz \
  "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Code-v3.0_R3.10.tar.gz"
tar -xzf COM_Likelihood_Code-v3.0_R3.10.tar.gz
```

The likelihood data are already in `Observations/`. The baseline data tarball (`COM_Likelihood_Data-baseline_R3.00.tar.gz`, from the same archive) is needed only for the plik_lite and Commander folders, should they ever be missing.
::::

::::{dropdown} 3. Build CLIK

Before building, apply the known fix to the lensing module (a leftover of the 2015 to 2018 transition):

```bash
conda activate kosmulator
cd ~/Kosmulator_extras/Clik/code/plc_3.0/plc-3.1

sed -i.bak '79,83c\      pars_proxy=nm.PyArray_ContiguousFromAny(apars,nm.NPY_DOUBLE,1,1)' src/python/clik/lkl_lensing.pyx

python waf configure --install_all_deps
python waf install
```
::::

::::{dropdown} 4. Load CLIK with the environment

This sets `CLIKROOT` and the paths CLIK needs every time the environment is activated:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

cat > $CONDA_PREFIX/etc/conda/activate.d/clik.sh <<'SH'
export CLIKROOT=$HOME/Kosmulator_extras/Clik/code/plc_3.0/plc-3.1
source "$CLIKROOT/bin/clik_profile.sh"

python - <<'PY'
import importlib.machinery as m
import os, pathlib, sys

suf = m.EXTENSION_SUFFIXES[0]
candidates = []
clikroot = os.environ.get("CLIKROOT")
if clikroot:
    candidates.append(pathlib.Path(clikroot) / "lib/python/site-packages")
for entry in os.environ.get("PYTHONPATH", "").split(":"):
    if entry.strip():
        candidates.append(pathlib.Path(entry.strip()))
for entry in sys.path:
    if entry:
        candidates.append(pathlib.Path(entry))

pkg = None
for base in candidates:
    p = base / "clik"
    if p.exists() and p.is_dir():
        pkg = p
        break
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
::::

::::{dropdown} 5. Check

```bash
conda deactivate
conda activate kosmulator

python - <<'PY'
import clik
import clik.lkl, clik.lkl_lensing
print("OK: CLIK imported successfully")
PY

cd ~/Kosmulator
kosmulator-doctor
```
::::

## Live AlterBBN

`BBN_DH_AlterBBN` reads the AlterBBN grid shipped in `Observations/BBN/bbn_grid.npz`. AlterBBN itself is needed only when the priors go beyond that grid (another $N_\mathrm{eff}$, a sampled $N_\mathrm{eff}$ or $\tau_n$, a wider $\Omega_b h^2$ range), when Kosmulator builds a new grid. The other BBN tags never need it, and Kosmulator does not use AlterBBN's standalone executable.

::::{dropdown} Build AlterBBN and Kosmulator's interface

AlterBBN v1.4 goes next to the Kosmulator folder, and Kosmulator's interface library into `Kosmulator/AlterBBN_files/`, where Kosmulator looks for it (no environment variable needed). Inside the conda environment `${CC:-gcc}` is conda's compiler, otherwise the system gcc.

```bash
cd ~                                              # the folder that holds Kosmulator/
git clone https://github.com/espensem/AlterBBN.git
cd AlterBBN
make CC="${CC:-gcc}" CFLAGS="-O3 -pipe -fPIC"
"${CC:-gcc}" -O3 -fPIC -shared -o ../Kosmulator/AlterBBN_files/libkosmo_bbn.so \
    ../Kosmulator/AlterBBN_files/kosmo_bbn.c -I. -Lsrc -lbbn -lm
```

A library elsewhere is found through `export KOSMO_BBN_LIB=/full/path/to/libkosmo_bbn.so` (in a conda environment: `conda env config vars set KOSMO_BBN_LIB=...`, then reactivate it).
::::

::::{dropdown} Check AlterBBN against the shipped grid

From the Kosmulator folder. AlterBBN should reproduce the shipped grid at $N_\mathrm{eff} = 3.044$ (anything below about $10^{-8}$ is fine), and `kosmulator-doctor` then reports `Live AlterBBN (optional) ... libkosmo_bbn.so found`.

```bash
python - <<'PY'
import numpy as np
from Kosmulator_main import Statistical_packages as SP
run_bbn = SP._import_run_bbn()
g = np.load("Observations/BBN/bbn_grid.npz")
idx = np.linspace(0, g["obh2"].size - 1, 25).astype(int)
d = [np.log(run_bbn(float(g["obh2"][i]), 3.044, 879.4)["D_H"]) - g["logDH"][i, 0, 0] for i in idx]
print("AlterBBN found; largest |ln D/H - grid| at N_eff 3.044 over 25 nodes:", max(abs(x) for x in d))
PY
```
::::

## A separately installed CLASS

Kosmulator does not need one. A globally installed `classy` would be used only for the flat-ΛCDM $r_d$ of BAO + BBN groups, which Kosmulator's own `Class/LCDM_v` build provides as well.
