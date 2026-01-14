# Kosmulator: A Python package for analysing modified gravity and alternative cosmology with MCMC simulations.
Kosmulator is a Python package utalising the EMCEE:Hammer package for running Markov Chain Monte Carlo (MCMC) simulations to study modified gravity and alternative cosmology models. 
The package is designed to be modular and user-friendly, allowing researchers to easily configure simulations, interact with various set of 
observationaldata, and visualize results.

## Requirements

Kosmulator is a research-grade cosmological inference framework and depends on several scientific Python libraries, numerical backends, and external tools.
It is strongly recommended to install Kosmulator inside a dedicated conda environment.

------------------------------------------------------------
1) MCMC Samplers
------------------------------------------------------------

Kosmulator supports both EMCEE and ZEUS as sampling backends.

- EMCEE (ensemble MCMC sampler)
  Documentation: https://emcee.readthedocs.io/en/stable/user/install/

- ZEUS (slice-sampling ensemble MCMC)
  Documentation: https://github.com/minaskar/zeus

Recommended installation via conda:

    conda install -c conda-forge emcee zeus-mcmc


------------------------------------------------------------
2) Plotting and Chain Analysis
------------------------------------------------------------

- GetDist (posterior analysis and plotting)
  Documentation: https://getdist.readthedocs.io/en/latest/intro.html

Install via pip:

    pip install getdist


------------------------------------------------------------
3) Core Scientific Python Stack
------------------------------------------------------------

Kosmulator requires the following numerical and data-handling packages:

- NumPy
- SciPy
- Matplotlib
- h5py
- Pandas

Recommended installation via conda:

    conda install numpy scipy matplotlib h5py pandas

These packages are used throughout Kosmulator for:
- likelihood evaluation
- numerical integration
- interpolation
- plotting
- I/O of cosmological datasets


------------------------------------------------------------
4) Standard Python Library Modules
------------------------------------------------------------

The following modules are used internally but require no installation,
as they are part of the Python standard library:

    time, sys, os, platform, inspect, warnings, re, shutil


------------------------------------------------------------
LaTeX Dependencies for Plot Rendering
------------------------------------------------------------

Kosmulator uses Matplotlib’s LaTeX rendering backend to produce
publication-quality plots (axis labels, legends, annotations, tables).

Verify LaTeX installation with:

    latex --version

If LaTeX is installed correctly, this command should return version
information.


------------------------------------------------------------
Common LaTeX Errors
------------------------------------------------------------

If LaTeX is not installed or is missing required packages, you may
encounter errors such as:

    RuntimeError: latex was not able to process the following string

or messages referencing missing files (e.g. type1ec.sty).


------------------------------------------------------------
Install LaTeX by Operating System
------------------------------------------------------------

Windows:
- Install MiKTeX:
  https://miktex.org/howto/install-miktex
- IMPORTANT: Enable automatic installation of missing packages during setup.

macOS:
- Install via Homebrew:

    brew install mactex

Linux (Ubuntu / Debian):
- Install the full TeX Live distribution:

    sudo apt install texlive-full

- If additional packages are still missing:

    sudo apt-get install texlive-latex-recommended texlive-fonts-recommended


------------------------------------------------------------
Disabling LaTeX Rendering (Optional)
------------------------------------------------------------

If you prefer not to install LaTeX, you can disable LaTeX rendering
in Kosmulator by setting:

    "latex_enabled": False

inside your PLOT_SETTINGS configuration.

Plots will still be generated, but without LaTeX formatting.


------------------------------------------------------------
Installation
------------------------------------------------------------

1) Clone the repository:

    git clone https://github.com/renierht/Kosmulator.git
    cd Kosmulator


2) (Recommended) Create a conda environment:

    conda create -n Kosmulator python=3.11
    conda activate Kosmulator


3) Install Python dependencies:

    conda install -c conda-forge numpy scipy matplotlib h5py pandas emcee zeus-mcmc cython astropy
    pip install getdist


4) Install Kosmulator:

    python setup.py install

------------------------------------------------------------
Installing CLASS (Cosmic Linear Anisotropy Solving System)
------------------------------------------------------------

Kosmulator relies on the CLASS Boltzmann solver via its Python interface
(classy). CLASS is NOT installed automatically and must be compiled
locally to ensure full compatibility with Kosmulator, modified gravity
models, and external likelihoods (e.g. CLIK).

The recommended and supported installation method is to:
1) Clone CLASS locally
2) Compile the C/C++ backend
3) Install the Python interface (classy) into your active conda environment

Do NOT rely on pre-built wheels or system-wide CLASS installations.


------------------------------------------------------------
Prerequisites
------------------------------------------------------------

Before installing CLASS, ensure the following are available:

- A working C/C++ compiler (gcc / g++)
- make
- Python 3.10 or newer
- pip
- NumPy and Cython installed in the active environment

On Linux (Ubuntu/Debian), install build tools with:

    sudo apt install build-essential

On macOS:

    xcode-select --install


------------------------------------------------------------
Step 1: Clone the CLASS Repository
------------------------------------------------------------

From your chosen working directory:

    git clone https://github.com/lesgourg/class_public.git CLASS

This creates a local CLASS directory that Kosmulator will interface with.


------------------------------------------------------------
Step 2: Activate Your Conda Environment
------------------------------------------------------------

CLASS must be built inside the same environment that will run Kosmulator.

Example:

    conda activate Kosmulator


------------------------------------------------------------
Step 3: Compile the CLASS C Backend
------------------------------------------------------------

Navigate into the CLASS directory and build:

    cd CLASS
    make clean
    make -j

If compilation is successful, you should see a binary named `class`
and a static library `libclass.a`.


------------------------------------------------------------
Step 4: Install the Python Interface (classy)
------------------------------------------------------------

Still inside the CLASS directory, install the Python bindings:

    python -m pip install .

This will compile and install the `classy` module into your active
conda environment.

If a previous installation exists and causes conflicts, you may need:

    python -m pip install . --no-build-isolation


------------------------------------------------------------
Step 5: Verify the Installation
------------------------------------------------------------

Run the following Python test:

python - << 'PY'
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

If this runs without errors and prints cosmological values,
CLASS and classy are installed correctly.


------------------------------------------------------------
Important Notes for Kosmulator Users
------------------------------------------------------------

- Kosmulator requires a FULL CLASS build with:
  - background
  - perturbations
  - power spectra
  - growth functions
- Building CLASS inside the Kosmulator environment is mandatory.
- Mixing system-installed CLASS with conda Python environments
  will almost always lead to runtime failures.
- CLASS version mismatches can silently break likelihood calculations.

If Kosmulator fails to import `classy`, re-check:
- that the correct conda environment is active
- that `classy` is installed inside that environment
- that no other CLASS installations exist on your system PATH



------------------------------------------------------------
Important Notes
------------------------------------------------------------

- Kosmulator requires a working installation of CLASS and CLIK.
- CLASS and CLIK are NOT installed automatically.
- Kosmulator will not run correctly without these external backends.
- Installation instructions for CLASS and CLIK are provided in the
  following sections of the README.


# Test Run
In Kosmulator.py --  set model_names to ['LCDM'], 
					 observations to [['JLA']], 
					 nwalkers: int = 10 
					 nsteps: int = 200 
					 burn: int = 10
then run the command python Kosmulator.py in your terminal. If it ran successfully, it has been installed correctly!

## Features

- **Configurable Simulation Framework**: Define and manage simulation parameters via a configuration file.
- **Built-in Statistical Tools**: Use pre-built Gaussian statistical models for observational data, with the addition of a covariance matrix calculation for cosmic chronometer and BAO data.
- **Customizable Models**: Extend the functionality by adding new observational models or datasets. Currently we have SNe Type 1A, OHD, CC, BAO, and RSD data.
- **Visualization**: Generate basic auto correlation and MCMC informative plots to analyze the simulation results. Feel free to add different plots
- **Re-use save chains**.


## Project Structure

```plaintext
Kosmulator/
├── Kosmulator/          		# Core package\
│   ├── __init__.py            	# Program initialisation	
│   ├── Config.py            	# Configuration management
│   ├── EMCEE.py             	# Main MCMC simulation script
│   ├── Statistic_packages.py   # Statistical functions and models
│   ├── MCMC_setup.py   		# Calls the EMCEE function based on your CONFIG setup
├── Observations/            	# Observational data directory
├── Plots/          				# Save directory for the analysed output Plots
│   ├── Plots.py                # User-facing script for generating plots
│   ├── Plot_functions.py       # Random functions to improve plots.	
├── Kosmulator.py  				# User-facing script to set up simulations
├── User_defined_modules.py       	# User-facing script for alternative gravity/cosmology models
├── LICENSE                  	# License for the project
├── README.md                	# Project description
├── setup.py                 	# Configuration for packaging
```

## Setting up your model
1) Use Kosmulator.py to set up the parameters of the MCMC simulation for your gravity model and configuration file.
2) Use User_defined_modules.py to set up a dictonary entry of your model which the MCMC simulation needs to analyse.
3) In Kosmulator.py, specify you model name and which observation types you want it analysed against.
4) Ex.  	model_names = [ "LCDM", "your_model_name"] # if you want to want to run multiple models
			observations = [['JLA'],['OHD'], ['CC', 'OHD', 'Pantheon']] 
		# If you want to test your model against multiple observations. If the list contains multiple entries
		# the MCMC simulation will combine their likelihoods and minimized the combined likelihood.

## Multiprocessing runs:
```bash
With MPI:                      mpiexec -n num_cores python Kosmulator.py --OUTPUT_SUFFIX="Your project name" --latex_enabled=True --overwrite=True --plot_table=True
With python multiprocessing:   python Kosmulator.py --OUTPUT_SUFFIX="Your project name" --num_cores=num_cores --latex_enabled=True --overwrite=True --plot_table=True
Using nohup to close terminal: nohup mpiexec -n num_cores python Kosmulator.py --OUTPUT_SUFFIX="Your project name" --latex_enabled=True --overwrite=True --plot_table=True > model_name.log 2>&1 & 

--OUTPUT_SUFFIX: The name under which the folder will be created:
--latex_enabled: Flag to switch latex usages for the Plots on and off
--overwrite:     Flag to overwrite previosuly saved chain or use the original (usefull for making new plots from existing chains)
--plot_table:    Flag to enable or disable plotting the table with the calculated parameter values on the corner plot
--num_cores:     The amount of cores that you want to use on you system. If num_cores=1, then the simulation will run in series.
```
 
## LaTeX Dependencies for Plot Rendering
Kosmulator uses Matplotlib's LaTeX rendering to generate high-quality formatted plots. If you encounter an error such as:
---RuntimeError: latex was not able to process the following string: ...
this typically means that your system is missing some required LaTeX packages (e.g., type1ec.sty).	

## References
1) Original version: 	Hough, R.T., Abebe, A. and Ferreira, S.E.S., 2020. Viability tests of f (R)-gravity models with Supernovae Type 1A data. EPCJ, 80(8), p.787. https://doi.org/10.1140/epjc/s10052-020-8342-7
2) EMCEE package: 		Foreman-Mackey, D., Hogg, D.W., Lang, D. et al., 2013. emcee: the MCMC hammer. PASP, 125(925), p.306. https://doi.org/10.1086/670067

Observations:

1) JLA: 				1) M. Hicken, P. Challis, S. Jha et al., 2009. CfA3: 185 Type Ia Supernova light curves from the CfA. ApJ, 700, p.331. https://doi.org/10.1088/0004-637X/700/1/331
						2) J.D. Neill, M. Sullivan, D.A. Howell et al., 2009. The local hosts of Type Ia Supernovae. ApJ, 707, p.1449. https://doi.org/10.1088/0004-637X/707/2/1449
						3) A. Conley, J. Guy, M. Sullivan et al., 2010. Supernova constraints and systematic uncertainties from the first three years of the Supernova Legacy Survey. ApJ Suppl. Ser., 192, p.1. https://doi.org/10.1088/0067-0049/192/1/1
2) Pantheon:	
3) Pantheon+: 			1) D. Brout, D. Scolnic, D., B. Popovic et al., 2022. The Pantheon+ analysis: cosmological constraints. ApJ, 938(2), p.110. https://doi.org/10.3847/1538-4357/ac8e04
4) OHD:
5) CC:					1) M. Moresco, R. Jimenez, L. Verde et al., 2020. Setting the stage for cosmic chronometers. II. Impact of stellar population synthesis models systematics and full covariance matrix. ApJ, 898(1), p.82.  https://doi.org/10.3847/1538-4357/ab9eb0
6) BAO/DESI:			1) A. G. Adame, J. Aguilar, S. Ahlen et al., 2024. DESI 2024 VI: Cosmological constraints from the measurements of baryon acoustic oscillations. arXiv preprint https://arxiv.org/pdf/2404.03002.
6) fsigma8:				1) F. Skara and L. Perivolaropoulos, L., 2020. Tension of the EG statistic and redshift space distortion data with the Planck-Λ CDM model and implications for weakening gravity. PRD, 101(6), p.063521.
7) sigma8:				1) L. Perenon, J. Bel, R. Maartens et al. 2019. Optimising growth of structure constraints on modified gravity. JCAP, 2019(06), p.020. https://doi.org/10.1088/1475-7516/2019/06/020

## Citation
The current version does not have a dedicated research paper, you can just cite the original paper and note that Kosmulator is an updated MCMC simulation from the one used in that paper. There is plans for the future to have a dedicated research paper for Kosmulator when all features such as compatibility with classy have been built-in.

## Acknowledgements
I would like to thank the EMCEE: Hammer group for making their MCMC simulation software publically available. This code would not be possible without their hard work. I would also like to thank the **ChatGPT** software for assisting with debugging, improving the code structure, and optimizing features.

## Contributions
Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request, or send your tested code directly to me and I will add it to the software.

## Contact
For questions or feedback, please reach out to [renierht@gmail.com, 25026097@mynwu.ac.za].
