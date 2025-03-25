# Kosmulator: A Python package for analysing modified gravity and alternative cosmology with MCMC simulations.
Kosmulator is a Python package utalising the EMCEE:Hammer package for running Markov Chain Monte Carlo (MCMC) simulations to study modified gravity and alternative cosmology models. 
The package is designed to be modular and user-friendly, allowing researchers to easily configure simulations, interact with various set of 
observationaldata, and visualize results.

## Requirements
1) EMCEE python packages (controls the MCMC simulation) (https://emcee.readthedocs.io/en/stable/user/install/)
2) Getdist (Plotting packages) (https://getdist.readthedocs.io/en/latest/intro.html)
3) Numpy, Matplotlip, Scipy, h5py, Pandas (https://numpy.org/install/, https://matplotlib.org/stable/install/index.html, https://scipy.org/install/, https://pypi.org/project/h5py/, https://pypi.org/project/pandas/)
4) Generic Python packages: time, sys, os, platform, inspect, warnings, re, shutil (already part of the Python library)
5) LaTeX (MikTeX/TexLive) - This is a required dependency for enhanced plot quality. Follow the instructions below to install LaTeX for your operating system.

## LaTeX Dependencies for Plot Rendering and installation
Kosmulator uses Matplotlib's LaTeX rendering to generate high-quality formatted plots, ensure LaTeX is installed on your system.

Verify LaTeX installation with:
```bash
latex --version
```


If requirement not met or rf you encounter an error such as:
RuntimeError: latex was not able to process the following string: ... (your system is missing some required LaTeX packages (e.g., type1ec.sty)
Follow os installation or update below:
- **Windows**: [MiKTeX Installation Guide](https://miktex.org/howto/install-miktex, ensure that you enable the option for automatic installation of missing packages)
- **macOS**: Install via Homebrew:
  ```bash
  brew install mactex
  ```
- **Linux**: Install the full TeX Live package:
  ```bash
  sudo apt install texlive-full
  sudo apt-get install texlive-latex-recommended texlive-fonts-recommended (if missing packages are required)
  ```
If you prefer not to install a full LaTeX distribution, you can disable LaTeX rendering by setting "latex_enabled": False in your PLOT_SETTINGS within the configuration. This will disable the LaTeX formatting for your plots.

## Installation 
Install package directly from GitHub

git clone https://github.com/renierht/Kosmulator.git

cd Kosmulator

python setup.py install

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
├── MCMC_Chains/             	# Save directory for MCMC simulation chains
├── Plots/          				# Save directory for the analysed output Plots\
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
With MPI:                      mpiexec -n num_cores python Kosmulator.py --OUTPUT_SUFFIX="Your project name" --latex_enabled=True --overwrite=True --plot_table=True
With python multiprocessing:   python Kosmulator.py --OUTPUT_SUFFIX="Your project name" --num_cores=num_cores --latex_enabled=True --overwrite=True --plot_table=True
Using nohup to close terminal: nohup mpiexec -n num_cores python Kosmulator.py --OUTPUT_SUFFIX="Your project name" --latex_enabled=True --overwrite=True --plot_table=True > model_name.log 2>&1 & 

--OUTPUT_SUFFIX: The name under which the folder will be created:
--latex_enabled: Flag to switch latex usages for the Plots on and off
--overwrite:     Flag to overwrite previosuly saved chain or use the original (usefull for making new plots from existing chains)
--plot_table:    Flag to enable or disable plotting the table with the calculated parameter values on the corner plot
--num_cores:     The amount of cores that you want to use on you system. If num_cores=1, then the simulation will run in series.
 
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

## Acknowledgements
I would like to thank the EMCEE: Hammer group for making their MCMC simulation software publically available. This code would not be possible without their hard work. I would also like to thank the **ChatGPT** software for assisting with debugging, improving the code structure, and optimizing features.

## Contributions
Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request, or send your tested code directly to me and I will add it to the software.

## Contact
For questions or feedback, please reach out to [renierht@gmail.com, 25026097@mynwu.ac.za].
