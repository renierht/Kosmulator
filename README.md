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

### LaTeX Installation
To use Kosmulator's LaTeX features for enhanced plot quality, ensure LaTeX is installed on your system.

Verify LaTeX installation with:
```bash
latex --version
```

If requirement not met, follow os installation below:
- **Windows**: [MiKTeX Installation Guide](https://miktex.org/howto/install-miktex)
- **macOS**: Install via Homebrew:
  ```bash
  brew install mactex
  ```
- **Linux**: Install the full TeX Live package:
  ```bash
  sudo apt install texlive-full
  ```

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
- **Resume Simulations**: Restart simulations using saved MCMC chains. N.B. Chains can be restarted, but not necessarily recommended.


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
		
## References
1) Original version: 	Hough, R.T., Abebe, A. and Ferreira, S.E.S., 2020. Viability tests of f (R)-gravity models with Supernovae Type 1A data. The European Physical Journal C, 80(8), p.787. https://doi.org/10.1140/epjc/s10052-020-8342-7
2) EMCEE package: 		Foreman-Mackey, D., Hogg, D.W., Lang, D. and Goodman, J., 2013. emcee: the MCMC hammer. Publications of the Astronomical Society of the Pacific, 125(925), p.306. https://doi.org/10.1086/670067

Observations:

1) JLA: 				1) M. Hicken, P. Challis, S. Jha et al., CfA3: 185 Type Ia Supernova light curves from the CfA. Astrophys. J. 700, 331–357 (2009). https://doi.org/10.1088/0004-637X/700/1/331
						2) J.D. Neill, M. Sullivan, D.A. Howell et al., The local hosts of Type Ia Supernovae. Astrophys. J. 707, 1449–1465 (2009). https://doi.org/10.1088/0004-637X/707/2/1449
						3) A. Conley, J. Guy, M. Sullivan et al., Supernova constraints and systematic uncertainties from the first three years of the Supernova Legacy Survey. Astrophys. J. Suppl. Ser. 192, 1 (2010). https://doi.org/10.1088/0067-0049/192/1/1
2) Pantheon:	
3) Pantheon+: 
4) OHD:
5) CC:
6) BAO:
6) fsigma8:
7) sigma8:

## Acknowledgements
I would like to thank the EMCEE: Hammer group for making their MCMC simulation software publically available. This code would not be possible without their hard work. I would also like to thank the **ChatGPT** software for assisting with debugging, improving the code structure, and optimizing features.

## Contributions
Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request, or send your tested code directly to me and I will add it to the software.

## Contact
For questions or feedback, please reach out to [renierht@gmail.com, 25026097@mynwu.ac.za].