# MCMC Cosmology

MCMC Cosmology is a Python package for running Markov Chain Monte Carlo (MCMC) simulations to study modified gravity models. 
The package is designed to be modular and user-friendly, allowing researchers to easily configure simulations, interact with data, and visualize results.

## Installation 
Install package directly from GitHub
git clone https://github.com/renierht/MCMC_cosmology.git

## Features

- **Configurable Simulation Framework**: Define and manage simulation parameters via a configuration file.
- **Built-in Statistical Tools**: Use pre-built Gaussian statistical models for observational data, with the addition of a covariance matrix calculation for cosmic chronometer data.
- **Customizable Models**: Extend the functionality by adding new observational models or datasets. Currently we have Supernovae Type 1A, Observable Hubble data, and Cosmic Chronometer data.
- **Visualization**: Generate basic auto correlation and MCMC informative plots to analyze the simulation results. Feel free to add different plots
- **Resume Simulations**: Restart simulations using saved MCMC chains. N.B. Chains can be restarted, but not necessarily recommended.

## Project Structure

```plaintext
MCMC_cosmology/
├── mcmc_cosmology/         # Core package
│   ├── Config.py           # Configuration management
│   ├── EMCEE.py            # Main MCMC simulation script
│   ├── Statistic_packages.py  # Statistical functions and models
├── Observations/           # Observational data directory
├── MCMC_Chains/            # Directory for MCMC simulation chains
├── Modified_gravity_MCMC.py  # User-facing script to set up simulations
├── Plots.py                # User-facing script for generating plots
├── README.md               # Project description
├── setup.py                # Configuration for packaging
├── LICENSE                 # License for the project
```

## Setting up examples
1) Use Modified_gravity_MCMC.py to set up your modified gravity model and configuration file.
2) Specify your observational data in CONFIG['observation'], even when new observations were added
3) Run simulation using EMCEE.run_mcmc(...)

## Contributions
Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request. 

## Contact
For questions or feedback, please reach out to [renierht@gmail.com].
