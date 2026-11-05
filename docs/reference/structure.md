# Project structure

```text
Kosmulator/
├── Kosmulator.py                # Run settings: models, data groups, priors, switches (start here)
├── User_defined_modules.py      # Models: E(z), restrictions, model registry, CMB helpers
│
├── Kosmulator_main/             # The inference engine
│   ├── Config.py                # Loads the data and builds each group's parameters and priors
│   ├── MCMC_setup.py            # Runs every model on every group (start-up checks, engines, pools)
│   ├── Kosmulator_MCMC.py       # Samples one model on one group (zeus or emcee)
│   ├── Statistical_packages.py  # Likelihoods
│   ├── Post_processing.py       # Best fits and statistics (chi^2, AIC, BIC, DIC, WAIC, ...)
│   ├── Class_run.py             # Builds, caches and loads CLASS per model; Planck likelihood code
│   ├── rd_helpers.py, rd_class.py   # The sound horizon r_d (rd_Class: CLASS with the model's E(z))
│   ├── theta_mc.py, cmb_early.py    # 100 theta_MC, and the compressed CMB (ThetaStar_CMB)
│   ├── planck_pr4.py            # Planck PR4 CamSpec likelihood (CMB_hil_PR4; LGPL-3.0)
│   ├── ground_cmb.py            # ACT DR6 and SPT-3G likelihoods, P-ACT and CMB-SPA
│   ├── sampler_moves.py         # emcee and zeus proposal moves
│   ├── peek.py                  # --peek: look at a running chain
│   ├── citations.py             # What to cite for each run
│   ├── doctor.py                # kosmulator-doctor
│   ├── get_spt_data.py          # Restores the SPT-3G data
│   ├── constants.py             # Constants, thresholds and less-used data options
│   └── utils.py                 # Command-line options and shared helpers
│
├── Class/                       # CLASS sources, one copy per model fitted to CMB data
│   ├── LCDM_v/
│   ├── wowaCDM_v/
│   └── f1CDM_v/                 # modified CLASS for the f(T) model
│
├── Observations/                # All data sets and likelihood files (see DATA_SOURCES.md)
├── AlterBBN_files/              # Kosmulator's interface to AlterBBN (optional)
├── Plots/                       # Plotting code; results go to Plots/Saved_Plots/<suffix>/
├── MCMC_Chains/                 # Chains, per run
├── Statistical_analysis_tables/ # Statistics and citations, per run
│
├── docs/, .readthedocs.yaml     # This documentation (Sphinx, Markdown pages)
├── environment.yml              # conda environment
├── setup.py, pyproject.toml, requirements.txt
├── LICENSE, LICENSES/
└── README.md
```

As a user you normally edit only **`Kosmulator.py`** and **`User_defined_modules.py`**. The less-used data options (systematic covariances of Pantheon and the cosmic chronometers, the convergence thresholds) are in `Kosmulator_main/constants.py`.
