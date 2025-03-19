import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import argparse
from Kosmulator import Config, EMCEE, Statistic_packages
from Plots import Plots as MP  # Custom module for creating plots (e.g., autocorrelation plot)
import User_defined_modules as UDM  # Custom module with user-defined functions for cosmological calculations
from Kosmulator.MCMC_setup import run_mcmc_for_all_models
from Plots.Plot_functions import print_aligned_latex_table
import scipy.linalg as la
import gc

# Safeguard: Check Python version and warn if outdated
if sys.version_info[0] == 2:
    print(f'\033[4;31mNote\033[0m: Your Python version {sys.version_info[0]}.{sys.version_info[1]} is outdated. Be careful when executing the program.')

#'OHD', 'JLA', 'Pantheon', 'PantheonP', 'CC', 'BAO', 'f_sigma_8', 'f'
# Constants for the simulation
#model_names = ["f1CDM","f1CDM_v"]#"f3CDM","f3CDM_v"]#"f1CDM","f1CDM_v"]#,"f2CDM","f2CDM_v",]
model_names = ["LCDM"]
observations =  [['DESI']]#['CC','BAO','PantheonP','f_sigma_8']]#,['PantheonP'],['CC','BAO','PantheonP','f','f_sigma_8'],['CC','BAO','PantheonP','f_sigma_8'],['CC','BAO','PantheonP','f'], ['CC','BAO','PantheonP']]
true_model = "LCDM" # True model will always run first irregardless of model names, due to the statistical analysis
nwalkers: int = 10
nsteps: int = 100
burn: int = 10
convergence = 0.01

prior_limits = {
    "Omega_m": (0.10, 0.4),
    "H_0": (60.0, 80.0),
    "r_d": (100.0, 200.0),
    "M_abs": (-22.0, -15.0),
    "zeta": (0.0,0.3),
    "gamma": (0.4, 0.7),
    "sigma_8": (0.5, 1.0),
    "n": (0.0,0.5), #0.0,0.5
    "p": (0.0, 1.0),
    "Gamma": (2.0, 10.0),
    "q0": (-0.8, -0.01),
    "q1": (-0.75, 1.0),
    "beta": (0.01, 5.0),
    "alpha": (0.1, 100.0),
    "Omega_w": (0.0, 1.0),
}

true_values = {
    "Omega_m": 0.315,
    "H_0": 67.4,
    "gamma": 0.55,
    "sigma_8": 0.8,
    "q0": -0.537,
    "ns": 0.96,
    "As": 3.1,
    "Omega_b": 0.045,
    "rd": 147.5,
    "M_abs": -19.2,
}

# Create an argument parser
parser = argparse.ArgumentParser(description="Run Kosmulator MCMC simulation.")
parser.add_argument("--num_cores", type=int, default=8,
                    help="Number of cores to use (default: 8).")
parser.add_argument("--OUTPUT_SUFFIX", type=str, default="",
                    help="Suffix for the MCMC chain folder (default: empty).")
parser.add_argument("--latex_enabled", type=lambda x: (str(x).lower() == "true"),
                    default=True, help="Enable LaTeX in plots (default: True).")
parser.add_argument("--use_mpi", type=lambda x: (str(x).lower() == "true"),
                    default=None, help="Force MPI usage (default: auto-detect).")
parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == "true"),
                    default=False, help="Overwrite existing MCMC chains (default: False).")
parser.add_argument("--plot_table", type=lambda x: (str(x).lower() == "true"),
                    default=True, help="Enable Best-fit table on plots (default: True).")
args = parser.parse_args()

# Auto-detect MPI if not explicitly forced
if args.use_mpi is None:
    try:
        from mpi4py import MPI
        # If running locally, the MPI world will have size 1.
        if MPI.COMM_WORLD.Get_size() == 1:
            use_mpi = False
        else:
            use_mpi = True
    except ImportError:
        use_mpi = False
else:
    use_mpi = args.use_mpi

num_cores = args.num_cores
OUTPUT_SUFFIX = args.OUTPUT_SUFFIX
latex_enabled = args.latex_enabled
plot_table = args.plot_table
overwrite = args.overwrite


full_colors = ['r', 'b', 'green', 'cyan', 'purple', 'grey', 'yellow', 'm',
               'k', 'gray', 'orange', 'pink', 'crimson', 'darkred', 'salmon']

PLOT_SETTINGS = {
    "color_schemes": full_colors[:len(observations)],
    "line_styles": ["-", "--", ":", "-."],
    "marker_size": 4,
    "legend_font_size": 12,
    "title_font_size": 12,
    "label_font_size": 12,
    "latex_enabled": latex_enabled,
    "dpi": 300,
    "autocorr_save_path": "./Plots/auto_corr/",
    "Table": plot_table,
    "table_anchor": (0.98, 1.0),  # pushes the table further up

    # Additional table-specific settings for finer control:
    "base_table_width": 0.4,
    "width_increment": 0.045,
    "base_table_height": 0.3,
    "height_increment": 0.04,
    "font_base": 14,
    "font_reduction": 0.7,
    "min_font_size": 10,
    "cell_height_base": 0.10,
    "cell_scaling": 3.8,
}

# Enable LaTeX settings for plots if configured
if PLOT_SETTINGS["latex_enabled"]:
    if sys.version_info >= (3, 10):
        import shutil
        if shutil.which("latex"):
            plt.rc("text", usetex=True)
            plt.rc("font", family="Arial")
            plt.rc("text.latex", preamble=r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}")
    elif sys.version_info < (3, 10):
        from distutils.spawn import find_executable
        if find_executable("latex"):
            plt.rc("text", usetex=True)
            plt.rc("font", family="Arial")

# Ensure the true_model is at the front of the model_names list
if true_model in model_names:
    model_names.remove(true_model)  # Remove it from its current position
model_names.insert(0, true_model)  # Insert it at the front

# --- MPI Setup ---
# --- MPI Setup ---
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    rank = 0
    comm = None

# Only master loads heavy CONFIG, data, and models; then broadcast them.
if rank == 0:
    print(f"\033[33m{'#'*48}\033[0m", flush=True)
    print(f"\033[33m####\033[0m Safeguards + Warnings", flush=True)
    print(f"\033[33m{'#'*48}\033[0m", flush=True)

    models_local = UDM.Get_model_names(model_names)
    models_local = Config.Add_required_parameters(models_local, observations)
    CONFIG, data = Config.create_config(
        models=models_local,
        true_values=true_values,
        prior_limits=prior_limits,
        observation=observations,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burn=burn,
        model_name=model_names,
    )
    print(f"CONFIG: {CONFIG}", flush=True)
    print(f"data: {data}", flush=True)
else:
    CONFIG, data, models_local = None, None, None

if comm is not None:
    CONFIG = comm.bcast(CONFIG, root=0)
    data = comm.bcast(data, root=0)
    models_local = comm.bcast(models_local, root=0)

pantheon_cov = None
pantheon_required = any(
    "PantheonP" in obs for obs in CONFIG[list(models_local.keys())[0]]['observations']
)

if pantheon_required:
    if comm is not None:
        if rank == 0:
            # Always use the precomputed mask from the PantheonP data.
            mask = data["PantheonP"]["mask"]
            cov_raw = np.loadtxt("./Observations/PantheonP.cov")[1:].reshape(1701, 1701)
            reduced_cov = cov_raw[np.ix_(mask, mask)]
            pantheon_cov = la.cholesky(reduced_cov, lower=True, overwrite_a=True)
        else:
            pantheon_cov = None
        pantheon_cov = comm.bcast(pantheon_cov, root=0)
    else:
        mask = data["PantheonP"]["mask"]
        cov_raw = np.loadtxt("./Observations/PantheonP.cov")[1:].reshape(1701, 1701)
        reduced_cov = cov_raw[np.ix_(mask, mask)]
        pantheon_cov = la.cholesky(reduced_cov, lower=True, overwrite_a=True)

# 🔹 Step 3: Now create the Schwimmbad MPI pool AFTER broadcasting
if use_mpi:
    try:
        from schwimmbad import MPIPool
        pool = MPIPool()

        # Only master process continues; workers wait and then exit.
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        print("Using MPI Pool with schwimmbad.")
    except ImportError:
        pool = None
else:
    pool = None


# Main execution block: use the broadcasted CONFIG and models_local.
if __name__ == "__main__":
    start = time.time()

    All_Samples = run_mcmc_for_all_models(
        models=models_local,  # use the broadcasted models dictionary
        observations=observations,
        CONFIG=CONFIG,
        data=data,
        overwrite=overwrite,
        convergence=convergence,
        PLOT_SETTINGS=PLOT_SETTINGS,
        use_mpi=use_mpi,
        num_cores=num_cores,
        suffix=OUTPUT_SUFFIX,
        pool=pool,
        pantheon_cov=pantheon_cov,
    )

    if pantheon_cov is not None:
        del pantheon_cov
        gc.collect()

    if pool is not None and rank == 0:
        pool.close()

    if rank == 0:
        print(f"\n\033[33m{'#'*48}\033[0m", flush=True)
        print(f"\033[33m####\033[0m Generating Plots :)", flush=True)
        print(f"\033[33m{'#'*48}\033[0m", flush=True)
        
        # Re-read the Pantheon+ covariance matrix from the provided file path.
        if "PantheonP" in data:
            cov_raw = np.loadtxt(data["PantheonP"]["cov_path"])[1:].reshape(1701, 1701)
            mask = data["PantheonP"]["mask"]
            reduced_cov = cov_raw[np.ix_(mask, mask)]
            data["PantheonP"]["cov"] = la.cholesky(reduced_cov, lower=True, overwrite_a=True)
        
        if "PantheonP" in data and "cov" not in data["PantheonP"]:
            data["PantheonP"]["cov"] = pantheon_cov

        best_fit_values, All_LaTeX_Tables, statistical_results = MP.generate_plots(
            All_Samples, CONFIG, PLOT_SETTINGS, data, true_model
        )

        for model_name, (aligned_table, parameter_labels, observation_names) in All_LaTeX_Tables.items():
            print(f"\nModel: {model_name} Aligned LaTeX Table:")
            print_aligned_latex_table(aligned_table, parameter_labels, observation_names)

        end = time.time()
        formatted_time = Config.format_elapsed_time(end - start)
        print(f"\n\n\033[33m{'#'*75}\033[0m")
        print(f"\033[33m#### \033[0m")
        print(f"\033[33m#### All models processed successfully in a total time of {formatted_time}!!!\033[0m")
        print(f"\033[33m#### \033[0m")
        print(f"\033[33m#### Thank you for using Kosmulator :D\033[0m")
        print(f"\033[33m#### \033[0m")
        print(f"\033[33m{'#'*75}\033[0m")
