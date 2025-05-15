import numpy as np
import emcee
import os
import pandas as pd
import scipy.linalg as la
import inspect
import warnings
import re
import textwrap
import h5py

def load_mcmc_results(output_path, file_name="tutorial.h5", CONFIG=None):
    """
    Load saved MCMC chains from an HDF5 backend.
    """
    chain_path = os.path.join(output_path, file_name)
    reader = emcee.backends.HDFBackend(chain_path)
    chain = reader.get_chain(discard=CONFIG['burn'], flat=True)
    return chain

#######################Robert added code#####################################
def load_cmb_data(file_path):
    """
    Load cmb data from a file.
    """
    l = np.loadtxt(file_path, usecols = 0)
    Dl = np.loadtxt(file_path, usecols = 1)
    Dl_minus = np.loadtxt(file_path, usecols = 2)
    Dl_plus = np.loadtxt(file_path, usecols = 3)

    l = l[:2499]
    Dl = Dl[:2499]
    Dl_minus = Dl_minus[:2499]
    Dl_plus = Dl_plus[:2499]
        
    return {'l': l,'Dl':Dl,'Dl_plus':Dl_plus,'Dl_minus':Dl_minus}

#############################################################################
    
def load_data(file_path):
    """
    Load observational data from a file.
    """
    data = np.loadtxt(file_path)
    return {"redshift": data[:, 0], "type_data": data[:, 1], "type_data_error": data[:, 2]}

def prepare_pantheonP_data(data, z_min=0.01):
    """
    Pre-filter the PantheonP data: select SNe with redshift > z_min
    or flagged as calibrators, and replace main arrays with their masked versions.
    """
    # Create a boolean mask for good SNe.
    mask = (data["zHD"] > z_min) | (data["IS_CALIBRATOR"] > 0)
    data["mask"] = mask
    data["indices"] = np.where(mask)[0]
    
    # Overwrite original arrays with masked arrays.
    data["zHD"] = data["zHD"][mask]
    data["m_b_corr"] = data["m_b_corr"][mask]
    data["IS_CALIBRATOR"] = data["IS_CALIBRATOR"][mask]
    data["CEPH_DIST"] = data["CEPH_DIST"][mask]
    
    return data

def load_DESI_data(file_path):
    """
    Load DESI VI data from the file and return a dictionary of arrays.
    Only the chosen data points (the last 10 lines, corresponding to type 8 and 6)
    are returned.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find the marker that indicates the beginning of the chosen data
    marker = "#Theese are the chosen one to compute:"
    start_index = 0
    for i, line in enumerate(lines):
        if marker in line:
            start_index = i + 1  # Data starts after this marker
            break

    # Parse the chosen data points
    redshifts = []
    values = []
    errors = []
    types = []

    for line in lines[start_index:]:
        if line.strip() == "" or line.startswith("#"):
            continue
        parts = line.split()
        # Assuming the file columns are: EXP, z_eff, value, error, type
        redshifts.append(float(parts[1]))
        values.append(float(parts[2]))
        errors.append(float(parts[3]))
        types.append(int(parts[4]))

    return {
        "redshift": np.array(redshifts),
        "measurement": np.array(values),
        "measurement_error": np.array(errors),
        "type": np.array(types)
    }

def load_DESI_cov(file_path):
    return np.loadtxt(file_path)

def load_all_data(config):
    """
    Load all datasets specified in the CONFIG dictionary.
    """
    observation_data = {}
    for obs_list in config["observations"]:
        for obs in obs_list:
            file_path = os.path.join("./Observations", f"{obs}.dat")
            if obs == "PantheonP":
                df = pd.read_csv(file_path, sep=r'\s+')
                pantheon_data = {
                    "zHD": df["zHD"].values,
                    "m_b_corr": df["m_b_corr"].values,
                    #"m_b_corr_err_DIAG": df["m_b_corr_err_DIAG"].values,
                    "IS_CALIBRATOR": df["IS_CALIBRATOR"].values,
                    "CEPH_DIST": df["CEPH_DIST"].values,
                    #"biasCor_m_b": df["biasCor_m_b"].values,
                    "cov_path": "./Observations/PantheonP.cov",
                }
                del df
                # Preprocess PantheonP data (now without computing the heavy covariance)
                observation_data[obs] = prepare_pantheonP_data(pantheon_data)
            elif obs == "BAO":
                observation_data[obs] = {"covd1": np.loadtxt(file_path)}
            ####################################Robert added code#############################
            elif obs == 'CMB_TT':
                cmb_data = load_cmb_data(os.path.join("./Observations", "CMB_TT.dat"))
                observation_data[obs] = cmb_data
            elif obs == 'CMB_EE':
                cmb_data = load_cmb_data(os.path.join("./Observations", "CMB_EE.dat"))
                observation_data[obs] = cmb_data
            elif obs == 'CMB_TE':
                cmb_data = load_cmb_data(os.path.join("./Observations", "CMB_TE.dat"))
                observation_data[obs] = cmb_data
            ##################################################################################                
            elif obs == "DESI":
                desi_data = load_DESI_data(os.path.join("./Observations", "Isma_desi_VI.txt"))
                # Read the matrix directly, assuming it is the inverse covariance matrix.
                inv_cov = np.loadtxt(os.path.join("./Observations", "Isma_desi_covtot_VI.txt"))
                desi_data["inv_cov"] = inv_covC
                # Also store it under "cov" so that other functions (e.g. statistical_analysis) find it.
                desi_data["cov"] = inv_cov
                observation_data[obs] = desi_data
            else:
                observation_data[obs] = load_data(file_path)
    return observation_data

def create_config(models, true_values=None, prior_limits=None, observation=None, nwalkers=20, nsteps=200, burn=20, model_name=None):
    """
    Create a CONFIG dictionary dynamically, ensuring each observation has its own parameter set.
    """
    observation = observation or [["SNe"]]
    true_values = true_values or {}
    prior_limits = prior_limits or {}
    
    # Define observation type mappings
    obs_type_map = {
        "JLA": "SNe",
        "Pantheon": "SNe",
        "PantheonP": "SNe",
        "OHD": "OHD",
        "CC": "CC",
        "BAO": "BAO",
        "DESI": "DESI",
        "f_sigma_8": "f_sigma_8",
        "f": "f",
        "CMB_TT": "CMB",
        "CMB_EE": "CMB",
        "CMB_TE": "CMB",
    }

    config, data = {}, {}

    for mod in model_name:
        parameters_list = models[mod]['parameters']  # Now a list of lists

        ndim_list = [len(params) for params in parameters_list]  # Compute dimensions per observation
        adjusted_nwalkers = max(nwalkers, 2 * max(ndim_list) + 2)
        
        # Generate observation types dynamically
        observation_types = [
            [obs_type_map[obs_item] for obs_item in obs_list if obs_item in obs_type_map] 
            for obs_list in observation
        ]

        config[mod] = {
            "parameters": parameters_list,  # List of lists now
            "true_values": [
                [
                    true_values[p] if p in true_values else sum(prior_limits[p]) / 2 if p in prior_limits else 0.5
                    for p in param_set
                ]
                for param_set in parameters_list
            ],
            "prior_limits": [
                {p: prior_limits[p] for p in param_set if p in prior_limits}
                for param_set in parameters_list
            ],
            "observations": observation,
            "observation_types": observation_types, 
            "ndim": ndim_list,
            "nwalker": adjusted_nwalkers,
            "nsteps": nsteps,
            "burn": burn,
            "model_name": model_name,
        }

        # Load the data
        data = load_all_data(config[mod])

    return config, data

def Add_required_parameters(models, observations):
    """
    Modify the parameter list structure: ensure each observation set has the correct parameters.
    Prints warnings when parameters are added automatically.
    """

    params_map = {
        "BAO": ["H_0","r_d"],
        "DESI": ["H_0", "r_d"],
        "PantheonP": ["H_0","M_abs"], 
        "f_sigma_8": ["sigma_8", "gamma"], 
        "f": ["gamma"],
        "JLA": ["H_0"],
        "OHD": ["H_0"],
        "CC": ["H_0"],
        "Pantheon": ["H_0"],
        #######################Robert added code######################################
        "CMB_TT":["h_0","ln_A_s","n_s","Omega_bh^2","tau_reio"],
        "CMB_EE":["h_0","ln_A_s","n_s","Omega_bh^2","tau_reio"],
        "CMB_TE":["h_0","ln_A_s","n_s","Omega_bh^2","tau_reio"],
        ##############################################################################
    }

    for mod, mod_data in models.items():
        core_parameters = mod_data["parameters"][:]  # Copy user-defined parameters
        new_param_list = []  # Create a fresh list for observations
        
        
        # Loop through each observation and assign the correct parameters
        for obs in observations:
            obs_parameters = list(core_parameters)  # Start with core parameters
            added_params = []

            for key, value in params_map.items():
                if key in obs:
                    for v in value:
                        if v not in obs_parameters:
                            obs_parameters.append(v)
                            added_params.append(v)

            # Show a warning if parameters were added
            if added_params:
                print(f"\033[4;31mSafeguard:\033[0m Added {added_params} to the parameter list for {obs} (Required for this observation type).")

            new_param_list.append(obs_parameters)  # Append modified parameter list

        # Ensure we **replace** instead of appending
        mod_data["parameters"] = new_param_list  

    return models

def create_output_directory(model_name, observations, suffix=""):
    """
    Create output directories for saving MCMC results.

    Args:
        model_name (str): Name of the model.
        observations (list): List of observation names.
        suffix (str): Suffix string to create a subdirectory.

    Returns:
        tuple: (save_dir, output_paths)
    """
    if not observations:
        raise ValueError("No observations provided.")
    
    base_dir = "MCMC_Chains"
    if suffix:
        save_dir = os.path.join(base_dir, suffix, model_name)
    else:
        save_dir = os.path.join(base_dir, model_name)

    os.makedirs(save_dir, exist_ok=True)

    output_paths = {}
    for obs in observations:
        # Create a subdirectory for each observation
        obs_dir = os.path.join(save_dir, obs)
        os.makedirs(obs_dir, exist_ok=True)
        output_paths[obs] = obs_dir

    return save_dir, output_paths

def generate_label(obs):
    """Generate a label based on the observation list."""
    return obs[0] if len(obs) == 1 else '+'.join(obs)

def format_elapsed_time(seconds):
    """
    Format elapsed time into a human-readable string.
    """
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    
    if days:
        return f"{days}:{hours:02}:{minutes:02}:{seconds:02} days"
    if hours:
        return f"{hours}:{minutes:02}:{seconds:02} hours"
    if minutes:
        return f"{minutes}:{seconds:02} minutes"
    return f"{seconds} seconds"

def save_stats_to_file(model, folder, stats_list):
    """Save statistical results to a file with aligned columns."""
    file_path = os.path.join(folder, "stats_summary.txt")
    with open(file_path, "w") as f:
        f.write(f"Statistical Results for Model: {model}\n")
        
        # Construct header with fixed widths:
        header = (
            f"{'Observation':<30} | "
            f"{'Log-Likelihood':>18} | "
            f"{'Chi-Squared':>15} | "
            f"{'Reduced Chi-Squared':>25} | "
            f"{'AIC':>10} | "
            f"{'BIC':>10} | "
            f"{'dAIC':>10} | "
            f"{'dBIC':>10}"
        )
        f.write(header + "\n")
        # Separator line matching the header's width:
        f.write("-" * len(header) + "\n")
        
        # Write each row with the same fixed-width formatting.
        for stats in stats_list:
            row = (
                f"{stats['Observation']:<30} | "
                f"{stats['Log-Likelihood']:>18.4f} | "
                f"{stats['Chi_squared']:>15.4f} | "
                f"{stats['Reduced_Chi_squared']:>25.4f} | "
                f"{stats['AIC']:>10.3f} | "
                f"{stats['BIC']:>10.3f} | "
                f"{stats['dAIC']:>10.3f} | "
                f"{stats['dBIC']:>10.3f}"
            )
            f.write(row + "\n")

def save_interpretations_to_file(model, folder, interpretations_list):
    """Save interpretations to a file with aligned, wrapped commentary in columns."""
    file_path = os.path.join(folder, "interpretations_summary.txt")
    
    # Set fixed widths for each column.
    obs_width = 30
    diag_width = 50
    aic_width = 35
    bic_width = 35

    with open(file_path, "w") as f:
        f.write(f"Interpretations for Model: {model}\n\n")
        
        # Create header row.
        header = (
            f"{'Observation':<{obs_width}} | "
            f"{'Reduced Chi2 Diagnostics':<{diag_width}} | "
            f"{'AIC Interpretation':<{aic_width}} | "
            f"{'BIC Interpretation':<{bic_width}}"
        )
        f.write(header + "\n")
        total_width = obs_width + diag_width + aic_width + bic_width + 9
        f.write("-" * total_width + "\n")
        
        # Process each interpretation.
        for interp in interpretations_list:
            obs = interp["Observation"]
            diag = interp["Reduced Chi2 Diagnostics"]
            aic_interp = interp["AIC Interpretation"]
            bic_interp = interp["BIC Interpretation"]
            
            # Wrap the text for each commentary column.
            diag_lines = textwrap.wrap(diag, width=diag_width)
            aic_lines = textwrap.wrap(aic_interp, width=aic_width)
            bic_lines = textwrap.wrap(bic_interp, width=bic_width)
            
            # For the observation column we usually have one line.
            obs_line = obs.ljust(obs_width)
            
            # Determine the number of lines needed for this row.
            max_lines = max(1, len(diag_lines), len(aic_lines), len(bic_lines))
            
            # Write out the row line by line.
            for i in range(max_lines):
                obs_str = obs_line if i == 0 else " " * obs_width
                diag_str = diag_lines[i] if i < len(diag_lines) else ""
                aic_str = aic_lines[i] if i < len(aic_lines) else ""
                bic_str = bic_lines[i] if i < len(bic_lines) else ""
                line = (
                    f"{obs_str:<{obs_width}} | "
                    f"{diag_str:<{diag_width}} | "
                    f"{aic_str:<{aic_width}} | "
                    f"{bic_str:<{bic_width}}"
                )
                f.write(line + "\n")

def save_latex_table_to_file(model, folder, table_data):
    """Save LaTeX tables to a file with aligned columns."""
    aligned_table, parameter_labels, observation_names = table_data
    file_path = os.path.join(folder, "aligned_table.txt")
    
    # Set fixed widths: observation column 30 characters, each parameter column 30 characters.
    obs_width = 30
    param_width = 30

    with open(file_path, "w") as f:
        # Write header title.
        f.write(f"Aligned Table for Model: {model}\n")
        
        # Construct header line.
        header = f"{'Observation':<{obs_width}} | " + " | ".join(f"{col:<{param_width}}" for col in parameter_labels) + "\n"
        f.write(header)
        
        # Construct a separator line.
        total_width = obs_width + 3 + len(parameter_labels) * (param_width + 3) - 3
        f.write("-" * total_width + "\n")
        
        # Write each row with fixed-width formatting.
        for obs, row in zip(observation_names, aligned_table):
            row_str = f"{obs:<{obs_width}} | " + " | ".join(f"{cell:<{param_width}}" for cell in row) + "\n"
            f.write(row_str)


def print_stats_table(model, stats_list):
    """Print the statistical results table to the console."""
    print(f"Statistical Results for Model: {model}")
    print("Observation            | Log-Likelihood | Chi-Squared | Reduced Chi-Squared | AIC     | BIC     | dAIC   | dBIC")
    print("-" * 120)
    for stats in stats_list:
        print(
            f"{stats['Observation']:<22} | "
            f"{stats['Log-Likelihood']:<15.4f} | "
            f"{stats['Chi_squared']:<12.4f} | "
            f"{stats['Reduced_Chi_squared']:<20.4f} | "
            f"{stats['AIC']:<8.3f} | {stats['BIC']:<8.3f} | "
            f"{stats['dAIC']:<8.3f} | {stats['dBIC']:<8.3f}"
        )
    print("\n")

