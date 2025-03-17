import os
import numpy as np
from scipy.interpolate import interp1d
import User_defined_modules as UDM  # Custom module with user-defined functions
import re

def compute_model(model_name, redshift_values, params, obs_type="SNe"):
        model_func = UDM.Get_model_function(model_name)
        if obs_type == "OHD" or obs_type=="CC":
            model = [params["H_0"] * model_func(z, params, obs_type) for z in redshift_values]
            y_label = "Hubble Parameter H(z)"
            return model, y_label
        elif obs_type in ["f_sigma_8", "f"]:
            if obs_type == "f_sigma_8":
                # Compute f_sigma_8 model
                model = [
                    params["sigma_8"]
                    * (UDM.matter_density_z(z, model_func, params, obs_type) ** params["gamma"])
                    * np.exp(-1 * UDM.integral_term(z, model_func, params, obs_type))
                    for z in redshift_values
                ]
                y_label = "$f\sigma_{8}$"
                return model, y_label
            elif obs_type == "f":
                # Compute f model
                model = [
                    UDM.matter_density_z(z, model_func, params, obs_type) ** params["gamma"]
                    for z in redshift_values
                ]
                y_label = "$f$"
                return model, y_label
        else:
            # Distance modulus (SNe)
            comoving_distances = UDM.Comoving_distance_vectorized(model_func, redshift_values, params, obs_type)
            model = 25 + 5 * np.log10(comoving_distances * (1 + redshift_values))
            y_label = "Distance modulus: $m - M$ ($Mpc$)"
            return model, y_label

def setup_folder(folder_path):
    """Ensure the output folder exists."""
    os.makedirs(folder_path, exist_ok=True)
        
def fetch_best_fit_values(combined_best_fit):
    """Extract median, upper, and lower values from combined best-fit dictionary."""
    params_combined_median = {param: values[0] for param, values in combined_best_fit.items()}
    params_combined_upper = {param: values[1] for param, values in combined_best_fit.items()}
    params_combined_lower = {param: values[2] for param, values in combined_best_fit.items()}
    return params_combined_median, params_combined_upper, params_combined_lower

def prepare_data(obs_set, data, params_combined_median):
    """Combine redshift, type data, and type data error for observations."""
    combined_redshift = []
    combined_type_data = []
    combined_type_data_error = []

    for obs in obs_set:
        obs_data = data[obs]
        if obs == 'PantheonP':
            redshift = obs_data['zHD']
            type_data = obs_data['m_b_corr'] - params_combined_median['M_abs']
            type_data_error = np.zeros(len(type_data))
        else:
            redshift = obs_data["redshift"]
            type_data = obs_data["type_data"]
            type_data_error = obs_data["type_data_error"]

        combined_redshift.extend(redshift)
        combined_type_data.extend(type_data)
        combined_type_data_error.extend(type_data_error)

    return combined_redshift, combined_type_data, combined_type_data_error
    
def greek_Symbols(parameters=None):
    """
    Generate LaTeX labels for the parameters, handling subscripts (e.g. H_0) properly.
    
    If parameters is a list of strings (e.g. ['Omega_m', 'H_0']), returns a list of formatted labels.
    If parameters is a list of lists (e.g. [['Omega_m', 'H_0'], ['Omega_m', 'H_0', 'M_abs']]),
    returns a list of lists of formatted labels.
    """
    # Mapping of parameter names to their LaTeX representations.
    greek_symbols = {
        'Omega': r'\Omega', 'omega': r'\omega', 'alpha': r'\alpha', 'beta': r'\beta',
        'gamma': r'\gamma', 'delta': r'\delta', 'epsilon': r'\epsilon', 'zeta': r'\zeta',
        'eta': r'\eta', 'theta': r'\theta', 'iota': r'\iota', 'kappa': r'\kappa',
        'lambda': r'\lambda', 'mu': r'\mu', 'nu': r'\nu', 'xi': r'\xi', 'pi': r'\pi',
        'rho': r'\rho', 'sigma': r'\sigma', 'tau': r'\tau', 'upsilon': r'\upsilon',
        'phi': r'\phi', 'chi': r'\chi', 'psi': r'\psi', 'Lambda': r'\Lambda'
    }

    def format_param(param):
        # Split on '_' to separate the base from the subscript.
        parts = param.split('_')
        base = parts[0]
        sub = parts[1] if len(parts) > 1 else ''
        # Get the LaTeX version for the base if available.
        base_label = greek_symbols.get(base, base)
        if sub:
            # Split further in case the subscript itself has an underscore.
            sub_parts = sub.split('_')
            # If the whole sub-string is in the mapping, use it;
            # otherwise, just use the first element.
            if sub in greek_symbols:
                sub_label = ''.join([greek_symbols.get(sp, sp) for sp in sub_parts])
            else:
                sub_label = sub_parts[0]
            formatted_label = f"{base_label}_{{{sub_label}}}"
        else:
            formatted_label = f"{base_label}"
        return formatted_label

    # Check if parameters is a list of lists or a list of strings.
    if parameters and isinstance(parameters[0], list):
        # Process each sublist individually.
        formatted = [ [ format_param(param) for param in sublist ] for sublist in parameters ]
    else:
        # Assume parameters is a list of strings.
        formatted = [ format_param(param) for param in parameters ]
    
    #print(formatted)
    return formatted
 
def format_for_latex(input_list):
    """Format a list of strings for LaTeX by adding dollar signs."""
    return [f"${item}$" for item in input_list]

def add_corner_table(g, latex_table, labels, PLOT_SETTINGS, parameter_labels, flat_parameters, num_params):
    """
    Dynamically adjust and add a corner table to a figure.
    
    The table's size and position are scaled based on the number of free parameters,
    and several base values (for width, height, font size, etc.) can be provided in
    PLOT_SETTINGS to help fine-tune the layout for different plots.
    
    Parameters:
      g : get-dist object that has a 'fig' attribute (the matplotlib Figure)
      latex_table : list of lists
          LaTeX-formatted table entries.
      labels : list of str
          Row labels for the table.
      PLOT_SETTINGS : dict
          Dictionary with additional settings. Recognized keys include:
            - "table_anchor": (x, y) tuple for the table's top-right anchor in normalized coordinates 
                              (default: (0.98, 0.98)).
            - "base_table_width": base table width (default: 0.4).
            - "width_increment": extra width per additional column beyond 4 (default: 0.05).
            - "base_table_height": base table height (default: 0.3).
            - "height_increment": extra height per additional parameter beyond 4 (default: 0.05).
            - "font_base": base font size (default: 16).
            - "font_reduction": reduction in font size per parameter beyond 4 (default: 1).
            - "min_font_size": minimum font size (default: 8).
            - "cell_height_base": base cell height (default: 0.1).
            - "cell_scaling": scaling factor for cell height relative to (num_params/4) (default: 4).
      parameter_labels : list of str
          Column labels (the full set of parameters).
      flat_parameters : any
          (Unused here, but available for potential further adjustments.)
      num_params : int
          Number of free parameters (affects scaling).
    """
    fig = g.fig

    # Extract settings from PLOT_SETTINGS with defaults.
    table_anchor      = PLOT_SETTINGS.get("table_anchor", (0.98, 0.98))
    base_table_width  = PLOT_SETTINGS.get("base_table_width", 0.4)
    width_increment   = PLOT_SETTINGS.get("width_increment", 0.05)
    base_table_height = PLOT_SETTINGS.get("base_table_height", 0.3)
    height_increment  = PLOT_SETTINGS.get("height_increment", 0.05)
    font_base         = PLOT_SETTINGS.get("font_base", 16)
    font_reduction    = PLOT_SETTINGS.get("font_reduction", 1)
    min_font_size     = PLOT_SETTINGS.get("min_font_size", 8)
    cell_height_base  = PLOT_SETTINGS.get("cell_height_base", 0.1)
    cell_scaling      = PLOT_SETTINGS.get("cell_scaling", 4)

    # Format column labels for LaTeX rendering.
    cols = format_for_latex(parameter_labels)
    num_cols = len(parameter_labels)

    # Pad each row in the latex_table so that each has exactly num_cols entries.
    padded_latex_table = [row + [""] * (num_cols - len(row)) for row in latex_table]

    # Calculate table width and height based on the number of columns and free parameters.
    table_width = base_table_width + width_increment * max(0, num_cols - 4)
    table_height = base_table_height + height_increment * max(0, num_params - 4)

    # Position the table using the table_anchor.
    # table_anchor represents the (x,y) normalized coordinate for the table's top-right corner.
    table_left = table_anchor[0] - table_width
    table_bottom = table_anchor[1] - table_height

    ax_table = fig.add_axes([table_left, table_bottom, table_width, table_height])
    ax_table.axis("off")  # Hide axes

    # Calculate font size dynamically.
    font_size = max(min_font_size, font_base - font_reduction * max(0, num_params - 4))

    # Calculate cell height dynamically.
    cell_height = cell_height_base / max(1, num_params / cell_scaling)

    # Create and configure the table.
    table = ax_table.table(
        cellText=padded_latex_table,
        rowLabels=labels,
        colLabels=cols,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, cell_height)

    # Optionally adjust each cell's properties.
    for key, cell in table.get_celld().items():
        cell.set_height(cell_height)
        cell.set_edgecolor("black")

def print_aligned_latex_table(latex_table, parameter_labels, observation_names):
    """
    Print the aligned LaTeX table in a readable format for the terminal.
    The header row is printed in blue, while the observation names in the first column of each row are printed in red.
    The observation column is set to 30 characters wide.
    """
    # Set the observation column width to 30 characters.
    obs_col_width = 30

    # ANSI escape codes for colors.
    blue = "\033[34m"   # Blue for the header.
    red = "\033[31m"    # Red for observation values.
    reset = "\033[0m"   # Reset color.

    # Construct the header row with the first column ("Observation") and the parameter columns.
    header = ["Observation"] + parameter_labels
    header_str = f"{header[0]:<{obs_col_width}} | " + " | ".join(f"{col:<30}" for col in header[1:])
    print(blue + header_str + reset)
    total_width = obs_col_width + 3 + 31 * len(parameter_labels)
    print("-" * total_width)

    # Print each row: the observation name in the first column (red) and the remaining columns uncolored.
    for obs_name, row in zip(observation_names, latex_table):
        obs_str = f"{obs_name:<{obs_col_width}}"
        obs_str = red + obs_str + reset
        row_str = " | ".join(f"{col:<30}" for col in row)
        print(f"{obs_str} | {row_str}")

def align_table_to_parameters(latex_table, parameters):
    """
    Align rows of a LaTeX table so that each row has an entry for every parameter
    in the full parameter list. Missing values are replaced with an empty string.
    
    Parameters:
    -----------
    latex_table : list of list of str
        Each sub-list contains the LaTeX values for one row.
    parameters : list of list of str
        The list of parameter names for each row. The first element should be the full
        parameter list (i.e. all parameters in the correct order).
    
    Returns:
    --------
    aligned_table : list of list of str
        The LaTeX table with each row aligned to the full parameter list.
    """
    # Use the full parameter list from the first row
    full_param_list = parameters[0]
    
    aligned_table = []
    
    # Loop over each row and its corresponding parameter list.
    for row, obs_params in zip(latex_table, parameters):
        # Start with an empty row having one slot per full parameter.
        aligned_row = [''] * len(full_param_list)
        
        # Fill in the values in the correct position.
        for value, param in zip(row, obs_params):
            try:
                # Find the index of this parameter in the full list.
                index = full_param_list.index(param)
                aligned_row[index] = value
            except ValueError:
                # In case the parameter is not found in the full list, skip it.
                pass
        
        aligned_table.append(aligned_row)
    
    return aligned_table

def extract_observation_data(obs, obs_data, params_median):
    """
    Extract redshift, type data, and error for a given observation.
    """
    if obs == 'PantheonP':
        redshift = obs_data['zHD']
        type_data = obs_data['m_b_corr'] - params_median['M_abs']
        type_data_error = np.zeros(len(type_data))
    else:
        redshift = obs_data["redshift"]
        type_data = obs_data["type_data"]
        type_data_error = obs_data["type_data_error"]
    return redshift, type_data, type_data_error
