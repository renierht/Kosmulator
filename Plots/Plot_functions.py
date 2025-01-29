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
        elif obs_type == 'f_sigma_8' or  obs_type=='sigma_8':
            if obs_type == "f_sigma_8":
                model = [params['sigma_8'] * (UDM.matter_density_z(z, model_func, params, obs_type) / model_func(z, params, obs_type )**2) ** params['gamma'] for z in redshift_values]
                y_label = '$f\sigma_{8}$'
                return model, y_label
            else:
                model = [(UDM.matter_density_z(z, model_func, params, obs_type) / model_func(z, params, obs_type )**2) ** params['gamma']for z in redshift_values]
                y_label = 'Growth rate'
                return model, y_label
        else:
            model = [25 + 5 * np.log10(UDM.Comoving_distance(model_func, z, params, obs_type) * (1 + z)) for z in redshift_values]
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
    # Generate LaTeX labels for the parameters, handling subscripts like H_0 properly
    # Define a mapping of parameter names to their LaTeX representations 
    parameter_labels = []
    greek_symbols = { 'Omega': r'\Omega', 'omega': r'\omega', 'alpha': r'\alpha', 'beta': r'\beta', 'gamma': r'\gamma', 
                      'delta': r'\delta', 'epsilon': r'\epsilon', 'zeta': r'\zeta', 'eta': r'\eta', 'theta': r'\theta', 
                      'iota': r'\iota', 'kappa': r'\kappa', 'lambda': r'\lambda', 'mu': r'\mu', 'nu': r'\nu', 'xi': r'\xi', 
                      'pi': r'\pi', 'rho': r'\rho', 'sigma': r'\sigma', 'tau': r'\tau', 'upsilon': r'\upsilon', 'phi': r'\phi', 
                      'chi': r'\chi', 'psi': r'\psi', 'Lambda':r'\Lambda'}
    for param in parameters:
        parts = param.split('_') 
        base = parts[0] 
        sub = parts[1] if len(parts) > 1 else ''
        if base in greek_symbols: 
            base_label = greek_symbols[base] 
        else: 
            base_label = base

        if sub: 
            sub_parts = sub.split('_') 
            if sub in greek_symbols:
                sub_label = ''.join([greek_symbols.get(sp, sp) for sp in sub_parts]) 
                formatted_label = ""+str(base_label)+"_{"+str(sub_label)+"}"
            else:
                formatted_label = ""+str(base_label)+"_{"+str(sub_parts[0])+"}"
        else: 
            formatted_label = ""+str(base_label)+""
            
        parameter_labels.append(formatted_label)
    print (parameter_labels)

    return parameter_labels
    
def format_for_latex(input_list):
    """Format a list of strings for LaTeX by adding dollar signs."""
    return [f"${item}$" for item in input_list]

def add_corner_table(g, latex_table, labels, parameter_labels, PLOT_SETTINGS, num_params):
    """Dynamically adjust and add a corner table to a figure."""
    fig = g.fig
    cols = format_for_latex(parameter_labels)
    
    # Calculate table size and position dynamically
    table_width = 0.32  # Start with a default width
    table_height = 0.3
    if num_params > 4:
        table_width = min(0.6, 0.32 + (num_params - 4) * 0.05)  # Increase the width progressively, max out at 0.6
        table_height = min(0.8, 0.3 + (num_params - 4) * 0.03)  # Increase the height progressively, max out at 0.9

    # Dynamically adjust the table's horizontal position
    table_left = 0.62 - (num_params - 4) * 0.05  # Move the table left as the width increases
    table_left = max(0.45, table_left)  # Ensure the table doesn't move too far left
    table_bottom = 0.5 - (table_height / 2) + 0.28  # Keep the table vertically centered

    ax_table = fig.add_axes([table_left, table_bottom, table_width, table_height])
    ax_table.axis("off")

    table = ax_table.table(cellText=latex_table,
                            rowLabels=labels,
                            colLabels=cols,
                            cellLoc="center",
                            loc="center")
    
    # Dynamically scale the font size
    font_size = min(40, 16 + (num_params - 4) * 4)
    table.auto_set_font_size(False)  # Disable auto font size
    table.set_fontsize(font_size)

    # Adjust table cell scale dynamically for better fit
    base_scale_y = max(3.0, 1.0 - (num_params - 4) * 0.05)
    table.scale(1.0, base_scale_y)

    # Dynamically adjust cell height and padding based on the number of parameters
    base_cell_height = 0.15  # Default cell height for a small number of parameters
    base_cell_width = 0.3 
    base_padding = 0.2  # Default padding for a small number of parameters

    if num_params > 4:
        # Scale cell height and padding based on the number of parameters
        cell_height = max(0.02, base_cell_height - (num_params - 4) * 0.03)  # Min height of 0.03
        cell_width = max(0.01, base_cell_width - (num_params - 4) * 0.058)  # Min width of 0.03
        padding = max(0.01, base_padding - (num_params - 4) * 0.01)  # Min padding of 0.01
    else:
        cell_height = base_cell_height
        cell_width = base_cell_width
        padding = base_padding

    # Apply the height and padding adjustments
    for key, cell in table.get_celld().items():
        cell.set_height(cell_height)  # Set the dynamic cell height
        cell.set_width(cell_width)
        cell.PAD = padding  # Set the dynamic padding 
        
    pass