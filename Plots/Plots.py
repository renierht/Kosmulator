import matplotlib.pyplot as plt
from getdist import plots, MCSamples
from Kosmulator.Config import generate_label
import numpy as np
import shutil
import sys
import os

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
        #parameter_names.append(param)
    
    return parameter_labels
    
def autocorrPlot(autocorr, index, model_name, color, obs):
    folder_path = "./Plots/auto_corr/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    n = 100 * np.arange(1, index + 1)
    y = autocorr[:index]
    plt.plot(n, n / 100.0, "--k")

    if index == 1:
        plt.plot(n, y, label = f'{generate_label(obs)}', color = color)
    else:
        plt.plot(n, y, color = color)
    plt.xlim(0, 10000)
    if y.max()>100:
         plt.ylim(0, y.max()+10)
    else:
        plt.ylim(0, 100)
    plt.title(f"Auto-Correlator: Check for convergence - {model_name} model")
    plt.xlabel("Number of steps")
    plt.ylabel(r"Mean $\hat{\tau}$")
    plt.legend()
    plt.savefig(f"./Plots/auto_corr/{model_name}.png", dpi=200)
       
def make_CornerPlot(Samples, CONFIG=None, model_name = None, color = ['red']):
    if CONFIG is None:
        raise ValueError("CONFIG must be provided from the main script")

    print ("\033[4;31mNote\033[0m: GetDist's read chains ignores burn in, due to EMCEE already applying a burn fraction") 
    folder_path = "./Plots/corner_plots/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    distributions =[]
    labels =[]
    parameter_labels = []
    
     # Enable LaTeX rendering if possible
    if sys.version_info[0] == 3 and  sys.version_info[1]>=10:
        import shutil
        if shutil.which('latex'): 
            plt.rc('text', usetex=True) 
            plt.rc('font', family='Arial')
            parameter_labels = greek_Symbols(parameters=CONFIG['parameters'])
    elif sys.version_info[0] == 3 and  sys.version_info[1]<10:
        from distutils.spawn import find_executable
        if find_executable('latex'): 
            plt.rc('text', usetex=True) 
            plt.rc('font', family='Arial')
            parameter_labels = greek_Symbols(parameters=CONFIG['parameters'])
    else:
        parameter_labels = CONFIG['parameters']
    
    #print ("Generated corner plot labels:", parameter_labels)

    g = plots.get_subplot_plotter(subplot_size_ratio = 0.8)
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add=0.4
    g.settings.title_limit_fontsize = 14
    g.settings.tight_layout = True
    
    line_styles = ['-','--',':', '-.','-','--',':', '-.',]
    line_widths = [1.2, 1.2, 1.5, 1.5, 1.2, 1.2, 1.5, 1.5]
    line_colors = color
    line_args = [{'ls': ls, 'lw': lw, 'color': color} 
                             for ls, lw, color in zip(line_styles, line_widths * len(line_styles), line_colors[:len(line_styles)])]
    
    if not Samples: 
        raise ValueError("Samples dictionary is empty or not properly set up.")
    
    for obs in Samples:
        if not Samples[obs].size: 
            raise ValueError(f"Samples for '{obs}' are empty.")
        distribution = MCSamples(samples=Samples[obs],names = parameter_labels, labels = parameter_labels)
        distributions.append(distribution)
        labels.append(obs)
    g.triangle_plot(distributions, 
                    filled=True, 
                    legend_labels=labels, 
                    legend_loc='upper right', 
                    line_args=line_args, 
                    contour_colors=color,
                    title_limit=3,
                    markers={'x2':0})
    plt.savefig(f"./Plots/corner_plots/{model_name}.png", dpi = 300)
    #plt.show()