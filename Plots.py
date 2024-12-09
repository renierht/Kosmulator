import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import numpy as np

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
    
    print ("Generated labels:", parameter_labels)
    return parameter_labels
    
def autocorrPlot(autocorr,index):
    n = 100 * np.arange(1, index + 1)
    y = autocorr[:index]
    plt.plot(n, n / 100.0, "--k")
    plt.plot(n, y, color='m')
    plt.xlim(0, n.max()+100)
    plt.ylim(0, y.max() + 0.5 * (y.max() - y.min()))
    plt.xlabel("number of steps")
    plt.ylabel(r"mean $\hat{\tau}$");
        
def make_CornerPlot(Samples, CONFIG=None):
    
    if CONFIG is None:
        raise ValueError("CONFIG must be provided from the main script")
    # Enable LaTeX rendering 
    plt.rc('text', usetex=True) 
    plt.rc('font', family='serif')
    distributions =[]
    labels =[]
    parameter_labels = []
    
    g = plots.get_subplot_plotter()
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add=0.4
    g.settings.title_limit_fontsize = 14

    parameter_labels = greek_Symbols(parameters=CONFIG['parameters'])
    print ("Generated labels:", parameter_labels)
    
    if not Samples: 
        raise ValueError("Samples dictionary is empty or not properly set up.")
    
    for obs in Samples:
        if not Samples[obs].size: 
            raise ValueError(f"Samples for '{obs}' are empty.")
            
        #print (parameter_labels)
        #['r'\Omega_m', r'H_0'']
        distribution = MCSamples(samples=Samples[obs],names = parameter_labels, labels = parameter_labels)
        distributions.append(distribution)
        labels.append(obs)
        
    g.triangle_plot(distributions, 
                    filled=True, 
                    legend_labels=labels, 
                    legend_loc='upper right', 
                    line_args=[{'ls':'--', 'color':'black'}, {'lw':1, 'color':'blue'},  {'lw':1, 'color':'blue'}], 
                    contour_colors=['green','blue','red'],
                    title_limit=3,
                    markers={'x2':0})
    plt.show()