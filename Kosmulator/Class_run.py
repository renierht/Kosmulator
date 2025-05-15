###################################Robert added file###########################
import os
import numpy as np
from classy import Class
import subprocess
import glob
import User_defined_modules as UDM

#from Kosmulator.MCMC_setup import global_model_name

def find_classy_so(model_name):
    so_files = glob.glob(os.path.join("./Classy",model_name, "**", "*.so"), recursive=True)
    if not so_files:
        raise FileNotFoundError("No classy .so file found in model directory.")
    return so_files[0] 

def run_model(model_name):
    model_dir = os.path.join("./Class", model_name)

    makefile_path = os.path.join(model_dir, "Makefile")
    if os.path.exists(makefile_path):

        subprocess.run(["make", "clean", "-C", model_dir], check=True)
        
        subprocess.run(["make", "-C", model_dir], check=True)

###############################################################################
