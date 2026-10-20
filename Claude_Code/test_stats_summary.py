"""Test the proposed save_stats_to_file (Claude_Code/utils.py) against values from the real wowaCDM_v row."""
import importlib.util, sys, os, csv, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # repo root, for Kosmulator_main.constants
spec = importlib.util.spec_from_file_location("utils_proposed", os.path.join(os.path.dirname(__file__), "utils.py"))
U = importlib.util.module_from_spec(spec); spec.loader.exec_module(U)

out = os.path.join(os.path.dirname(__file__), "test_stats_summary", "out")
# Real row (Statistical_analysis_tables/.../wowaCDM_v/stats_summary.txt); k = (AIC - chi2)/2 = 5
real = {"Observation": "DESI+DR2+DESY5 [rd: free]", "Log-Likelihood": -822.5229, "Chi_squared": 1645.0457,
        "Reduced_Chi_squared": 0.8955, "AIC": 1655.046, "BIC": 1682.639, "AICc": 1655.078, "DIC": 1655.020,
        "WAIC": 1652.672, "dAIC": -9.417, "dBIC": 1.620, "dAICc": -9.398, "dDIC": -9.526, "dChi": -13.417,
        "sigma": 3.2, "dWAIC": -14.946, "n_params": 5}
# Synthetic: long name, NaN DIC/WAIC, numpy scalar and 1-element array inputs, no extra parameters
import numpy as np
synth = dict(real, Observation="DESI+DR2 (BAO only) [rd: Gaussian prior, long name]", DIC=float("nan"),
             WAIC=np.float64("nan"), dDIC=float("nan"), dWAIC=float("nan"), dChi=np.array([0.0]), sigma=0.0, n_params=3)

U.save_stats_to_file("wowaCDM_v", out, [real, synth], reference_model="LCDM_v")
os.makedirs(out + "/ref", exist_ok=True)
U.save_stats_to_file("LCDM_v", out + "/ref", [real], reference_model="LCDM_v")
os.makedirs(out + "/noref", exist_ok=True)
U.save_stats_to_file("wowaCDM_v", out + "/noref", [real])   # old call signature still works

# Checks
rows = list(csv.DictReader(open(out + "/stats_summary.csv")))
assert len(rows) == 2 and rows[0]["Reference_Model"] == "LCDM_v" and rows[1]["Observation"] == synth["Observation"]
assert abs(float(rows[0]["DOF"]) - 1645.0457 / 0.8955) < 1e-6 and math.isnan(float(rows[1]["WAIC"]))
assert float(rows[0]["dAIC"]) == -9.417 and float(rows[0]["Free_parameters"]) == 5
txt = open(out + "/stats_summary.txt").read().splitlines()
widths = {len(l) for l in txt[txt.index(next(l for l in txt if l.startswith("Statistic "))):] if l}
assert len(widths) == 1, f"table lines not equal width: {widths}"
print("all checks passed; max line width:", max(len(l) for l in txt))
