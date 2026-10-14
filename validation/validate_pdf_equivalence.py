"""Compare direct transcriptions of the PDF's generic formulas with production code.

Run from the repository root in Kosmulator_test:
    python validation/validate_pdf_equivalence.py

The explicit expressions are transcribed, not parsed from Markdown. They do
not use the production density formulas or its coupling-mapping helper.
The reference implementation is invoked only for comparisons. This is a
finite generic-branch consistency check, not a proof over all parameters.
Independent conservation-ODE/exceptional-limit tests live in
 tests/test_warm_background.py. The report is written beside this script.
"""
import sys
from pathlib import Path
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from Kosmulator_main import ide_background as I

def explicit_pdf_densities(m, a, p):
    w = p['w']
    v = p['w_dm']
    d = p['delta']
    c = p['Omega_m'] - p['Omega_b']
    e = 1 - p['Omega_m'] - p['Omega_r']
    r = c / e
    u = w - v
    if m <= 3:
        dc, de = {1: (p['delta_dm'], p['delta_de']), 2: (d, d), 3: (d, -d)}[m]
        D = np.sqrt((dc + de + u) ** 2 - 4 * dc * de)
        first = e * (dc - de + u - D) + c * (dc - de - u - D)
        second = e * (dc - de + u + D) + c * (dc - de - u + D)
        plus = a ** (1.5 * (dc - de - w - v - 2 + D))
        minus = a ** (1.5 * (dc - de - w - v - 2 - D))
        dm = -(dc - de + u + D) / (4 * u * D) * first * plus + (dc - de + u - D) / (4 * u * D) * second * minus
        den = (dc - de - u + D) / (4 * u * D) * first * plus - (dc - de - u - D) / (4 * u * D) * second * minus
        total = -first / (2 * D) * plus + second / (2 * D) * minus
    elif m == 4:
        dm = c * a ** (-3 * (1 + v - d))
        den = (e + c * d / (d + u) * (1 - a ** (3 * (d + u)))) * a ** (-3 * (1 + w))
        total = dm + den
    elif m == 5:
        dm = (c + e * d / (d + u) * (1 - a ** (-3 * (d + u)))) * a ** (-3 * (1 + v))
        den = e * a ** (-3 * (1 + w + d))
        total = dm + den
    elif m == 6:
        f = ((1 + r * a ** (3 * (u + d))) / (1 + r)) ** (-d / (u + d))
        dm = c * a ** (-3 * (1 + v - d)) * f
        den = e * a ** (-3 * (1 + w)) * f
        total = dm + den
    elif m == 7:
        f = a ** (-3 * (1 + v - u * d / (u - d))) * (((u + d * r) * a ** (-3 * u) + r * (u - d)) / (u * (1 + r))) ** (d / (u - d))
        dm = c * f
        den = e * ((u + d * r) * a ** (-3 * u) - d * r) / u * f
        total = dm + den
    elif m == 8:
        f = a ** (-3 * (1 + w + u * d / (u - d))) * (((u * r + d) * a ** (3 * u) + u - d) / (u * (1 + r))) ** (d / (u - d))
        dm = c * ((u * r + d) * a ** (3 * u) - d) / (u * r) * f
        den = e * f
        total = dm + den
    elif m == 9:
        f = ((2 * u + d * (1 + 3 * w) + d * (1 + 3 * v) * r) * a ** (-1.5 * (2 * u + d * (1 + 3 * w))) + u * (2 + 3 * d) * r) / ((2 * u + d * (1 + 3 * w)) * (1 + r))
        pre = a ** (-6 * (1 + v + d) / (2 + 3 * d))
        dm = c * pre * f ** (-3 * d / (2 + 3 * d))
        total = (c + e) * pre * f ** (2 / (2 + 3 * d))
        den = total - dm
    else:
        f = ((2 * u + d * (1 + 3 * v)) * r + d * (1 + 3 * w)) * a ** (1.5 * (2 * u + d * (1 + 3 * v))) / ((2 * u + d * (1 + 3 * v)) * (1 + r)) + u * (2 - 3 * d) / ((2 * u + d * (1 + 3 * v)) * (1 + r))
        pre = a ** (-6 * (1 + w - d) / (2 - 3 * d))
        den = e * pre * f ** (3 * d / (2 - 3 * d))
        total = (c + e) * pre * f ** (2 / (2 - 3 * d))
        dm = total - den
    return (np.array([dm, den]), total)
rng = np.random.default_rng(25)
maximum = 0
hc = 0
switch = 0
for m in range(1, 11):
    for j in range(10):
        p = dict(Omega_m=0.307, Omega_b=0.048, Omega_r=8e-05, w=rng.uniform(-1.3, -0.7), w_dm=rng.uniform(-0.07, 0.07), delta=rng.uniform(0.003, 0.06) * (-1 if m == 10 else 1), delta_dm=0.02, delta_de=0.03)
        a = np.geomspace(0.02, 3, 81)
        got, total = explicit_pdf_densities(m, a, p)
        ref = np.array(I.densities(m, a, p))
        scale = np.maximum(np.max(abs(ref), axis=0), 1e-20)
        err = np.max(abs(got - ref) / scale)
        maximum = max(maximum, err)
        assert err < 1e-09, (m, err)
        h = np.sqrt(total + p['Omega_b'] / a ** 3 + p['Omega_r'] / a ** 4)
        expected = I.expansion(m, 1 / a - 1, p)
        err = np.max(abs(h / expected - 1))
        hc = max(hc, err)
        assert err < 1e-09, (m, err)
        if m >= 9:
            w = p['w']
            v = p['w_dm']
            d = p['delta']
            u = w - v
            r = (p['Omega_m'] - p['Omega_b']) / (1 - p['Omega_m'] - p['Omega_r'])
            if m == 9:
                z = (-2 * u * (1 + 3 * v) * r / ((1 + 3 * w) * (2 * u + d * (1 + 3 * w) + d * (1 + 3 * v) * r))) ** (2 / (3 * (2 * u + d * (1 + 3 * w)))) - 1
            else:
                z = (-2 * u * (1 + 3 * w) / ((1 + 3 * v) * ((2 * u + d * (1 + 3 * v)) * r + d * (1 + 3 * w)))) ** (-2 / (3 * (2 * u + d * (1 + 3 * v)))) - 1
            dm, de = I.densities(m, 1 / (1 + z), p)
            err = abs(((1 + 3 * v) * dm + (1 + 3 * w) * de) / (dm + de))
            switch = max(switch, err)
            assert err < 1e-10
print(f'PASS: 100 PDF-form cases, 8100 scale-factor points; max scaled density error {maximum:.3g}; H relative error {hc:.3g}; switching residual {switch:.3g}')
report = dict(seed=25, models=10, cases_per_model=10, parameter_cases=100, scale_factor_points_per_case=81, scale_factor_points=8100, scale_factor_range=[0.02, 3.0], w_range=[-1.3, -0.7], w_dm_range=[-0.07, 0.07], coupling_magnitude_range=[0.003, 0.06], model_10_negative_coupling=True, model_1_delta_dm=0.02, model_1_delta_de=0.03, Omega_m=0.307, Omega_b=0.048, Omega_r=8e-05, max_scaled_component_density_error=float(maximum), max_relative_H_error=float(hc), max_switch_proxy_residual=float(switch), density_and_H_tolerance=1e-09, switch_tolerance=1e-10, scope='Generic branches only; independent ODE and exceptional-limit checks are in tests/test_warm_background.py')
Path(__file__).with_name('pdf_equivalence_results.json').write_text(json.dumps(report, indent=2) + '\n')
