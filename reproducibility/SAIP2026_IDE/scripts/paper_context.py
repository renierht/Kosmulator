"""Paper likelihood context recovered from the historical polish_mle.py.

Only regime settings and configuration/covariance setup are retained here.
Likelihood polishing is performed by the production Model_comparison module.
The top-level CLASS constants intentionally match the recovered helper.
"""

import os

import numpy as np

import Kosmulator as KC
import User_defined_modules as UDM

from Kosmulator_main import Config
from Kosmulator_main import Kosmulator_MCMC as KM
from Kosmulator_main import constants as K
from Kosmulator_main.utils import compute_pantheon_cov, init_mpi


# ---------------------------------------------------------------------
# CLASS-derived r_d settings: same prescription as the production runs
# ---------------------------------------------------------------------
K.DERIVE_RD_WITH_MODEL_CLASS = True
K.RD_CLASS_OMEGA_B = 0.048
K.RD_CLASS_N_EFF = 3.044
K.RD_CLASS_SUM_MNU_EV = 0.06
K.RD_CLASS_N_NCDM = 3


REGIMES = {
    "lcdm": {
        "model": "LCDM_v",
        "switches": None,
    },
    "iw": {
        "model": "NonLinear_IDE_2",
        "switches": {
            "ALLOW_NEGATIVE_ENERGIES": True,
            "ALLOW_BIG_RIP": True,
            "ALLOW_DOOM_FACTOR_INSTABILITIES": True,
        },
    },
    "plus_iw": {
        "model": "NonLinear_IDE_2",
        "switches": {
            "ALLOW_NEGATIVE_ENERGIES": False,
            "ALLOW_BIG_RIP": True,
            "ALLOW_DOOM_FACTOR_INSTABILITIES": True,
        },
    },
    "siw": {
        "model": "NonLinear_IDE_2",
        "switches": {
            "ALLOW_NEGATIVE_ENERGIES": True,
            "ALLOW_BIG_RIP": True,
            "ALLOW_DOOM_FACTOR_INSTABILITIES": False,
        },
    },
}


def set_regime(regime):
    spec = REGIMES[regime]

    if spec["switches"] is not None:
        for key, value in spec["switches"].items():
            setattr(UDM, key, value)

    return spec["model"]


def build_context(model):
    models = UDM.Get_model_names([model])

    CONFIG, data = Config.create_config(
        models=models,
        true_values=KC.true_values,
        prior_limits=KC.prior_limits,
        restrictions=UDM.Get_model_restrictions([model]),
        observation=KC.observations,
        nwalkers=KC.nwalkers,
        nsteps=KC.nsteps,
        burn=KC.burn,
        model_name=[model],
        pantheonp_mode=getattr(KC, "pantheonp_mode", "PplusSH0ES"),
        logger=None,
    )

    # Reproduce the Pantheon covariance setup used in MCMC_setup.py.
    comm, rank = init_mpi()

    for tag in ("PantheonP", "PantheonPS"):
        if tag not in data:
            continue

        cov = compute_pantheon_cov(
            data,
            CONFIG[model],
            comm,
            rank,
            os.path.join(K.OBSERVATIONS_BASE, "PantheonP.cov"),
            obs_tag=tag,
        )

        if cov is not None:
            data[tag]["cov"] = cov
            data[tag]["type_data_error"] = np.sqrt(
                np.sum(cov ** 2, axis=1)
            )

    obs_index = 0
    obs = CONFIG[model]["observations"][obs_index]
    obs_types = CONFIG[model]["observation_types"][obs_index]
    param_names = list(CONFIG[model]["parameters"][obs_index])
    prior_map = CONFIG[model]["prior_limits"][obs_index]
    model_func = UDM.Get_model_function(model)

    return (
        CONFIG,
        data,
        obs_index,
        obs,
        obs_types,
        param_names,
        prior_map,
        model_func,
    )
