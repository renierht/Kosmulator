"""
Proposal moves of the samplers (Kosmulator.py: emcee_moves, zeus_light_mode).

emcee
-----
"stretch" (default): emcee's own stretch move (Goodman & Weare 2010,
doi:10.2140/camcos.2010.5.65). Exact and robust, but its autocorrelation time grows quickly
with the number of parameters (about 470 - 600 iterations for the 27 parameters of the
Planck baseline).

"de_snooker": differential evolution (80% of the steps; emcee's DEMove, ter Braak 2006,
doi:10.1007/s11222-006-8769-1) mixed with a snooker move (20%) written as in ter Braak &
Vrugt (2008), doi:10.1007/s11222-008-9104-9, with its distances measured in units of the
other walkers' spread of each parameter (13 Oct). Both are exact Metropolis-Hastings moves,
so any mixture of them samples the posterior. DE alone is not offered: on curved posteriors
with prior walls its walkers can get stuck for long stretches; the snooker steps free them.

emcee's own DESnookerMove (3.1.6) is not used: it projects with u = delta / sqrt(|delta|)
instead of the unit vector delta / |delta|, and weights the acceptance with
(d - 1)/2 log(|q - z| / |x - z|) instead of (d - 1) log(...). On a unit Gaussian it returns
a standard deviation of 0.95 (2 parameters) and 0.81 (7 parameters).

zeus
----
zeus keeps its default DifferentialMove. ZEUS_LIGHT_MODE = True: no expansions (stepping
out) after zeus's tuning phase, about half the likelihood calls per step in our tests; zeus's
own docstring: it "works best in target distributions that are approximately Gaussian".
After tuning it is slice sampling with a fixed initial interval and shrinkage, which is
still exact (Neal 2003, doi:10.1214/aos/1056562461).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from Kosmulator_main import constants as K

try:
    from emcee.moves import DEMove, RedBlueMove
except Exception:   # emcee missing: Kosmulator stops elsewhere with a clear message
    DEMove = RedBlueMove = object   # type: ignore[assignment,misc]


class SnookerMove(RedBlueMove):  # type: ignore[misc,valid-type]
    """
    Snooker move of ter Braak & Vrugt (2008) for emcee's red-blue scheme, in scaled units.

    For walker x: pick z, z1, z2 from the three other quarters of the ensemble (they stay
    fixed while x's quarter moves, as in emcee's DESnookerMove). Distances use the metric
    M = diag(1 / sd^2), sd the spread of each parameter over those three quarters; with
    u = (x - z) / |x - z|_M the proposal is x' = x + gamma <u, z1 - z2>_M u, gamma ~ U(1.2, 2.2),
    i.e. x' = x + s (x - z) with s = gamma <x - z, z1 - z2>_M / |x - z|_M^2, accepted with
    min(1, p(x') |1 + s|^(d-1) / p(x)). The swap z1 <-> z2 is equally likely, so the step
    along the line is symmetric, and |1 + s|^(d-1) = (|x' - z| / |x - z|)^(d-1) is the
    Jacobian of moving along a line through z. M is fixed while x's quarter moves, so the
    move stays exact; it is the paper's move in coordinates divided by sd.

    Why scaled: in raw units the line through x and z follows the parameter with the largest
    numbers (H_0 in km/s/Mpc, sd ~ 0.6, against sd ~ 0.007 for tau_reio). When two walkers
    have almost the same H_0 the line points along a small parameter and the step, set by an
    H_0 difference, throws it far out: with the Planck-lite posterior 3.6% of snooker
    proposals put tau_reio beyond the SimAll table (t11, 13 Oct) and 10% more than 10 sd away.
    Scaled, none did; acceptance 0.245 -> 0.278 and tau_max 29.4 -> 28.2 on that posterior.
    """

    def __init__(self, gamma_range: Tuple[float, float] = (1.2, 2.2), **kwargs):
        self.gamma_range = (float(gamma_range[0]), float(gamma_range[1]))
        kwargs["nsplits"] = 4
        super().__init__(**kwargs)

    def get_proposal(self, s, c, random):
        n, ndim = s.shape
        sizes = [len(cc) for cc in c]
        q = np.array(s, dtype=float, copy=True)
        factors = np.zeros(n, dtype=np.float64)
        lo, hi = self.gamma_range
        # metric from the other three quarters (fixed during this move); a parameter that does
        # not vary among them is left out of the distances
        sd = np.std(np.concatenate([np.asarray(cc, dtype=float) for cc in c], axis=0), axis=0)
        metric = np.where(sd > 0.0, 1.0 / np.where(sd > 0.0, sd, 1.0) ** 2, 0.0)
        for i in range(n):
            w = np.array([c[j][random.randint(sizes[j])] for j in range(3)])
            random.shuffle(w)
            z, z1, z2 = w
            delta = s[i] - z
            den = float(np.sum(metric * delta * delta))
            if not np.isfinite(den) or den <= 0.0:
                continue                      # x == z (in the metric): stay (proposal = current point)
            step = random.uniform(lo, hi) * float(np.sum(metric * delta * (z1 - z2))) / den
            q[i] = s[i] + step * delta        # on the line through z and x
            factors[i] = (ndim - 1.0) * np.log(abs(1.0 + step)) if step != -1.0 else -np.inf
        return q, factors


def emcee_moves(name: Optional[str] = None) -> Optional[List[tuple]]:
    """
    The `moves` argument for emcee.EnsembleSampler: None (emcee's stretch move) for
    "stretch", DE 0.8 + snooker 0.2 for "de_snooker" (weights: constants.DE_SNOOKER_WEIGHTS).
    """
    name = (name or K.EMCEE_MOVES).strip().lower()
    if name == "stretch":
        return None
    if name == "de_snooker":
        w_de, w_sn = K.DE_SNOOKER_WEIGHTS
        return [(DEMove(), float(w_de)), (SnookerMove(), float(w_sn))]
    raise ValueError(f"emcee_moves must be one of {K.EMCEE_MOVES_OPTIONS}, not {name!r}")


def describe(engine: str) -> str:
    """One line for the log: the moves the engine uses."""
    if engine == "zeus":
        return "zeus DifferentialMove" + (", light mode" if K.ZEUS_LIGHT_MODE else "")
    if K.EMCEE_MOVES == "de_snooker":
        w_de, w_sn = K.DE_SNOOKER_WEIGHTS
        return f"emcee DE {w_de:g} + snooker {w_sn:g} (ter Braak & Vrugt 2008)"
    return "emcee stretch move"
