"""Display labels and rectangular KDE bounds; no likelihood/statistic changes."""
import re
import numpy as np

SWITCHES = ('ALLOW_NEGATIVE_ENERGIES', 'ALLOW_BIG_RIP', 'ALLOW_DOOM_FACTOR_INSTABILITIES')


def regime_flags(config):
    flags = config.get('_postprocessing_switches')
    if flags is not None:
        return flags
    import User_defined_modules as udm
    return {name: bool(getattr(udm, name)) for name in SWITCHES}


def model_label(model, config=None, flags=None, mathtext=False):
    if model == 'LCDM_v':
        return r'$\Lambda$CDM' if mathtext else 'LCDM'
    if model != 'NonLinear_IDE_2':
        return model
    flags = flags if flags is not None else regime_flags(config or {})
    signature = tuple(bool(flags[name]) for name in SWITCHES)
    return {(True, True, True): 'iwCDM', (False, True, True): '+iwCDM',
            (True, True, False): 'SiwCDM'}.get(signature, model)


def observation_label(key, mathtext=False, extra_names=(), pretty_names=None):
    text = str(key).replace('PantheonP_SH0ES', 'PantheonPS').replace('Pantheon+SH0ES', 'PantheonPS')
    text = re.sub(r'DESI[+_]DR([12])', r'DESI_DR\1', text)
    # Protect dataset-name underscores before treating underscore-separated groups.
    protected = ('DESI_DR1', 'DESI_DR2', 'f_sigma_8', 'CMB_TT', 'CMB_TE', 'CMB_EE', 'CMB_PP', *extra_names)
    stored = {}
    for i, name in enumerate(sorted(set(protected), key=len, reverse=True)):
        if '_' in name:
            token = f'@DATASET{i}@'
            text = text.replace(name, token)
            stored[token] = name
    tokens = [stored.get(t, t) for t in re.split(r'[+_]', text) if t]
    mapping = {'DESI_DR1': 'DESI DR1', 'DESI_DR2': 'DESI DR2',
               'PantheonPS': r'Pantheon$^{+}$ + SH0ES' if mathtext else 'Pantheon+ + SH0ES',
               'PantheonP': r'Pantheon$^{+}$' if mathtext else 'Pantheon+'}
    for name, pair in (pretty_names or {}).items():
        if name not in mapping:
            mapping[name] = pair[0 if mathtext else 1]
    return ' + '.join(mapping.get(t, t) for t in tokens)


def corner_ranges(config, index, model, samples):
    """KDE's axis-aligned prior support, plus verified NonLinear_IDE_2 limits.

    Coupled physicality restrictions are encoded by the retained samples; this
    helper does not claim to implement their full multidimensional boundary.
    It never clips or filters samples.
    """
    names = config['parameters'][index]
    prior = config['prior_limits'][index]
    bounds = {name: [float(prior[name][0]), float(prior[name][1])] for name in names}
    if model == 'NonLinear_IDE_2':
        flags = regime_flags(config)
        if not flags[SWITCHES[0]] and 'delta' in bounds:
            bounds['delta'][0] = max(0., bounds['delta'][0])
        if not flags[SWITCHES[2]] and 'w' in bounds:
            bounds['w'][1] = min(-1., bounds['w'][1])
        if not flags[SWITCHES[1]] and 'w' in bounds:
            bounds['w'][0] = max(-1., bounds['w'][0])
    array = np.asarray(samples)
    if array.ndim != 2 or array.shape[1] != len(names) or not np.all(np.isfinite(array)):
        raise ValueError('Invalid retained samples for corner plot')
    for j, name in enumerate(names):
        low, high = bounds[name]
        if low >= high:
            raise ValueError(f'Empty plotting support for {name}: {low}, {high}')
        if array[:, j].min() < low - 1e-12 or array[:, j].max() > high + 1e-12:
            raise ValueError(f'Retained samples contradict declared plotting support for {name}')
    return bounds
