"""Persist parameter meaning independently of chain dimension and engine."""
import json
from pathlib import Path
from Kosmulator_main.ide_background import SELECTORS


def policy(config, index):
    import User_defined_modules as UDM
    return dict(model=config.get('model_name'),
                parameters=list(config['parameters'][index]),
                fixed_parameters=config.get('fixed_parameters', {}),
                eos_policy=config.get('eos_policy', {}),
                background_version='warm-ide-1',
                switches=config.get('_postprocessing_switches') or {
                    key: bool(getattr(UDM,key)) for key in (
                        'ALLOW_NEGATIVE_ENERGIES','ALLOW_BIG_RIP','ALLOW_DOOM_FACTOR_INSTABILITIES')})


def check(path, config, index):
    """Read-only guard. Legacy IDE chains retain free w / fixed cold DM."""
    path=Path(path)
    sidecar=Path(str(path)+'.metadata.json')
    expected=policy(config,index)
    if sidecar.exists():
        saved=json.loads(sidecar.read_text())
        if saved != expected:
            raise ValueError(f'Chain parameter/physics policy mismatch: {sidecar}; use a new output path')
    elif path.exists() and config.get('model_name') in SELECTORS:
        names=config['parameters'][index]
        fixed=config.get('fixed_parameters',{})
        if ('w' not in names or 'w_dm' in names or fixed.get('w_dm',0.) != 0.
                or SELECTORS[config['model_name']] > 8):
            raise ValueError('Legacy IDE chain has no EOS metadata; only free w / fixed w_dm=0 is compatible')


def write(path, config, index):
    target=Path(str(path)+'.metadata.json')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(policy(config,index),indent=2)+'\n')
