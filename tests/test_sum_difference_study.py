"""The six EOS cases must have distinct sampled/fixed parameter identities."""
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import User_defined_modules as U
from Kosmulator_main import Config, chain_policy, constants as K
from Kosmulator_main.run_manifest import validate_or_record

@pytest.mark.parametrize('model',['Linear_IDE_2','Linear_IDE_3'])
@pytest.mark.parametrize('uncoupled',[True,False])
@pytest.mark.parametrize('fw,fd',[(False,False),(True,False),(False,True)])
def test_six_cases(model,uncoupled,fw,fd):
    priors=dict(Omega_m=(.1,.5),H_0=(60,90),M_abs=(-20.5,-18))
    if fw: priors['w']=(-2,-.33)
    if fd: priors['w_dm']=(0,.01) # test bounds only, not a study prior
    if not uncoupled: priors['delta']=(-1,1)
    fixed={model:{'delta':0.}} if uncoupled else None
    with patch.object(Config,'load_all_data',return_value={}),patch.object(K,'DERIVE_RD_WITH_MODEL_CLASS',True):
        config,_=Config.create_config(U.Get_model_names([model]),prior_limits=dict(priors,r_d=(130,170)),
            observation=[['PantheonPS','DESI_DR2']],model_name=[model],
            free_w=fw,free_w_dm=fd,fixed_parameters_by_model=fixed)
    c=config[model]
    assert set(c['parameters'][0])==set(priors)
    assert c['ndim'][0]==3+int(fw or fd)+int(not uncoupled)
    expected={}
    if not fw: expected['w']=-1.
    if not fd: expected['w_dm']=0.
    if uncoupled: expected['delta']=0.
    assert c['fixed_parameters']==expected
    assert 'r_d' not in c['parameters'][0]
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'chain.h5';path.touch();chain_policy.write(path,c,0)
        changed=dict(c,fixed_parameters=dict(expected,delta=.001))
        with pytest.raises(ValueError): chain_policy.check(path,changed,0)
