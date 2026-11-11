"""Native plot smoke using real data and declared illustrative (not fitted) parameters."""
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
from Kosmulator_main import Config
import User_defined_modules as U
from Plots import Plots
names=['PantheonPS','PantheonP','DESY5','DES_Dovekie','Union3']
configs,data=Config.create_config(U.Get_model_names(['LCDM_v']),model_name=['LCDM_v'],observation=[[n] for n in names],prior_limits=dict(Omega_m=(.1,.5),H_0=(60,90),M_abs=(-20.5,-18)))
params=dict(Omega_m=[.3,.3,.3],H_0=[70.,70.,70.],M_abs=[-19.3,-19.3,-19.3])
values={'LCDM_v':{name:params for name in names}}
folder=Path('validation/native_sn_plot_review'); folder.mkdir(exist_ok=True)
def save(fig,model,key,kind,settings):
    fig.suptitle('Validation only: illustrative LCDM parameters, not posterior fit',fontsize=9)
    fig.savefig(folder/(key+'.png'),dpi=120)
with patch.object(Plots,'save_figure',save):
    Plots.best_fit_plots(values,configs,data,dict(latex_enabled=False))
print('Saved native validation plots to',folder)
