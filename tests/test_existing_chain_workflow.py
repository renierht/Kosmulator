"""Tests for the optional production load-only workflow and paper configuration."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def extracted(path,name,namespace):
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]

class ExistingChainTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/'chain.h5';self.path.touch()
        self.array=np.arange(30,dtype=float).reshape(5,2,3);self.reads=[]
        array,reads=self.array,self.reads
        class Backend:
            iteration=5
            def __init__(self,path,read_only=False):
                if not read_only:raise AssertionError('Writable backend')
                reads.append(('open',path))
            def get_chain(self,**kw):
                reads.append(('read',kw));return array[kw['discard']::kw['thin']].reshape(-1,3)
        self.km=types.ModuleType('Kosmulator_main.Kosmulator_MCMC')
        self.km.run_mcmc=lambda **kw:(_ for _ in ()).throw(AssertionError('Sampling attempted'))
        parent=types.ModuleType('Kosmulator_main');parent.Kosmulator_MCMC=self.km
        self.modules=patch.dict('sys.modules',{'Kosmulator_main':parent,'Kosmulator_main.Kosmulator_MCMC':self.km})
        self.modules.start();self.addCleanup(self.modules.stop)
        self.function=extracted(ROOT/'Kosmulator_main/utils.py','load_or_run_chain',{
            'Dict':dict,'Any':object,'os':__import__('os'),'np':np,'h5py':object(),
            'emcee':types.SimpleNamespace(backends=types.SimpleNamespace(HDFBackend=Backend))})
        self.kw={'output_dir':self.directory.name,'chain_file':'chain.h5','overwrite':False,
                 'CONFIG_model':{'burn':2,'parameters':[['Omega_m','H_0','M_abs']]},'data':{},
                 'MODEL_func':None,'convergence':.01,'parallel':False,'pool':None,'vectorised':True,'load_only':True}
    def test_load_only_applies_burn_once_and_uses_read_only_backend(self):
        result=self.function(**self.kw)
        np.testing.assert_array_equal(result,self.array[2:].reshape(-1,3))
        self.assertEqual(self.reads[-1],('read',{'discard':2,'thin':1,'flat':True}))
    def test_missing_chain_never_starts_sampler(self):
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):self.function(**self.kw)
        self.assertEqual(self.reads,[])
    def test_resume_and_overwrite_are_rejected_before_open(self):
        for field in ('overwrite','resumeChains'):
            with self.subTest(field=field),self.assertRaises(ValueError):self.function(**dict(self.kw,**{field:True}))
        self.assertEqual(self.reads,[])
    def test_empty_retention_negative_burn_and_wrong_dimension_are_rejected(self):
        for burn,names in ((5,['x','y','z']),(-1,['x','y','z']),(2,['x','y'])):
            with self.subTest(burn=burn),self.assertRaises(ValueError):
                self.function(**dict(self.kw,CONFIG_model={'burn':burn,'parameters':[names]}))
    def test_nonfinite_retained_parameters_are_rejected(self):
        self.array[3,0,1]=np.nan
        with self.assertRaises(ValueError):self.function(**self.kw)

class OverlayTests(unittest.TestCase):
    def test_growth_point_overlay_is_optional(self):
        tree=ast.parse((ROOT/'Plots/Plots.py').read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='obs_type not in SNE_TYPES')
        code=compile(ast.Module(body=[node],type_ignores=[]),'point_overlay','exec')
        for settings,expected in (({},0),({'overlay_model_at_data_points':True},1)):
            calls=[]
            namespace={'obs_type':'f','SNE_TYPES':('PantheonPS',),'PLOT_SETTINGS':settings,'np':np,
                'x_dat':[.2,.1],'y_mod_pts':[2,1],'MODEL_COLOR':'red','z_obs':1,'Z_BAND':10,
                'ax':types.SimpleNamespace(plot=lambda *a,**kw:calls.append((a,kw)))}
            exec(code,namespace);self.assertEqual(len(calls),expected)
            if calls:np.testing.assert_array_equal(calls[0][0][0],[.1,.2])

class PaperConfigurationTests(unittest.TestCase):
    def setUp(self):
        import json
        self.config=json.loads((ROOT/'reproducibility/SAIP2026_IDE/configs/paper.json').read_text())
        path=ROOT/'reproducibility/SAIP2026_IDE/scripts/reproduce.py'
        spec=importlib.util.spec_from_file_location('paper_workflow_test',path)
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
    def test_regime_selection_and_model_derived_rd_are_explicit(self):
        for regime,flags in self.config['regime_switches'].items():
            kc=types.SimpleNamespace(prior_limits={});udm=types.SimpleNamespace();K=types.SimpleNamespace()
            self.module.configure(kc,udm,K,self.config,regime)
            self.assertEqual(kc.model_names,['LCDM_v','NonLinear_IDE_2']);self.assertEqual(kc.burn,1000)
            self.assertEqual(kc.observations,[['PantheonPS','DESI_DR2']]);self.assertTrue(K.DERIVE_RD_WITH_MODEL_CLASS)
            self.assertEqual([udm.ALLOW_NEGATIVE_ENERGIES,udm.ALLOW_BIG_RIP,udm.ALLOW_DOOM_FACTOR_INSTABILITIES],flags)
        self.assertIn('FINAL_8500',self.config['cases']['iw']['chain'])
    def test_class_rebuild_guard_restores_functions_after_failure(self):
        calls=[]
        def original(model,**kw):calls.append(kw);return True
        run=lambda:None;cr=types.SimpleNamespace(ensure_class_ready=original,run_model=run)
        with self.assertRaises(RuntimeError):
            with self.module.forbid_class_rebuild(cr):
                cr.ensure_class_ready('LCDM_v',no_rebuild=False);cr.run_model()
        self.assertEqual(calls,[{'no_rebuild':True,'force':False}])
        self.assertIs(cr.ensure_class_ready,original);self.assertIs(cr.run_model,run)
    def test_archive_path_traversal_is_rejected(self):
        import copy
        c=copy.deepcopy(self.config);c['cases']['lcdm']['chain']='../chain.h5'
        with self.assertRaises(RuntimeError):self.module.chain_paths(c,Path('/tmp'),'iw')

if __name__=='__main__':unittest.main()

class SetupPreflightTests(unittest.TestCase):
    def test_existing_mode_rejects_unsafe_execution_before_configuration(self):
        namespace={'List':list,'Dict':dict,'Any':object,'Optional':__import__('typing').Optional,
                   'time':__import__('time'),'os':__import__('os')}
        function=extracted(ROOT/'Kosmulator_main/MCMC_setup.py','main',namespace)
        for options,paths,groups in (
            ({'overwrite':True}, {'LCDM_v':'unused'},[['PantheonPS']]),
            ({'resume':True}, {'LCDM_v':'unused'},[['PantheonPS']]),
            ({'num_cores':2}, {'LCDM_v':'unused'},[['PantheonPS']]),
            ({'use_mpi':True}, {'LCDM_v':'unused'},[['PantheonPS']]),
            ({}, {},[['PantheonPS']]),
            ({}, {'LCDM_v':'unused'},[['PantheonPS'],['DESI_DR2']]),
        ):
            namespace['parse_cli_args']=lambda options=options:types.SimpleNamespace(**options)
            with self.subTest(options=options),self.assertRaises(ValueError):
                function(['LCDM_v'],groups,'LCDM_v',{}, {},120,8500,1000,.01,existing_chain_paths=paths)

class EntryPointTests(unittest.TestCase):
    def test_normal_main_forwards_optional_workflow_without_changing_defaults(self):
        captures=[]
        parent=types.ModuleType('Kosmulator_main');utils=types.ModuleType('Kosmulator_main.utils')
        utils.print_init_banner=lambda *a:None;parent.utils=utils
        with patch.dict('sys.modules',{'Kosmulator_main':parent,'Kosmulator_main.utils':utils}):
            ns={'model_names':['LCDM_v'],'observations':[['PantheonPS']],'true_model':'LCDM_v',
                '_ensure_true_model_first':lambda models,reference:models,'logging':__import__('logging'),
                'nwalkers':120,'nsteps':8500,'burn':1000,'convergence':.01,'prior_limits':{},'true_values':{},
                'run_mcmc':lambda **kw:captures.append(kw)}
            function=extracted(ROOT/'Kosmulator.py','main',ns)
            function();function(workflow_options={'existing_chain_paths':{'LCDM_v':'chain.h5'}})
        self.assertNotIn('existing_chain_paths',captures[0])
        self.assertEqual(captures[1]['existing_chain_paths'],{'LCDM_v':'chain.h5'})
