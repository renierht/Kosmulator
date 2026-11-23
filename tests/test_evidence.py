import json
import numpy as np
import pytest
from scipy.special import ndtr
from postprocessing.evidence import estimate,temporal_subset,read_completed_chain,compare


def gaussian(n=20000,d=3,seed=99):
    rng=np.random.default_rng(seed);x=rng.normal(size=(n,d))
    return x,-.5*(x*x).sum(axis=1),np.tile([-10.,10.],(d,1))

@pytest.mark.parametrize('d',[1,3,5])
def test_analytic_gaussian(d):
    x,ll,b=gaussian(d=d)
    expected=d*(.5*np.log(2*np.pi)+np.log(ndtr(10)-ndtr(-10))-np.log(20))
    r=estimate(x,ll,b)
    assert abs(r['logZ']['1']-expected)<.08


def test_affine_jacobian_and_prior_volume():
    x,ll,b=gaussian();r=estimate(x,ll,b)['logZ']
    scale=np.array([.003,5.,100.]);offset=np.array([20.,-3.,1000.])
    a=estimate(x*scale+offset,ll,b*scale[:,None]+offset[:,None])['logZ']
    np.testing.assert_allclose(list(r.values()),list(a.values()),atol=1e-10)
    wide=estimate(x,ll,2*b)['logZ']['1']
    assert wide==pytest.approx(r['1']-3*np.log(2))
    shifted=estimate(x,ll-10000,b)['logZ']['1']
    assert shifted==pytest.approx(r['1']-10000)


def test_correlated_gaussian():
    x,ll,b=gaussian();A=np.array([[1,0,0],[.99,.1,0],[2,.2,.3]])
    y=x@A.T;bounds=np.tile([-50.,50.],(3,1))
    true=1.5*np.log(2*np.pi)+np.linalg.slogdet(A)[1]-3*np.log(100)
    assert abs(estimate(y,ll,bounds)['logZ']['1']-true)<.08

@pytest.mark.parametrize('case',['duplicates','singular','nan','sentinel','outside','bounds','alignment'])
def test_fail_closed(case):
    x,ll,b=gaussian(n=500)
    if case=='duplicates':x[1]=x[0]
    if case=='singular':x[:,2]=x[:,1]
    if case=='nan':ll[0]=np.nan
    if case=='sentinel':ll[0]=-5e299
    if case=='outside':x[0,0]=11
    if case=='bounds':b[0]=[1,1]
    if case=='alignment':ll=ll[:-1]
    with pytest.raises(ValueError):estimate(x,ll,b)


def test_temporal_thinning_preserves_likelihood_alignment():
    x=np.arange(100*3*2).reshape(100,3,2);ll=x[:,:,0]*.1
    a,b=temporal_subset(x,ll,5,123)
    np.testing.assert_allclose(a[:,0]*.1,b)
    assert len(a)==60


def test_uniform_boundary_is_flagged():
    x=np.random.default_rng(1).uniform(size=(10000,3))
    r=estimate(x,np.zeros(len(x)),np.tile([0.,1.],(3,1)))
    assert r['neighbour_balls_crossing_box_fraction']['1']>.01
    assert abs(r['logZ']['1'])<.2 # known Z=1; finite-sample boundary bias is not silently fixed


def fixture_chain(tmp_path):
    import h5py
    p=tmp_path/'chain.h5'
    x,ll,b=gaussian(n=400,d=1)
    with h5py.File(p,'w') as f:
        g=f.create_group('mcmc');g.attrs['iteration']=100
        # Preallocated tail deliberately contains garbage and MUST NOT be read.
        g.create_dataset('chain',data=np.concatenate([x.reshape(100,4,1),np.full((100,4,1),np.nan)]))
        g.create_dataset('blobs',data=np.concatenate([ll.reshape(100,4),np.full((100,4),np.nan)]))
    Path=__import__('pathlib').Path
    Path(str(p)+'.metadata.json').write_text(json.dumps({'model':'Test','parameters':['x'],'fixed_parameters':{}}))
    Path(str(p)+'.convergence.json').write_text(json.dumps({'burn':10,'status':{'converged':True,'stopped_at':100,'tau_max':2}}))
    return dict(chain=str(p),model='Test',parameters=['x'],priors={'x':[-10,10]},burn=10,prior_policy='normalized_uniform_box_zero_likelihood_outside_domain')


def test_hdf_committed_iterations_only(tmp_path):
    spec=fixture_chain(tmp_path);x,ll,*_=read_completed_chain(spec)
    assert x.shape==(90,4,1) and ll.shape==(90,4)

@pytest.mark.parametrize('key,value',[('burn',11),('parameters',['derived']),('prior_policy','conditioned'),('model','Wrong')])
def test_hdf_identity_guards(tmp_path,key,value):
    spec=fixture_chain(tmp_path);spec[key]=value
    with pytest.raises(ValueError):read_completed_chain(spec)


def test_comparison_identity_guards():
    audit=dict(dataset_hashes={'data':'abc'},likelihood_convention='same',class_settings={'a':1},zero_domain_convention='same')
    a=dict(label='A',audit=audit,logZ_nominal=0.,logZ_sensitivity_envelope=[-.1,.1])
    b=dict(label='B',audit=dict(audit),logZ_nominal=2.,logZ_sensitivity_envelope=[1.9,2.1])
    assert compare([a,b],'A')[1]['delta_logZ']==2
    b['audit']['dataset_hashes']={'data':'different'}
    with pytest.raises(ValueError):compare([a,b],'A')


def test_nonconstant_saved_prior_refused(tmp_path):
    import h5py
    spec=fixture_chain(tmp_path)
    with h5py.File(spec['chain'],'a') as f:
        f['mcmc'].create_dataset('log_prob',data=f['mcmc/blobs'][:]+.1)
    with pytest.raises(ValueError,match='nonconstant prior'):read_completed_chain(spec)


def test_missing_likelihood_refused(tmp_path):
    import h5py
    spec=fixture_chain(tmp_path)
    with h5py.File(spec['chain'],'a') as f:del f['mcmc/blobs']
    with pytest.raises(ValueError,match='missing saved likelihood'):read_completed_chain(spec)


def test_stale_completion_refused(tmp_path):
    import h5py
    spec=fixture_chain(tmp_path)
    with h5py.File(spec['chain'],'a') as f:f['mcmc'].attrs['iteration']=101
    with pytest.raises(ValueError,match='completion record'):read_completed_chain(spec)
