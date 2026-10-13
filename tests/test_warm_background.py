import unittest
import numpy as np
from scipy.integrate import solve_ivp
from Kosmulator_main import ide_background as I


def rhs(model, p):
    # Independent conservation equations: no analytic solutions or modes.
    w, v = p['w'], p['w_dm']
    d = p.get('delta', 0.)
    def f(x, y):
        dm, de = y
        if model == 1: q=3*(p['delta_dm']*dm+p['delta_de']*de)
        elif model == 2: q=3*d*(dm+de)
        elif model == 3: q=3*d*(dm-de)
        elif model == 4: q=3*d*dm
        elif model == 5: q=3*d*de
        elif model == 6: q=3*d*dm*de/(dm+de)
        elif model == 7: q=3*d*dm*dm/(dm+de)
        elif model == 8: q=3*d*de*de/(dm+de)
        else:
            proxy=((1+3*v)*dm+(1+3*w)*de)/(2*(dm+de))
            q=3*d*proxy*(dm if model == 9 else de)
        # Evolve densities scaled by a^{3(1+v)} to reduce dynamic range.
        return [q, -3*(w-v)*de-q]
    return f


class WarmBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.p=dict(Omega_m=.307,Omega_b=.048,Omega_r=8e-5,w=-1.08,
                    w_dm=.025,delta=.04,delta_dm=.03,delta_de=.02)

    def compare_ode(self, model, p, extent=(-7.,3.)):
        y0=[p['Omega_m']-p['Omega_b'],1-p['Omega_m']-p['Omega_r']]
        for stop in extent:
            xs=np.linspace(0,stop,81)
            sol=solve_ivp(rhs(model,p),(0,stop),y0,t_eval=xs,
                          method='DOP853',rtol=2e-11,atol=2e-13)
            self.assertTrue(sol.success, sol.message)
            ref=sol.y*np.exp(-3*(1+p['w_dm'])*xs)
            got=np.array(I.densities(model,np.exp(xs),p))
            scale=np.maximum(np.max(np.abs(ref),axis=0),1e-20)
            self.assertLess(np.max(np.abs(got-ref)/scale),2e-8,(model,stop))

    def test_ten_models_against_independent_conservation_odes(self):
        for m in range(1,11):
            with self.subTest(model=m):
                p=dict(self.p,delta=-.04 if m==10 else .04)
                self.compare_ode(m,p)
                self.assertAlmostEqual(I.expansion(m,0,p),1.,places=12)

    def test_random_warm_and_cold_points_against_odes(self):
        rng=np.random.default_rng(1013)
        for m in range(1,11):
            for _ in range(5):
                p=dict(self.p,w=rng.uniform(-1.3,-.7),w_dm=rng.uniform(-.08,.08),
                       delta=rng.uniform(.002,.05)*(-1 if m==10 else 1),
                       delta_dm=rng.uniform(.002,.04),delta_de=rng.uniform(.002,.04))
                with self.subTest(model=m,p=p): self.compare_ode(m,p,(-5.,2.))

    def test_positivity_priors_against_density_samples(self):
        for m in range(1,11):
            p=dict(self.p,delta=0. if m==3 else (-.04 if m==10 else .04))
            self.assertTrue(I.domain(m,p))
            self.assertTrue(I.globally_positive(m,p))
            dm,de=I.densities(m,np.geomspace(1e-8,1e8,801),p)
            self.assertTrue(np.all(dm>=0) and np.all(de>=0),m)
            self.assertTrue(np.isfinite(I.expansion(m,[0.,1.,2.],p,allow_negative=False)).all())
        for m in (1,2,3,4,5,7,8,9,10):
            p=dict(self.p,delta=.04 if m==10 else -.04,delta_dm=-.04)
            self.assertFalse(I.globally_positive(m,p),m)
            self.assertTrue(np.isnan(I.expansion(m,[0.,1.],p,allow_negative=False)).all())

    def test_sign_switch_proxy_excludes_baryons_and_radiation(self):
        a=np.geomspace(.1,2.,81)
        p=dict(self.p,Omega_r=.01)
        # Same present dark densities, different baryon/radiation split.
        q=dict(p,Omega_b=.038,Omega_m=.297,Omega_r=.02)
        for m in (9,10):
            np.testing.assert_allclose(I.densities(m,a,p),I.densities(m,a,q),rtol=1e-12)

    def test_uncoupled_and_cold_limits(self):
        a=np.geomspace(1e-14,10,60)
        for m in range(1,11):
            for w,v in [(-1.08,.025),(-1.,0.)]:
                p=dict(self.p,w=w,w_dm=v,delta=0.,delta_dm=0.,delta_de=0.)
                dm,de=I.densities(m,a,p)
                ref=(p['Omega_m']-p['Omega_b'])*a**(-3*(1+v))+(1-p['Omega_m']-p['Omega_r'])*a**(-3*(1+w))
                np.testing.assert_allclose(dm+de,ref,rtol=3e-12)
                np.testing.assert_allclose(dm,(p['Omega_m']-p['Omega_b'])*a**(-3*(1+v)),rtol=3e-12)
                np.testing.assert_allclose(de,(1-p['Omega_m']-p['Omega_r'])*a**(-3*(1+w)),rtol=3e-12)

    def test_near_exceptional_limits(self):
        for eps in (1e-8,1e-12,1e-15):
            for m,d in [(4,1.105+eps),(5,1.105+eps),(6,1.105+eps),
                        (9,-2/3+eps),(10,2/3-eps)]:
                with self.subTest(model=m,eps=eps):
                    self.compare_ode(m,dict(self.p,delta=d),(-.05,.05))
            self.compare_ode(1,dict(self.p,w=-1.,w_dm=0.,delta_dm=.25-eps,delta_de=.25-eps),(-1.,1.))

    def test_exceptional_limits(self):
        for m,updates in [(1,dict(w=-1.,w_dm=0.,delta_dm=.25,delta_de=.25)),
                          (4,dict(delta=1.105)),(5,dict(delta=1.105)),
                          (6,dict(delta=1.105)),(9,dict(delta=-2/3)),
                          (10,dict(delta=2/3)),
                          (9,dict(delta=-2*(-1.105)/(1+3*(-1.08)))),
                          (10,dict(delta=-2*(-1.105)/(1+3*.025)))]:
            with self.subTest(model=m,updates=updates):
                self.compare_ode(m,dict(self.p,**updates),(-.05,.05))

    def test_conditions_independent(self):
        p=dict(self.p,delta=-.1,w=-1.1)
        self.assertTrue(I.domain(9,p))
        self.assertFalse(I.globally_positive(9,p))
        self.assertEqual(I.future_behavior(9,p),'big_rip')
        self.assertEqual(I.future_behavior(10,dict(p,delta=.8)),'finite_scale_singularity')
        self.assertFalse(I.domain(10,dict(p,delta=.8)))
        self.assertTrue(I.globally_positive(3,dict(p,delta=0.)))
        self.assertFalse(I.doom_allowed(9,p))
        self.assertFalse(I.doom_allowed(7,p))  # warm stability unclassified

    def test_big_rip_uses_surviving_mode(self):
        # A negative growing mode is not a Big Rip or a viable expanding future.
        p=dict(self.p,w=-.9,w_dm=.01,delta=1.2)
        self.assertEqual(I.future_behavior(4,p),'negative_future_total')
        # Model 6 switches its future dominant component when w-w_dm+delta>0.
        self.assertEqual(I.future_behavior(6,dict(p,w=-1.1,delta=2.)),'no_big_rip')
        for v in (0.,.025):
            p=dict(self.p,w=-1.,w_dm=v,delta=.04)
            self.assertEqual(I.future_behavior(4,p),'no_big_rip')


if __name__=='__main__': unittest.main()
