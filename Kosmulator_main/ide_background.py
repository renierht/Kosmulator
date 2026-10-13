"""Ten constant-EOS IDE backgrounds, in present critical-density units.

Q>0 transfers DE to DM. The equation of state in the interaction proxy for
models 9/10 is the DARK-SECTOR proxy, excluding baryons and radiation.
Density solutions and optional physical priors are deliberately separate.
"""
import numpy as np

MODEL_NAMES = tuple([f"Linear_IDE_{i}" for i in range(1, 6)] +
                    [f"NonLinear_IDE_{i}" for i in range(1, 4)] +
                    ["SignSwitch_IDE_1", "SignSwitch_IDE_2"])
SELECTORS = {name: i + 1 for i, name in enumerate(MODEL_NAMES)}


def parameters(p):
    w, v = float(p.get("w", -1.)), float(p.get("w_dm", 0.))
    dm = float(p["Omega_m"]) - float(p.get("Omega_b", .048))
    de = 1. - float(p["Omega_m"]) - float(p.get("Omega_r", 0.))
    if not np.isfinite([w, v, dm, de]).all() or min(dm, de) <= 0:
        raise ValueError("IDE requires finite EOS and positive present dark densities")
    return w, v, dm, de


def couplings(model, p):
    d = float(p.get("delta", 0.))
    return {1: (float(p.get("delta_dm", 0.)), float(p.get("delta_de", 0.))),
            2: (d, d), 3: (d, -d), 4: (d, 0.), 5: (0., d)}[model]


def _positive(base):
    if not np.isfinite(base).all() or np.any(base <= 0.):
        raise ValueError("IDE real-power base or dark-sector total is nonpositive")
    return base


def _exprel(x):
    """expm1(x)/x, including its continuous value at zero."""
    x = np.asarray(x, dtype=float)
    return np.divide(np.expm1(x), x, out=np.ones_like(x), where=x != 0.)


def _switch_densities(model, x, w, v, d, dm0, de0):
    # Solve the linear density-ratio equation; integrate its reciprocal exactly.
    # This also covers delta=+/-2/3 and the logarithmic-ratio k=0 branches.
    u, A, B = w-v, 1+3*v, 1+3*w
    if model == 9:
        r0, k, b = de0/dm0, -1.5*(2*u+d*B), -1.5*d*A
    else:
        r0, k, b = dm0/de0, 1.5*(2*u+d*A), 1.5*d*B
    if k == 0.:
        r = r0 + b*x
        _positive(1+r)
        integral = x/(1+r0) if b == 0 else np.log1p(b*x/(1+r0))/b
    else:
        r = r0*np.exp(k*x) + b*np.expm1(k*x)/k
        _positive(1+r)
        D = 0. if (model == 9 and d == -2/3) or (model == 10 and d == 2/3) else 1-b/k
        if D == 0.:
            integral = -np.expm1(-k*x)/(k*(1+r0))
        else:
            # Avoid subtraction of nearly equal logarithms near delta=+/-2/3.
            q = D*np.expm1(-k*x)/(1+r0)
            integral = -np.log1p(q)/(k*D)
    if model == 9:
        dm = dm0*np.exp((-3*(1+v)+1.5*d*B)*x-4.5*d*u*integral)
        return dm, dm*r
    de = de0*np.exp((-3*(1+w)-1.5*d*A)*x-4.5*d*u*integral)
    return de*r, de


def densities(model, a, p):
    """Return (rho_dm, rho_de). Raise ValueError outside the real branch.

    Uses exact matrix modes for 1--5, including repeated eigenvalues; exact
    constant-ratio limit for 6; ratio-integral limits for 9/10.
    No positivity, future, or perturbative prior is imposed by this function.
    """
    if model not in range(1, 11):
        raise ValueError("IDE selector must be 1..10")
    a = np.asarray(a, dtype=float)
    _positive(a)
    x = np.log(a)
    w, v, dm0, de0 = parameters(p)
    u, d, r = w-v, float(p.get("delta", 0.)), dm0/de0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if model <= 5:
            ddm, dde = couplings(model, p)
            # d rho / d ln(a) = M rho. exp(Mx) uses the two real modes.
            m00, m01 = -3*(1+v-ddm), 3*dde
            m10, m11 = -3*ddm, -3*(1+w+dde)
            t = (m00+m11)/2
            disc = (ddm+dde+u)**2-4*ddm*dde
            if disc < 0 or not np.isfinite(disc):
                raise ValueError("Non-real IDE eigenvalues")
            s = 1.5*np.sqrt(disc)
            if ddm == dde == 0.:
                dm, de = dm0*np.exp(-3*(1+v)*x), de0*np.exp(-3*(1+w)*x)
            elif dde == 0.:
                dm = dm0*np.exp(-3*(1+v-ddm)*x)
                de = np.exp(-3*(1+w)*x)*(de0-3*ddm*dm0*x*_exprel(3*(u+ddm)*x))
            elif ddm == 0.:
                de = de0*np.exp(-3*(1+w+dde)*x)
                dm = np.exp(-3*(1+v)*x)*(dm0+3*dde*de0*x*_exprel(-3*(u+dde)*x))
            elif s == 0.:
                f, g = np.exp(t*x), x*np.exp(t*x)
                dm = f*dm0 + g*((m00-t)*dm0+m01*de0)
                de = f*de0 + g*(m10*dm0+(m11-t)*de0)
            else:
                plus, minus = np.exp((t+s)*x), np.exp((t-s)*x)
                dm_plus = .5*(dm0+((ddm+dde+u)*dm0+2*dde*de0)/np.sqrt(disc))
                de_plus = .5*(de0+(-2*ddm*dm0-(ddm+dde+u)*de0)/np.sqrt(disc))
                dm = dm_plus*plus+(dm0-dm_plus)*minus
                de = de_plus*plus+(de0-de_plus)*minus
                # Near a repeated eigenvalue the matrix limit avoids large
                # cancelling mode coefficients. No numerical ODE is used here.
                small = np.abs(s*x) < 1e-3
                f, g = np.exp(t*x)*np.cosh(s*x), np.exp(t*x)*x*_exprel(2*s*x)*np.exp(-s*x)
                dm = np.where(small, f*dm0+g*((m00-t)*dm0+m01*de0), dm)
                de = np.where(small, f*de0+g*(m10*dm0+(m11-t)*de0), de)
        elif model == 6:
            k = u+d
            if k == 0.:
                factor = np.exp(-3*(1+(v*r+w)/(1+r))*x)
                dm, de = dm0*factor, de0*factor
            else:
                y = 3*k*x
                logbase = np.where(y < 50, np.log1p(r/(1+r)*np.expm1(y)),
                                   np.logaddexp(0., np.log(r)+y)-np.log1p(r))
                dm = dm0*np.exp(-3*(1+v-d)*x-d/k*logbase)
                de = de0*np.exp(-3*(1+w)*x-d/k*logbase)
        elif model in (7, 8):
            upper = -u/r if model == 7 else -u*r
            if u >= 0 or d <= u or d > upper:
                raise ValueError("Outside Model 7/8 sufficient real branch")
            if model == 7:
                A = (u+d*r)*np.exp(-3*u*x)
                base = _positive((A+r*(u-d))/(u*(1+r)))
                dm = dm0*np.exp(-3*(1+v-u*d/(u-d))*x)*base**(d/(u-d))
                de = dm/r*(A-d*r)/u
            else:
                A = (u*r+d)*np.exp(3*u*x)
                base = _positive((A+u-d)/(u*(1+r)))
                de = de0*np.exp(-3*(1+w+u*d/(u-d))*x)*base**(d/(u-d))
                dm = de*r*(A-d)/(u*r)
        else:
            dm, de = _switch_densities(model, x, w, v, d, dm0, de0)
    if not np.isfinite(dm).all() or not np.isfinite(de).all():
        raise ValueError("Non-finite IDE densities")
    if model >= 6 and np.any(dm+de == 0.):
        raise ValueError("Undefined interaction at zero dark-sector total")
    return dm, de


def domain(model, p):
    """Global real-branch conditions, independent of component positivity."""
    w, v, dm, de = parameters(p)
    u, d, r = w-v, float(p.get("delta", 0.)), dm/de
    if model <= 5:
        ddm, dde = couplings(model, p)
        return (ddm+dde+u)**2 >= 4*ddm*dde
    if model == 6:
        return True
    if model in (7, 8):
        return u < 0 and u < d <= (-u/r if model == 7 else -u*r)
    proxy = (1+3*v)*r+1+3*w
    if not (w < -1/3 and v > -1/3 and proxy < 0):
        return False
    return (d <= -2*u/proxy or d >= -2/3) if model == 9 else (-2*u*r/proxy <= d <= 2/3)


def globally_positive(model, p):
    w, v, dm, de = parameters(p)
    u, d, r = w-v, float(p.get("delta", 0.)), dm/de
    if model <= 5:
        ddm, dde = couplings(model, p)
        if ddm == dde == 0:  # Uncoupled limit, including Model 3.
            return True
    if u >= 0:
        return model == 6
    if model == 1:
        return min(ddm, dde) >= 0 and ddm*r+dde <= -u*r/(1+r)
    if model == 2:
        return 0 <= d <= -u*r/(1+r)**2
    if model == 3:
        return d == 0
    if model == 4:
        return 0 <= d <= -u/(1+r)
    if model == 5:
        return 0 <= d <= -u*r/(1+r)
    if model == 6:
        return True
    if model in (7, 8):
        return 0 <= d <= (-u/r if model == 7 else -u*r)
    proxy = (1+3*v)*r+1+3*w
    return (d <= -2*u/proxy or d >= 0) if model == 9 else (-2*u*r/proxy <= d <= 0)


def future_behavior(model, p):
    """Asymptotic classification, WITHOUT a global positivity/reality prior.

    9/10 use the supplied addendum under its stated proxy assumptions.
    1--8 use the actual surviving density modes (zero coefficients matter).
    A negative leading total is reported separately, never as a Big Rip.
    """
    w, v, dm, de = parameters(p)
    u, d, r = w-v, float(p.get("delta", 0.)), dm/de
    if model >= 9:
        if not (w < -1/3 and v > -1/3 and (1+3*v)*r+1+3*w < 0):
            return "outside_assumptions"
        if model == 9:
            return "big_rip" if w < -1 and d > -1-v else "no_big_rip"
        if d > 2/3:
            return "finite_scale_singularity"
        return "no_big_rip" if d <= 1+w else "big_rip"
    if model <= 5:
        ddm, dde = couplings(model, p)
        disc = (ddm+dde+u)**2-4*ddm*dde
        if disc < 0:
            return "outside_assumptions"
        m00, m11 = -3*(1+v-ddm), -3*(1+w+dde)
        t, s = (m00+m11)/2, 1.5*np.sqrt(disc)
        total, deriv = dm+de, -3*((1+v)*dm+(1+w)*de)
        if s == 0:
            coeff, power = deriv-t*total, t
            if coeff == 0: coeff = total
        else:
            low, high = sorted((m00, m11)) if ddm*dde == 0. else (t-s, t+s)
            coeff, power = (deriv-low*total)/(high-low), high
            if coeff == 0.:
                coeff, power = total-coeff, low
        if coeff < 0:
            return "negative_future_total"
    elif model == 6:
        power = -3*(1+(w if u+d < 0 else v))
        if u+d == 0: power = -3*(1+(v*r+w)/(1+r))
    elif model == 7:
        if u >= 0 or d <= u or d > -u/r: return "outside_assumptions"
        power = -3*(1+w) if u+d*r != 0 else -3*(1+v-u*d/(u-d))
    else:
        if u >= 0 or d <= u or d > -u*r: return "outside_assumptions"
        power = -3*(1+w+u*d/(u-d))
    return "big_rip" if power > 0. else "no_big_rip"


def doom_allowed(model, p):
    """Existing sufficient cold-DM criteria; warm perturbations unclassified.

    A background extension is not a derivation of warm IDE perturbations.
    Uncoupled points have Q=0 and need no interacting doom-factor test.
    """
    w, v, _, _ = parameters(p)
    d = float(p.get("delta", 0.))
    if model <= 5:
        ddm, dde = couplings(model, p)
        if ddm == dde == 0: return True
    elif d == 0: return True
    if model >= 9 or v != 0: return False
    if model == 1: return dde > 0 and w < -1
    if model in (5, 6, 8): return d*(1+w) < 0
    return w < -1


def expansion(model, z, p, allow_negative=True, allow_big_rip=True, allow_doom=True):
    z = np.asarray(z, dtype=float)
    try:
        if np.any(z <= -1) or not np.isfinite(z).all() or not domain(model, p):
            raise ValueError("Invalid redshift or IDE domain")
        if not allow_negative and not globally_positive(model, p):
            raise ValueError("Global positivity prior")
        if not allow_big_rip and future_behavior(model, p) != "no_big_rip":
            raise ValueError("Future regularity prior")
        if not allow_doom and not doom_allowed(model, p):
            raise ValueError("Doom criterion unsupported or failed")
        dm, de = densities(model, 1/(1+z), p)
        e2 = dm+de+float(p.get("Omega_b", .048))*(1+z)**3+float(p.get("Omega_r", 0.))*(1+z)**4
        _positive(e2)
        result = np.sqrt(e2)
    except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
        result = np.full_like(z, np.nan)
    return result.item() if result.ndim == 0 else result
