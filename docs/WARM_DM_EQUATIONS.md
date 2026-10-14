# Ten constant-EOS interacting dark-sector backgrounds

This is the equation reference for `Kosmulator_main/ide_background.py` and
`Class/IDE_background/include/ide_densities.h`. It documents the implemented
branches and filters, including exceptional limits. It does not claim a
perturbation-complete warm-dark-matter model. Here “warm” means a constant
background EOS; no particle distribution or free-streaming prescription is added.

## Conventions and expansion

Let $v=w_{\rm dm}$, $w=w_{\rm de}$ (the code parameter `w`),
$u=w-v$, $x=\ln a$, and $a=(1+z)^{-1}$. Write $c=\rho_{\rm dm}$,
$e=\rho_{\rm de}$, $c_0=c(1)>0$, $e_0=e(1)>0$, and $r=c_0/e_0$.
Both EOS parameters are constant. The sign convention is

```math
\dot c+3H(1+v)c=Q,\qquad \dot e+3H(1+w)e=-Q.
```

Thus $Q>0$ transfers energy from DE to DM. All density formulas below can
be expressed in present critical-density units, as in the Python implementation:

```math
c_0=\Omega_m-\Omega_b,\qquad e_0=1-\Omega_m-\Omega_r,
\qquad E^2(a)=c(a)+e(a)+\Omega_b a^{-3}+\Omega_r a^{-4}.
```

The low-redshift Python default is $\Omega_r=0$. CLASS retains its calibrated
photons and neutrinos, including massive-neutrino evolution; comparisons must
match the radiation/density normalization rather than treating massive neutrinos
as a constant $a^{-4}$ component at every epoch.

| Model | Registry name | $Q/(3H)$ |
|---:|---|---|
| 1 | `Linear_IDE_1` | $\delta_{\rm dm}c+\delta_{\rm de}e$ |
| 2 | `Linear_IDE_2` | $\delta(c+e)$ |
| 3 | `Linear_IDE_3` | $\delta(c-e)$ |
| 4 | `Linear_IDE_4` | $\delta c$ |
| 5 | `Linear_IDE_5` | $\delta e$ |
| 6 | `NonLinear_IDE_1` | $\delta ce/(c+e)$ |
| 7 | `NonLinear_IDE_2` | $\delta c^2/(c+e)$ |
| 8 | `NonLinear_IDE_3` | $\delta e^2/(c+e)$ |
| 9 | `SignSwitch_IDE_1` | $\delta q_D c$ |
| 10 | `SignSwitch_IDE_2` | $\delta q_D e$ |

The sign-switch proxy is

```math
q_D=\frac{(1+3v)c+(1+3w)e}{2(c+e)}.
```

It excludes baryons and radiation and is not the full cosmological deceleration
parameter. Zero coupling gives $c=c_0a^{-3(1+v)}$, $e=e_0a^{-3(1+w)}$.
Setting $v=0$ recovers cold DM.

## Models 1–5: exact matrix solution

Set $(d_c,d_e)$ to $(\delta_{\rm dm},\delta_{\rm de})$, $(\delta,\delta)$,
$(\delta,-\delta)$, $(\delta,0)$ or $(0,\delta)$ for Models 1–5 respectively.
Then

```math
\frac{d}{dx}\begin{pmatrix}c\\e\end{pmatrix}
=M\begin{pmatrix}c\\e\end{pmatrix},\qquad
M=3\begin{pmatrix}-(1+v-d_c)&d_e\\-d_c&-(1+w+d_e)\end{pmatrix}.
```

Define $t=\operatorname{tr}(M)/2$,
$\Delta^2=(d_c+d_e+u)^2-4d_cd_e$ and $s=3\sqrt{\Delta^2}/2$.
The selected branch has $\Delta^2\ge0$. With $y_0=(c_0,e_0)^T$,

```math
y(x)=e^{tx}\left[\cosh(sx)I+\frac{\sinh(sx)}{s}(M-tI)\right]y_0.
```

At $s=0$, replace $\sinh(sx)/s$ by $x$. This includes the repeated-eigenvalue
logarithmic solution. Complex eigenvalue branches are not sampled even where
an alternative real oscillatory representation could be written.

For explicit triangular limits, define $\operatorname{exprel}(z)=(e^z-1)/z$,
with $\operatorname{exprel}(0)=1$:

```math
\begin{aligned}
 d_e=0:\quad
 c&=c_0e^{-3(1+v-d_c)x},\\
 e&=e^{-3(1+w)x}\left[e_0-3d_cc_0x\operatorname{exprel}(3(u+d_c)x)\right];\\[2pt]
 d_c=0:\quad
 e&=e_0e^{-3(1+w+d_e)x},\\
 c&=e^{-3(1+v)x}\left[c_0+3d_ee_0x\operatorname{exprel}(-3(u+d_e)x)\right].
\end{aligned}
```

These forms include $u+d_c=0$ and $u+d_e=0$ without division by zero.

## Model 6

For $k=u+\delta\ne0$, let $B=(1+ra^{3k})/(1+r)>0$. Then

```math
c=c_0a^{-3(1+v-\delta)}B^{-\delta/k},\qquad
 e=e_0a^{-3(1+w)}B^{-\delta/k}.
```

At $k=0$ the ratio is constant and

```math
(c,e)=(c_0,e_0)a^{-3[1+(vr+w)/(1+r)]}.
```

## Models 7 and 8

These formulas are used only on the sufficient global real branches stated
below; the implementation does not extend them through $u=\delta$.

For Model 7 define $A_7=(u+\delta r)a^{-3u}$ and
$B_7=[A_7+r(u-\delta)]/[u(1+r)]$. Then

```math
c=c_0a^{-3[1+v-u\delta/(u-\delta)]}B_7^{\delta/(u-\delta)},
\qquad e=\frac{c}{r}\frac{A_7-\delta r}{u}.
```

For Model 8 define $A_8=(ur+\delta)a^{3u}$ and
$B_8=[A_8+u-\delta]/[u(1+r)]$. Then

```math
e=e_0a^{-3[1+w+u\delta/(u-\delta)]}B_8^{\delta/(u-\delta)},
\qquad c=er\frac{A_8-\delta}{ur}.
```

The real-power bases $B_7,B_8$ must be strictly positive.

## Models 9 and 10: ratio-integral solution

Let $A=1+3v$, $B=1+3w$. Define $R$, $R_0$, $k$ and $b$ by

| Model | $R$ | $R_0$ | $k$ | $b$ |
|---:|---|---|---|---|
| 9 | $e/c$ | $1/r$ | $-3(2u+\delta B)/2$ | $-3\delta A/2$ |
| 10 | $c/e$ | $r$ | $3(2u+\delta A)/2$ | $3\delta B/2$ |

The exact ratio equation is $R'=kR+b$, giving

```math
R(x)=R_0e^{kx}+b\frac{e^{kx}-1}{k}\quad(k\ne0),
\qquad R(x)=R_0+bx\quad(k=0).
```

Let $J(x)=\int_0^x[1+R(\xi)]^{-1}d\xi$. For $k\ne0$, put $D=1-b/k$:

```math
J(x)=\begin{cases}
-\dfrac{\ln\left[1+D(e^{-kx}-1)/(1+R_0)\right]}{kD},&D\ne0,\\[6pt]
\dfrac{1-e^{-kx}}{k(1+R_0)},&D=0.
\end{cases}
```

For $k=0$,

```math
J(x)=\begin{cases}
\ln[1+bx/(1+R_0)]/b,&b\ne0,\\
x/(1+R_0),&b=0.
\end{cases}
```

The densities are

```math
\begin{aligned}
\text{Model 9:}\quad
 c&=c_0\exp\{[-3(1+v)+\tfrac32\delta B]x-\tfrac92\delta uJ(x)\},&e&=cR;\\
\text{Model 10:}\quad
 e&=e_0\exp\{[-3(1+w)-\tfrac32\delta A]x-\tfrac92\delta uJ(x)\},&c&=eR.
\end{aligned}
```

Require $1+R>0$ and valid logarithm arguments throughout the evaluated branch.
The $D=0$ limits cover Model 9 at $\delta=-2/3$ and Model 10 at
$\delta=2/3$ under the global assumptions. The $k=0$ logarithmic-ratio limits
are handled separately. Negative component densities can be allowed; zero
or undefined dark-sector total in a nonlinear interaction cannot.

## Global real domains and positivity filters

Every model requires finite EOS and positive finite $c_0,e_0$.
All requested expansion points require $a>0$ and finite $E^2>0$.
The following are the **implemented selected domains**, not a claim to
enumerate every mathematically possible analytic continuation.

| Model | Global real-domain condition |
|---:|---|
| 1–5 | $(d_c+d_e+u)^2\ge4d_cd_e$ |
| 6 | No additional global restriction for finite coupling |
| 7 | $u<0$, $u<\delta\le-u/r$ |
| 8 | $u<0$, $u<\delta\le-ur$ |
| 9 | $\delta\le-2u/P$ **or** $\delta\ge-2/3$ |
| 10 | $-2ur/P\le\delta\le2/3$ |

For Models 9–10, also require $w<-1/3$, $v>-1/3$ and
$P=(1+3v)r+1+3w<0$. These assumptions imply $u<0$.

The optional global component-positivity filter adds the following, within
the selected domain. For interacting Models 1–5, 7–10 it also requires $u<0$.
Uncoupled Models 1–5 pass positivity independently of $u$; Model 6 always passes.

| Model | Positivity condition |
|---:|---|
| 1 | $d_c,d_e\ge0$, $d_cr+d_e\le-ur/(1+r)$ |
| 2 | $0\le\delta\le-ur/(1+r)^2$ |
| 3 | $\delta=0$ |
| 4 | $0\le\delta\le-u/(1+r)$ |
| 5 | $0\le\delta\le-ur/(1+r)$ |
| 6 | Always, given positive present densities |
| 7 | $0\le\delta\le-u/r$ |
| 8 | $0\le\delta\le-ur$ |
| 9 | $\delta\le-2u/P$ **or** $\delta\ge0$ |
| 10 | $-2ur/P\le\delta\le0$ |

## Future behavior and exceptional coefficients

The code separates `big_rip`, `no_big_rip`, `negative_future_total`,
`finite_scale_singularity` and `outside_assumptions`.
The optional future filter accepts only `no_big_rip`.

For Models 1–5, use the surviving leading mode of $T=c+e$:
$T_0=c_0+e_0$, $T'_0=-3[(1+v)c_0+(1+w)e_0]$.
For distinct eigenvalues $\lambda_+>\lambda_-$,

```math
C_+=\frac{T'_0-\lambda_-T_0}{\lambda_+-\lambda_-},\qquad
C_-=T_0-C_+.
```

Use $(C_+,\lambda_+)$ unless $C_+=0$, when the lower mode survives.
For a repeated eigenvalue $t$,
$T=e^{tx}[T_0+(T'_0-tT_0)x]$; the logarithmic coefficient controls its
future sign unless it vanishes. Negative leading total is classified separately.
For a positive leading total, a positive surviving power means Big Rip;
a nonpositive power means no Big Rip. The repeated case with power zero
has at most linear growth in $\ln a$, which does not give finite future proper time.

For Models 6–8 the surviving total-density power $p$ is

| Model | $p$ in the future |
|---:|---|
| 6, $u+\delta<0$ | $-3(1+w)$ |
| 6, $u+\delta>0$ | $-3(1+v)$ |
| 6, $u+\delta=0$ | $-3[1+(vr+w)/(1+r)]$ |
| 7, $u+\delta r\ne0$ | $-3(1+w)$ |
| 7, $u+\delta r=0$ | $-3[1+v-u\delta/(u-\delta)]$ |
| 8 | $-3[1+w+u\delta/(u-\delta)]$ |

Within their real branches, Big Rip corresponds to $p>0$.
These rules retain vanishing leading coefficients rather than applying generic
inequalities at exceptional boundaries.

Under the proxy assumptions above, the implemented Model 9 rule is Big Rip
iff $w<-1$ **and** $\delta>-1-v$; otherwise it returns no Big Rip.
For Model 10: no Big Rip if $\delta\le1+w$; Big Rip if
$1+w<\delta\le2/3$; a finite-scale singularity if $\delta>2/3$.
The last branch is excluded by the global real-domain prior but is reported
independently by the future classifier. These classifications are used together
with the selected-domain checks, not as unconditional extensions outside them.

## Independent switches and perturbative scope

`ALLOW_NEGATIVE_ENERGIES=False` enforces global component positivity.
`ALLOW_BIG_RIP=False` accepts only the classified no-Big-Rip future.
`ALLOW_DOOM_FACTOR_INSTABILITIES=False` enforces the available cold-DM
sufficient doom criteria. Setting a switch to `True` disables that optional
filter, not mathematical domain or finite-expansion checks.

For nonzero interaction with $v=0$, the implemented doom rules are:
Model 1: $d_e>0$ and $w<-1$; Models 5, 6, 8:
$\delta(1+w)<0$; Models 2, 3, 4, 7: $w<-1$.
Zero interaction passes the pointwise check. Configuration permits enforcing
these criteria only for Models 1–8 with free `w` and fixed `w_dm=0`.
Warm/free DM EOS, fixed `w=-1`, and Models 9–10 are unclassified configurations
and raise an error if that filter is requested. Allowing them to run does not
establish perturbative stability.

## Shared CLASS pressure and sound horizon

All ten models use the same CLASS backend; selector 0 is the non-IDE route.
The warm homogeneous pressure and derivative are

```math
p_D=vc+we,\qquad
\frac{dp_D}{d\ln a}=-3v(1+v)c-3w(1+w)e+(v-w)\frac{Q}{H}.
```

CLASS uses $p'_D=aH\,dp_D/d\ln a$ for the conformal-time derivative.
DM pressure contributes $3vc$ to its radiation bookkeeping and $(1-3v)c$
to its matter bookkeeping. This bookkeeping does not introduce additional species.
Standard thermodynamics supplies `rs_drag()`; no warm IDE perturbation
closure is added. Failed strict model-derived sound horizons are rejected,
not replaced with a different sound-horizon approximation.

See [the warm-DM implementation/validation report](../validation/WARM_DM.md),
[the convergence policy](../reproducibility/SAIP2026_IDE/CONVERGENCE.md), and
[the performance/BBN limitation checkpoint](../validation/PERFORMANCE_CHECKPOINT.md).
