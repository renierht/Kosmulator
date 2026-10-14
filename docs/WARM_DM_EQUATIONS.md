# Warm-dark-matter IDE models: equations and conditions

This reference uses the same notation and model order as `Kosmulator_IDE_v2`
and the supplied **Warm_DM_10_models_Kosmulator_BigRip_Clarified_portrait.pdf**
(13 October 2026). The density solutions are written explicitly in $`a`$, and
$`H_j(z)`$ explicitly in redshift, without replacing the densities or EOS
parameters by implementation variables. The only extra EOS parameter relative
to the cold-DM reference is $`w_{\mathrm{dm}}`$.

The main formulas follow the PDF's generic branches. Separate notes distinguish
removable singularities of these displayed formulas from excluded parameter
domains, and distinguish the PDF's qualified Big-Rip table from the more general
future-mode checks in Kosmulator. This document changes notation and presentation,
not the implemented physics.

## Conventions

- $`w`$ is the constant dark-energy EOS; $`w_{\mathrm{dm}}`$ is the constant DM EOS.
- $`\rho_{\mathrm{dm},0}`$ and $`\rho_{\mathrm{de},0}`$ are present physical densities.
- $`\Omega_{\mathrm{bm},0}`$ denotes baryons, as in the PDF.
- $`Q>0`$ transfers energy from dark energy to dark matter.

```math
r_0=\frac{\rho_{\mathrm{dm},0}}{\rho_{\mathrm{de},0}}
=\frac{\Omega_{\mathrm{dm},0}}{\Omega_{\mathrm{de},0}},
\qquad a=\frac{1}{1+z},\qquad h_j(z)=\frac{H_j(z)}{H_0}.
```

```math
\begin{aligned}
\dot\rho_{\mathrm{dm}}+3H(1+w_{\mathrm{dm}})\rho_{\mathrm{dm}}&=Q,\\
\dot\rho_{\mathrm{de}}+3H(1+w)\rho_{\mathrm{de}}&=-Q,\\
H^2&=\frac{8\pi G}{3}(\rho_{\mathrm{bm}}+\rho_{\mathrm{r}}+\rho_{\mathrm{dm}}+\rho_{\mathrm{de}}),\\
\rho_{\mathrm{bm}}&=\rho_{\mathrm{bm},0}a^{-3},\qquad
\rho_{\mathrm{r}}=\rho_{\mathrm{r},0}a^{-4}.
\end{aligned}
```

Spatial flatness gives
$`\Omega_{\mathrm{de},0}=1-\Omega_{\mathrm{bm},0}-\Omega_{\mathrm{r},0}-\Omega_{\mathrm{dm},0}`$.
For Models 1–8, the exact cold-to-warm correspondence is

```math
\rho_i(a;w,w_{\mathrm{dm}})=a^{-3w_{\mathrm{dm}}}
\rho_i^{\mathrm{CDM}}(a;w-w_{\mathrm{dm}}),\qquad i\in\{\mathrm{dm},\mathrm{de}\}.
```

The extra factor is essential: replacing $`w`$ alone does not give the warm-DM
solution or its future-singularity condition. Setting $`w_{\mathrm{dm}}=0`$
recovers the original eight-model formulas.

For each model the effective EOS, where the relevant density is nonzero, follows from

```math
w_{\mathrm{dm}}^{\mathrm{eff}}=w_{\mathrm{dm}}-\frac{Q}{3H\rho_{\mathrm{dm}}},
\qquad w_{\mathrm{de}}^{\mathrm{eff}}=w+\frac{Q}{3H\rho_{\mathrm{de}}}.
```

## Explicit background solutions

### Model 1: `Linear_IDE_1`

```math
Q=3H(\delta_{\mathrm{dm}}\rho_{\mathrm{dm}}+\delta_{\mathrm{de}}\rho_{\mathrm{de}})
```

The discriminant is

```math
\Delta=\sqrt{(\delta_{\mathrm{dm}}+\delta_{\mathrm{de}}+w-w_{\mathrm{dm}})^2-4\delta_{\mathrm{dm}}\delta_{\mathrm{de}}}.
```

**Densities.**

```math
\begin{aligned}
\rho_{\mathrm{dm}}(a)={}&-\frac{\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}+w-w_{\mathrm{dm}}+\Delta}{4(w-w_{\mathrm{dm}})\Delta}
 \left[\rho_{\mathrm{de},0}(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}+w-w_{\mathrm{dm}}-\Delta)+\rho_{\mathrm{dm},0}(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}-w+w_{\mathrm{dm}}-\Delta)\right]
 a^{\frac{3}{2}(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}-w-w_{\mathrm{dm}}-2+\Delta)}\\
&+\frac{\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}+w-w_{\mathrm{dm}}-\Delta}{4(w-w_{\mathrm{dm}})\Delta}
 \left[\rho_{\mathrm{de},0}(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}+w-w_{\mathrm{dm}}+\Delta)+\rho_{\mathrm{dm},0}(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}-w+w_{\mathrm{dm}}+\Delta)\right]
 a^{\frac{3}{2}(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}-w-w_{\mathrm{dm}}-2-\Delta)},\\
\rho_{\mathrm{de}}(a)={}&\frac{\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}-w+w_{\mathrm{dm}}+\Delta}{4(w-w_{\mathrm{dm}})\Delta}
 \left[\rho_{\mathrm{de},0}(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}+w-w_{\mathrm{dm}}-\Delta)+\rho_{\mathrm{dm},0}(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}-w+w_{\mathrm{dm}}-\Delta)\right]
 a^{\frac{3}{2}(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}-w-w_{\mathrm{dm}}-2+\Delta)}\\
&-\frac{\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}-w+w_{\mathrm{dm}}-\Delta}{4(w-w_{\mathrm{dm}})\Delta}
 \left[\rho_{\mathrm{de},0}(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}+w-w_{\mathrm{dm}}+\Delta)+\rho_{\mathrm{dm},0}(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}-w+w_{\mathrm{dm}}+\Delta)\right]
 a^{\frac{3}{2}(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}-w-w_{\mathrm{dm}}-2-\Delta)}.
\end{aligned}
```

**Expansion history.**

```math
\boxed{\begin{aligned}
H_1(z)=H_0\Bigg\{&\Omega_{\mathrm{bm},0}(1+z)^3+\Omega_{\mathrm{r},0}(1+z)^4\\
&-\frac{\Omega_{\mathrm{de},0}(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}+w-w_{\mathrm{dm}}-\Delta)+\Omega_{\mathrm{dm},0}(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}-w+w_{\mathrm{dm}}-\Delta)}{2\Delta}
(1+z)^{-\frac{3}{2}(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}-w-w_{\mathrm{dm}}-2+\Delta)}\\
&+\frac{\Omega_{\mathrm{de},0}(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}+w-w_{\mathrm{dm}}+\Delta)+\Omega_{\mathrm{dm},0}(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}-w+w_{\mathrm{dm}}+\Delta)}{2\Delta}
(1+z)^{-\frac{3}{2}(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}-w-w_{\mathrm{dm}}-2-\Delta)}\Bigg\}^{1/2}.
\end{aligned}}
```

**Effective equations of state.**

```math
w_{\mathrm{dm}}^{\mathrm{eff}}=w_{\mathrm{dm}}-\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}\rho_{\mathrm{de}}/\rho_{\mathrm{dm}},\qquad w_{\mathrm{de}}^{\mathrm{eff}}=w+\delta_{\mathrm{de}}+\delta_{\mathrm{dm}}\rho_{\mathrm{dm}}/\rho_{\mathrm{de}}
```

### Model 2: `Linear_IDE_2`

```math
Q=3H\delta(\rho_{\mathrm{dm}}+\rho_{\mathrm{de}})
```

This is Model 1 with $`\delta_{\mathrm{dm}}=\delta_{\mathrm{de}}=\delta`$.

The discriminant is

```math
\Delta=\sqrt{(w-w_{\mathrm{dm}})[w-w_{\mathrm{dm}}+4\delta]}.
```

**Densities.**

```math
\begin{aligned}
\rho_{\mathrm{dm}}(a)={}&-\frac{w-w_{\mathrm{dm}}+\Delta}{4(w-w_{\mathrm{dm}})\Delta}
\left[\rho_{\mathrm{de},0}(w-w_{\mathrm{dm}}-\Delta)+\rho_{\mathrm{dm},0}(-w+w_{\mathrm{dm}}-\Delta)\right]
 a^{-\frac{3}{2}(2+w+w_{\mathrm{dm}}-\Delta)}\\
&+\frac{w-w_{\mathrm{dm}}-\Delta}{4(w-w_{\mathrm{dm}})\Delta}
\left[\rho_{\mathrm{de},0}(w-w_{\mathrm{dm}}+\Delta)+\rho_{\mathrm{dm},0}(-w+w_{\mathrm{dm}}+\Delta)\right]
 a^{-\frac{3}{2}(2+w+w_{\mathrm{dm}}+\Delta)},\\
\rho_{\mathrm{de}}(a)={}&\frac{-w+w_{\mathrm{dm}}+\Delta}{4(w-w_{\mathrm{dm}})\Delta}
\left[\rho_{\mathrm{de},0}(w-w_{\mathrm{dm}}-\Delta)+\rho_{\mathrm{dm},0}(-w+w_{\mathrm{dm}}-\Delta)\right]
 a^{-\frac{3}{2}(2+w+w_{\mathrm{dm}}-\Delta)}\\
&-\frac{-w+w_{\mathrm{dm}}-\Delta}{4(w-w_{\mathrm{dm}})\Delta}
\left[\rho_{\mathrm{de},0}(w-w_{\mathrm{dm}}+\Delta)+\rho_{\mathrm{dm},0}(-w+w_{\mathrm{dm}}+\Delta)\right]
 a^{-\frac{3}{2}(2+w+w_{\mathrm{dm}}+\Delta)},
\end{aligned}
```

**Expansion history.**

```math
\boxed{\begin{aligned}
H_2(z)=H_0\Bigg\{&\Omega_{\mathrm{bm},0}(1+z)^3+\Omega_{\mathrm{r},0}(1+z)^4\\
&-\frac{\Omega_{\mathrm{de},0}(w-w_{\mathrm{dm}}-\Delta)+\Omega_{\mathrm{dm},0}(-w+w_{\mathrm{dm}}-\Delta)}{2\Delta}
(1+z)^{\frac{3}{2}(2+w+w_{\mathrm{dm}}-\Delta)}\\
&+\frac{\Omega_{\mathrm{de},0}(w-w_{\mathrm{dm}}+\Delta)+\Omega_{\mathrm{dm},0}(-w+w_{\mathrm{dm}}+\Delta)}{2\Delta}
(1+z)^{\frac{3}{2}(2+w+w_{\mathrm{dm}}+\Delta)}\Bigg\}^{1/2}.
\end{aligned}}
```

**Effective equations of state.**

```math
w_{\mathrm{dm}}^{\mathrm{eff}}=w_{\mathrm{dm}}-\delta(1+\rho_{\mathrm{de}}/\rho_{\mathrm{dm}}),\qquad w_{\mathrm{de}}^{\mathrm{eff}}=w+\delta(1+\rho_{\mathrm{dm}}/\rho_{\mathrm{de}})
```

### Model 3: `Linear_IDE_3`

```math
Q=3H\delta(\rho_{\mathrm{dm}}-\rho_{\mathrm{de}})
```

This is Model 1 with $`\delta_{\mathrm{dm}}=\delta`$ and $`\delta_{\mathrm{de}}=-\delta`$. Its interaction changes sign at $`\rho_{\mathrm{dm}}=\rho_{\mathrm{de}}`$ when the coupling is nonzero.

The discriminant is

```math
\Delta=\sqrt{4\delta^2+(w-w_{\mathrm{dm}})^2}.
```

**Densities.**

```math
\begin{aligned}
\rho_{\mathrm{dm}}(a)={}&-\frac{2\delta+w-w_{\mathrm{dm}}+\Delta}{4(w-w_{\mathrm{dm}})\Delta}
\left[\rho_{\mathrm{de},0}(2\delta+w-w_{\mathrm{dm}}-\Delta)+\rho_{\mathrm{dm},0}(2\delta-w+w_{\mathrm{dm}}-\Delta)\right]
 a^{\frac{3}{2}(2\delta-w-w_{\mathrm{dm}}-2+\Delta)}\\
&+\frac{2\delta+w-w_{\mathrm{dm}}-\Delta}{4(w-w_{\mathrm{dm}})\Delta}
\left[\rho_{\mathrm{de},0}(2\delta+w-w_{\mathrm{dm}}+\Delta)+\rho_{\mathrm{dm},0}(2\delta-w+w_{\mathrm{dm}}+\Delta)\right]
 a^{\frac{3}{2}(2\delta-w-w_{\mathrm{dm}}-2-\Delta)},\\
\rho_{\mathrm{de}}(a)={}&\frac{2\delta-w+w_{\mathrm{dm}}+\Delta}{4(w-w_{\mathrm{dm}})\Delta}
\left[\rho_{\mathrm{de},0}(2\delta+w-w_{\mathrm{dm}}-\Delta)+\rho_{\mathrm{dm},0}(2\delta-w+w_{\mathrm{dm}}-\Delta)\right]
 a^{\frac{3}{2}(2\delta-w-w_{\mathrm{dm}}-2+\Delta)}\\
&-\frac{2\delta-w+w_{\mathrm{dm}}-\Delta}{4(w-w_{\mathrm{dm}})\Delta}
\left[\rho_{\mathrm{de},0}(2\delta+w-w_{\mathrm{dm}}+\Delta)+\rho_{\mathrm{dm},0}(2\delta-w+w_{\mathrm{dm}}+\Delta)\right]
 a^{\frac{3}{2}(2\delta-w-w_{\mathrm{dm}}-2-\Delta)},
\end{aligned}
```

**Expansion history.**

```math
\boxed{\begin{aligned}
H_3(z)=H_0\Bigg\{&\Omega_{\mathrm{bm},0}(1+z)^3+\Omega_{\mathrm{r},0}(1+z)^4\\
&-\frac{\Omega_{\mathrm{de},0}(2\delta+w-w_{\mathrm{dm}}-\Delta)+\Omega_{\mathrm{dm},0}(2\delta-w+w_{\mathrm{dm}}-\Delta)}{2\Delta}
(1+z)^{-\frac{3}{2}(2\delta-w-w_{\mathrm{dm}}-2+\Delta)}\\
&+\frac{\Omega_{\mathrm{de},0}(2\delta+w-w_{\mathrm{dm}}+\Delta)+\Omega_{\mathrm{dm},0}(2\delta-w+w_{\mathrm{dm}}+\Delta)}{2\Delta}
(1+z)^{-\frac{3}{2}(2\delta-w-w_{\mathrm{dm}}-2-\Delta)}\Bigg\}^{1/2}.
\end{aligned}}
```

**Effective equations of state.**

```math
w_{\mathrm{dm}}^{\mathrm{eff}}=w_{\mathrm{dm}}-\delta(1-\rho_{\mathrm{de}}/\rho_{\mathrm{dm}}),\qquad w_{\mathrm{de}}^{\mathrm{eff}}=w+\delta(\rho_{\mathrm{dm}}/\rho_{\mathrm{de}}-1)
```

### Model 4: `Linear_IDE_4`

```math
Q=3H\delta\rho_{\mathrm{dm}}
```

**Densities.**

```math
\begin{aligned}
\rho_{\mathrm{dm}}(a)&=\rho_{\mathrm{dm},0} a^{-3(1+w_{\mathrm{dm}}-\delta)},\\
\rho_{\mathrm{de}}(a)&=\left[\rho_{\mathrm{de},0}+\rho_{\mathrm{dm},0}\frac{\delta}{\delta+w-w_{\mathrm{dm}}}\big(1-a^{3(\delta+w-w_{\mathrm{dm}})}\big)\right]a^{-3(1+w)}
\end{aligned}
```

**Expansion history.**

```math
\boxed{\begin{aligned}
H_4(z)=H_0\Bigg\{&\Omega_{\mathrm{bm},0}(1+z)^3+\Omega_{\mathrm{r},0}(1+z)^4+\Omega_{\mathrm{dm},0}(1+z)^{3(1+w_{\mathrm{dm}}-\delta)}\\
&+\left[\Omega_{\mathrm{de},0}+\Omega_{\mathrm{dm},0}\frac{\delta}{\delta+w-w_{\mathrm{dm}}}\big(1-(1+z)^{-3(\delta+w-w_{\mathrm{dm}})}\big)\right](1+z)^{3(1+w)}\Bigg\}^{1/2}.
\end{aligned}}
```

**Effective equations of state.**

```math
w_{\mathrm{dm}}^{\mathrm{eff}}=w_{\mathrm{dm}}-\delta,\qquad w_{\mathrm{de}}^{\mathrm{eff}}=w+\delta\rho_{\mathrm{dm}}/\rho_{\mathrm{de}}
```

### Model 5: `Linear_IDE_5`

```math
Q=3H\delta\rho_{\mathrm{de}}
```

**Densities.**

```math
\begin{aligned}
\rho_{\mathrm{dm}}(a)&=\left[\rho_{\mathrm{dm},0}+\rho_{\mathrm{de},0}\frac{\delta}{\delta+w-w_{\mathrm{dm}}}\big(1-a^{-3(\delta+w-w_{\mathrm{dm}})}\big)\right]a^{-3(1+w_{\mathrm{dm}})},\\
\rho_{\mathrm{de}}(a)&=\rho_{\mathrm{de},0} a^{-3(1+w+\delta)}
\end{aligned}
```

**Expansion history.**

```math
\boxed{\begin{aligned}
H_5(z)=H_0\Bigg\{&\Omega_{\mathrm{bm},0}(1+z)^3+\Omega_{\mathrm{r},0}(1+z)^4+\Omega_{\mathrm{de},0}(1+z)^{3(1+w+\delta)}\\
&+\left[\Omega_{\mathrm{dm},0}+\Omega_{\mathrm{de},0}\frac{\delta}{\delta+w-w_{\mathrm{dm}}}\big(1-(1+z)^{3(\delta+w-w_{\mathrm{dm}})}\big)\right](1+z)^{3(1+w_{\mathrm{dm}})}\Bigg\}^{1/2}.
\end{aligned}}
```

**Effective equations of state.**

```math
w_{\mathrm{dm}}^{\mathrm{eff}}=w_{\mathrm{dm}}-\delta\rho_{\mathrm{de}}/\rho_{\mathrm{dm}},\qquad w_{\mathrm{de}}^{\mathrm{eff}}=w+\delta
```

### Model 6: `NonLinear_IDE_1`

```math
Q=3H\delta\,\rho_{\mathrm{dm}}\rho_{\mathrm{de}}/(\rho_{\mathrm{dm}}+\rho_{\mathrm{de}})
```

**Densities.**

```math
\begin{aligned}
\rho_{\mathrm{dm}}(a)&=\rho_{\mathrm{dm},0} a^{-3(1+w_{\mathrm{dm}}-\delta)}
\left[\frac{1+r_0a^{3(w-w_{\mathrm{dm}}+\delta)}}{1+r_0}\right]^{-\frac{\delta}{w-w_{\mathrm{dm}}+\delta}},\\
\rho_{\mathrm{de}}(a)&=\rho_{\mathrm{de},0} a^{-3(1+w)}
\left[\frac{1+r_0a^{3(w-w_{\mathrm{dm}}+\delta)}}{1+r_0}\right]^{-\frac{\delta}{w-w_{\mathrm{dm}}+\delta}}
\end{aligned}
```

**Expansion history.**

```math
\boxed{\begin{aligned}
H_6(z)=H_0\Bigg\{&\Omega_{\mathrm{bm},0}(1+z)^3+\Omega_{\mathrm{r},0}(1+z)^4\\
&+\left[\Omega_{\mathrm{dm},0}(1+z)^{3(1+w_{\mathrm{dm}}-\delta)}+\Omega_{\mathrm{de},0}(1+z)^{3(1+w)}\right]
\left[\frac{1+r_0(1+z)^{-3(w-w_{\mathrm{dm}}+\delta)}}{1+r_0}\right]^{-\frac{\delta}{w-w_{\mathrm{dm}}+\delta}}\Bigg\}^{1/2}.
\end{aligned}}
```

**Effective equations of state.**

```math
w_{\mathrm{dm}}^{\mathrm{eff}}=w_{\mathrm{dm}}-\delta\rho_{\mathrm{de}}/(\rho_{\mathrm{dm}}+\rho_{\mathrm{de}}),\qquad w_{\mathrm{de}}^{\mathrm{eff}}=w+\delta\rho_{\mathrm{dm}}/(\rho_{\mathrm{dm}}+\rho_{\mathrm{de}})
```

### Model 7: `NonLinear_IDE_2`

```math
Q=3H\delta\,\rho_{\mathrm{dm}}^2/(\rho_{\mathrm{dm}}+\rho_{\mathrm{de}})
```

**Densities.**

```math
\begin{aligned}
\rho_{\mathrm{dm}}(a)={}&\rho_{\mathrm{dm},0} a^{-3\left(1+w_{\mathrm{dm}}-\frac{(w-w_{\mathrm{dm}})\delta}{w-w_{\mathrm{dm}}-\delta}\right)}
\left[\frac{[w-w_{\mathrm{dm}}+\delta r_0]a^{-3(w-w_{\mathrm{dm}})}+r_0(w-w_{\mathrm{dm}}-\delta)}{(w-w_{\mathrm{dm}})(1+r_0)}\right]^{\frac{\delta}{w-w_{\mathrm{dm}}-\delta}},\\
\rho_{\mathrm{de}}(a)={}&\rho_{\mathrm{de},0} a^{-3\left(1+w_{\mathrm{dm}}-\frac{(w-w_{\mathrm{dm}})\delta}{w-w_{\mathrm{dm}}-\delta}\right)}
\frac{[w-w_{\mathrm{dm}}+\delta r_0]a^{-3(w-w_{\mathrm{dm}})}-\delta r_0}{w-w_{\mathrm{dm}}}
\left[\frac{[w-w_{\mathrm{dm}}+\delta r_0]a^{-3(w-w_{\mathrm{dm}})}+r_0(w-w_{\mathrm{dm}}-\delta)}{(w-w_{\mathrm{dm}})(1+r_0)}\right]^{\frac{\delta}{w-w_{\mathrm{dm}}-\delta}},
\end{aligned}
```

**Expansion history.**

```math
\boxed{\begin{aligned}
H_7(z)=H_0\Bigg\{&\Omega_{\mathrm{bm},0}(1+z)^3+\Omega_{\mathrm{r},0}(1+z)^4\\
&+\Bigg[\Omega_{\mathrm{dm},0}+\Omega_{\mathrm{de},0}\,\frac{[w-w_{\mathrm{dm}}+\delta r_0](1+z)^{3(w-w_{\mathrm{dm}})}-\delta r_0}{w-w_{\mathrm{dm}}}\Bigg]
(1+z)^{3\left(1+w_{\mathrm{dm}}-\frac{(w-w_{\mathrm{dm}})\delta}{w-w_{\mathrm{dm}}-\delta}\right)}\\
&\qquad\times
\left[\frac{[w-w_{\mathrm{dm}}+\delta r_0](1+z)^{3(w-w_{\mathrm{dm}})}+r_0(w-w_{\mathrm{dm}}-\delta)}{(w-w_{\mathrm{dm}})(1+r_0)}\right]^{\frac{\delta}{w-w_{\mathrm{dm}}-\delta}}\Bigg\}^{1/2}.
\end{aligned}}
```

**Effective equations of state.**

```math
w_{\mathrm{dm}}^{\mathrm{eff}}=w_{\mathrm{dm}}-\delta\rho_{\mathrm{dm}}/(\rho_{\mathrm{dm}}+\rho_{\mathrm{de}}),\qquad w_{\mathrm{de}}^{\mathrm{eff}}=w+\delta\rho_{\mathrm{dm}}^2/[\rho_{\mathrm{de}}(\rho_{\mathrm{dm}}+\rho_{\mathrm{de}})]
```

### Model 8: `NonLinear_IDE_3`

```math
Q=3H\delta\,\rho_{\mathrm{de}}^2/(\rho_{\mathrm{dm}}+\rho_{\mathrm{de}})
```

**Densities.**

```math
\begin{aligned}
\rho_{\mathrm{dm}}(a)={}&\rho_{\mathrm{dm},0} a^{-3\left(1+w+\frac{(w-w_{\mathrm{dm}})\delta}{w-w_{\mathrm{dm}}-\delta}\right)}
\frac{[(w-w_{\mathrm{dm}})r_0+\delta]a^{3(w-w_{\mathrm{dm}})}-\delta}{(w-w_{\mathrm{dm}})r_0}
\left[\frac{[(w-w_{\mathrm{dm}})r_0+\delta]a^{3(w-w_{\mathrm{dm}})}+w-w_{\mathrm{dm}}-\delta}{(w-w_{\mathrm{dm}})(1+r_0)}\right]^{\frac{\delta}{w-w_{\mathrm{dm}}-\delta}},\\
\rho_{\mathrm{de}}(a)={}&\rho_{\mathrm{de},0} a^{-3\left(1+w+\frac{(w-w_{\mathrm{dm}})\delta}{w-w_{\mathrm{dm}}-\delta}\right)}
\left[\frac{[(w-w_{\mathrm{dm}})r_0+\delta]a^{3(w-w_{\mathrm{dm}})}+w-w_{\mathrm{dm}}-\delta}{(w-w_{\mathrm{dm}})(1+r_0)}\right]^{\frac{\delta}{w-w_{\mathrm{dm}}-\delta}},
\end{aligned}
```

**Expansion history.**

```math
\boxed{\begin{aligned}
H_8(z)=H_0\Bigg\{&\Omega_{\mathrm{bm},0}(1+z)^3+\Omega_{\mathrm{r},0}(1+z)^4\\
&+\Bigg[\Omega_{\mathrm{dm},0}\frac{[(w-w_{\mathrm{dm}})r_0+\delta](1+z)^{-3(w-w_{\mathrm{dm}})}-\delta}{(w-w_{\mathrm{dm}})r_0}+\Omega_{\mathrm{de},0}\Bigg]
(1+z)^{3\left(1+w+\frac{(w-w_{\mathrm{dm}})\delta}{w-w_{\mathrm{dm}}-\delta}\right)}\\
&\qquad\times\left[\frac{[(w-w_{\mathrm{dm}})r_0+\delta](1+z)^{-3(w-w_{\mathrm{dm}})}+w-w_{\mathrm{dm}}-\delta}{(w-w_{\mathrm{dm}})(1+r_0)}\right]^{\frac{\delta}{w-w_{\mathrm{dm}}-\delta}}\Bigg\}^{1/2}.
\end{aligned}}
```

**Effective equations of state.**

```math
w_{\mathrm{dm}}^{\mathrm{eff}}=w_{\mathrm{dm}}-\delta\rho_{\mathrm{de}}^2/[\rho_{\mathrm{dm}}(\rho_{\mathrm{dm}}+\rho_{\mathrm{de}})],\qquad w_{\mathrm{de}}^{\mathrm{eff}}=w+\delta\rho_{\mathrm{de}}/(\rho_{\mathrm{dm}}+\rho_{\mathrm{de}})
```

### Model 9: `SignSwitch_IDE_1`

```math
Q=\frac{3H\delta}{2}\,\frac{\rho_{\mathrm{dm}}[(1+3w_{\mathrm{dm}})\rho_{\mathrm{dm}}+(1+3w)\rho_{\mathrm{de}}]}{\rho_{\mathrm{dm}}+\rho_{\mathrm{de}}}
```

The interaction contains the dark-sector deceleration expression only; baryons and radiation do not enter this kernel.

**Densities.**

```math
\begin{aligned}
\rho_{\mathrm{dm}}(a)={}&\rho_{\mathrm{dm},0}\,a^{-\frac{6(1+w_{\mathrm{dm}}+\delta)}{2+3\delta}}
\left[\frac{\big[2(w-w_{\mathrm{dm}})+\delta(1+3w)+\delta(1+3w_{\mathrm{dm}})r_0\big]
 a^{-\frac{3}{2}[2(w-w_{\mathrm{dm}})+\delta(1+3w)]}
 +(w-w_{\mathrm{dm}})(2+3\delta)r_0}
 {[2(w-w_{\mathrm{dm}})+\delta(1+3w)](1+r_0)}\right]^{-\frac{3\delta}{2+3\delta}},\\
\rho_{\mathrm{de}}(a)={}&a^{-\frac{6(1+w_{\mathrm{dm}}+\delta)}{2+3\delta}}
\Bigg\{(\rho_{\mathrm{dm},0}+\rho_{\mathrm{de},0})
\left[\frac{\big[2(w-w_{\mathrm{dm}})+\delta(1+3w)+\delta(1+3w_{\mathrm{dm}})r_0\big]
 a^{-\frac{3}{2}[2(w-w_{\mathrm{dm}})+\delta(1+3w)]}+(w-w_{\mathrm{dm}})(2+3\delta)r_0}
 {[2(w-w_{\mathrm{dm}})+\delta(1+3w)](1+r_0)}\right]^{\frac{2}{2+3\delta}}\\
&-\rho_{\mathrm{dm},0}
\left[\frac{\big[2(w-w_{\mathrm{dm}})+\delta(1+3w)+\delta(1+3w_{\mathrm{dm}})r_0\big]
 a^{-\frac{3}{2}[2(w-w_{\mathrm{dm}})+\delta(1+3w)]}+(w-w_{\mathrm{dm}})(2+3\delta)r_0}
 {[2(w-w_{\mathrm{dm}})+\delta(1+3w)](1+r_0)}\right]^{-\frac{3\delta}{2+3\delta}}\Bigg\},
\end{aligned}
```

**Expansion history.**

```math
\boxed{\begin{aligned}
H_9(z)=H_0\Bigg\{&\Omega_{\mathrm{bm},0}(1+z)^3+\Omega_{\mathrm{r},0}(1+z)^4\\
&+(\Omega_{\mathrm{dm},0}+\Omega_{\mathrm{de},0})(1+z)^{\frac{6(1+w_{\mathrm{dm}}+\delta)}{2+3\delta}}
\Bigg[\frac{\big[2(w-w_{\mathrm{dm}})+\delta(1+3w)+\delta(1+3w_{\mathrm{dm}})r_0\big]
 (1+z)^{\frac{3}{2}[2(w-w_{\mathrm{dm}})+\delta(1+3w)]}+(w-w_{\mathrm{dm}})(2+3\delta)r_0}
 {[2(w-w_{\mathrm{dm}})+\delta(1+3w)](1+r_0)}\Bigg]^{\frac{2}{2+3\delta}}\Bigg\}^{1/2}.
\end{aligned}}
```

**Effective equations of state.**

```math
\begin{aligned}
w_{\mathrm{dm}}^{\mathrm{eff}}&=w_{\mathrm{dm}}-\frac{\delta[(1+3w_{\mathrm{dm}})\rho_{\mathrm{dm}}+(1+3w)\rho_{\mathrm{de}}]}{2(\rho_{\mathrm{dm}}+\rho_{\mathrm{de}})},\\
w_{\mathrm{de}}^{\mathrm{eff}}&=w+\frac{\delta\rho_{\mathrm{dm}}[(1+3w_{\mathrm{dm}})\rho_{\mathrm{dm}}+(1+3w)\rho_{\mathrm{de}}]}{2\rho_{\mathrm{de}}(\rho_{\mathrm{dm}}+\rho_{\mathrm{de}})}.
\end{aligned}
```

**Switching redshift.** This expression describes an actual sign change only for nonzero coupling, a positive powered base and a crossing on the selected real branch with positive component densities. The crossing satisfies $`(1+3w_{\mathrm{dm}})\rho_{\mathrm{dm}}+(1+3w)\rho_{\mathrm{de}}=0`$.

```math
z_{\mathrm{switch},9}=\left[\frac{-2(w-w_{\mathrm{dm}})(1+3w_{\mathrm{dm}})r_0}{(1+3w)\big\{[2(w-w_{\mathrm{dm}})+\delta(1+3w)]+\delta(1+3w_{\mathrm{dm}})r_0\big\}}\right]^{\frac{2}{3[2(w-w_{\mathrm{dm}})+\delta(1+3w)]}}-1.
```

### Model 10: `SignSwitch_IDE_2`

```math
Q=\frac{3H\delta}{2}\,\frac{\rho_{\mathrm{de}}[(1+3w_{\mathrm{dm}})\rho_{\mathrm{dm}}+(1+3w)\rho_{\mathrm{de}}]}{\rho_{\mathrm{dm}}+\rho_{\mathrm{de}}}
```

The interaction contains the dark-sector deceleration expression only; baryons and radiation do not enter this kernel.

**Densities.**

```math
\begin{aligned}
\rho_{\mathrm{de}}(a)={}&\rho_{\mathrm{de},0}\,a^{-\frac{6(1+w-\delta)}{2-3\delta}}
\left[\frac{\big[2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})\big]r_0+\delta(1+3w)}{[2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})](1+r_0)}
 a^{\frac{3}{2}[2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})]}
 +\frac{(w-w_{\mathrm{dm}})(2-3\delta)}{[2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})](1+r_0)}\right]^{\frac{3\delta}{2-3\delta}},\\
\rho_{\mathrm{dm}}(a)={}&a^{-\frac{6(1+w-\delta)}{2-3\delta}}
\Bigg\{(\rho_{\mathrm{dm},0}+\rho_{\mathrm{de},0})
\left[\frac{\big[2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})\big]r_0+\delta(1+3w)}{[2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})](1+r_0)}
 a^{\frac{3}{2}[2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})]}
 +\frac{(w-w_{\mathrm{dm}})(2-3\delta)}{[2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})](1+r_0)}\right]^{\frac{2}{2-3\delta}}\\
&-\rho_{\mathrm{de},0}
\left[\frac{\big[2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})\big]r_0+\delta(1+3w)}{[2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})](1+r_0)}
 a^{\frac{3}{2}[2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})]}
 +\frac{(w-w_{\mathrm{dm}})(2-3\delta)}{[2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})](1+r_0)}\right]^{\frac{3\delta}{2-3\delta}}\Bigg\},
\end{aligned}
```

**Expansion history.**

```math
\boxed{\begin{aligned}
H_{10}(z)=H_0\Bigg\{&\Omega_{\mathrm{bm},0}(1+z)^3+\Omega_{\mathrm{r},0}(1+z)^4\\
&+(\Omega_{\mathrm{dm},0}+\Omega_{\mathrm{de},0})(1+z)^{\frac{6(1+w-\delta)}{2-3\delta}}
\Bigg[\frac{\big[2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})\big]r_0+\delta(1+3w)}{[2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})](1+r_0)}
 (1+z)^{-\frac{3}{2}[2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})]}
 +\frac{(w-w_{\mathrm{dm}})(2-3\delta)}{[2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})](1+r_0)}\Bigg]^{\frac{2}{2-3\delta}}\Bigg\}^{1/2}.
\end{aligned}}
```

**Effective equations of state.**

```math
\begin{aligned}
w_{\mathrm{dm}}^{\mathrm{eff}}&=w_{\mathrm{dm}}-\frac{\delta\rho_{\mathrm{de}}[(1+3w_{\mathrm{dm}})\rho_{\mathrm{dm}}+(1+3w)\rho_{\mathrm{de}}]}{2\rho_{\mathrm{dm}}(\rho_{\mathrm{dm}}+\rho_{\mathrm{de}})},\\
w_{\mathrm{de}}^{\mathrm{eff}}&=w+\frac{\delta[(1+3w_{\mathrm{dm}})\rho_{\mathrm{dm}}+(1+3w)\rho_{\mathrm{de}}]}{2(\rho_{\mathrm{dm}}+\rho_{\mathrm{de}})}.
\end{aligned}
```

**Switching redshift.** This expression describes an actual sign change only for nonzero coupling, a positive powered base and a crossing on the selected real branch with positive component densities. The crossing satisfies $`(1+3w_{\mathrm{dm}})\rho_{\mathrm{dm}}+(1+3w)\rho_{\mathrm{de}}=0`$.

```math
z_{\mathrm{switch},10}=\left[\frac{-2(w-w_{\mathrm{dm}})(1+3w)}{(1+3w_{\mathrm{dm}})\big\{[2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})]r_0+\delta(1+3w)\big\}}\right]^{-\frac{2}{3[2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})]}}-1.
```

## Conditions for real and well-defined densities

These are the selected global branches used by Kosmulator. All models require
finite EOS/couplings, strictly positive present dark densities and a finite,
positive square-root argument in $`H(z)`$ at every requested point. For nonlinear
interactions, the dark-sector denominator must not vanish, and every real-power
base must stay positive. A zero denominator in a **generic displayed formula**
is not automatically a physical singularity: implemented limits are listed below.

| Model | Reality / selected global branch | Generic formula requiring a separate limit |
|---:|---|---|
| 1 | $`(\delta_{\mathrm{dm}}+\delta_{\mathrm{de}}+w-w_{\mathrm{dm}})^2\ge4\delta_{\mathrm{dm}}\delta_{\mathrm{de}}`$ | $`w=w_{\mathrm{dm}}`$ or $`\Delta=0`$ |
| 2 | $`(w-w_{\mathrm{dm}})(w-w_{\mathrm{dm}}+4\delta)\ge0`$ | $`w=w_{\mathrm{dm}}`$ or $`\Delta=0`$ |
| 3 | $`\Delta`$ is real for all real parameters | $`w=w_{\mathrm{dm}}`$; $`\Delta=0`$ at the uncoupled equal-EOS point |
| 4 | Always real | $`\delta+w-w_{\mathrm{dm}}=0`$ |
| 5 | Always real | $`\delta+w-w_{\mathrm{dm}}=0`$ |
| 6 | Real on its positive solution branch | $`\delta+w-w_{\mathrm{dm}}=0`$ |
| 7 | $`w-w_{\mathrm{dm}}<0`$, $`w-w_{\mathrm{dm}}<\delta\le-(w-w_{\mathrm{dm}})/r_0`$ | The chosen sufficient branch excludes $`w=w_{\mathrm{dm}}`$ and $`\delta=w-w_{\mathrm{dm}}`$ |
| 8 | $`w-w_{\mathrm{dm}}<0`$, $`w-w_{\mathrm{dm}}<\delta\le-(w-w_{\mathrm{dm}})r_0`$ | The chosen sufficient branch excludes $`w=w_{\mathrm{dm}}`$ and $`\delta=w-w_{\mathrm{dm}}`$ |
| 9 | $`\delta\le-\frac{2(w-w_{\mathrm{dm}})}{(1+3w_{\mathrm{dm}})r_0+1+3w}`$ **or** $`\delta\ge-2/3`$ | $`2+3\delta=0`$ or $`2(w-w_{\mathrm{dm}})+\delta(1+3w)=0`$ |
| 10 | $`-\frac{2(w-w_{\mathrm{dm}})r_0}{(1+3w_{\mathrm{dm}})r_0+1+3w}\le\delta\le2/3`$ | $`2-3\delta=0`$ or $`2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})=0`$ |

The Model 9–10 conditions additionally assume

```math
w<-\frac{1}{3},\qquad w_{\mathrm{dm}}>-\frac{1}{3},\qquad
\rho_{\mathrm{dm},0}>0,\quad\rho_{\mathrm{de},0}>0,\qquad
(1+3w_{\mathrm{dm}})r_0+1+3w<0.
```

The last condition is present acceleration of the **dark-sector proxy**,
not necessarily acceleration of the complete baryon/radiation-containing universe.
The global branch checks remain active even in an “unrestricted” run; that term
means that the optional physical filters below are disabled.

## Conditions for globally nonnegative dark-sector densities

Apply this table together with the real-domain conditions. For the interacting
Models 1–5 and 7–8 the adopted bounds assume $`w<w_{\mathrm{dm}}`$; Models 9–10
use the assumptions above. Uncoupled Models 1–5 pass the positivity check
without requiring that hierarchy. Model 6 is positive for either EOS hierarchy.

| Model | Global nonnegative-density condition |
|---:|---|
| 1 | $`\delta_{\mathrm{dm}}\ge0`$, $`\delta_{\mathrm{de}}\ge0`$, $`\delta_{\mathrm{dm}}r_0+\delta_{\mathrm{de}}\le-\frac{(w-w_{\mathrm{dm}})r_0}{1+r_0}`$ |
| 2 | $`0\le\delta\le-\frac{(w-w_{\mathrm{dm}})r_0}{(1+r_0)^2}`$ |
| 3 | No nonzero-coupling globally positive domain; $`\delta=0`$ is uncoupled |
| 4 | $`0\le\delta\le-\frac{w-w_{\mathrm{dm}}}{1+r_0}`$ |
| 5 | $`0\le\delta\le-\frac{w-w_{\mathrm{dm}}}{1+1/r_0}`$ |
| 6 | Any finite $`\delta`$ on the positive regular solution branch |
| 7 | $`0\le\delta\le-\frac{w-w_{\mathrm{dm}}}{r_0}`$ |
| 8 | $`0\le\delta\le-(w-w_{\mathrm{dm}})r_0`$ |
| 9 | $`\delta\le-\frac{2(w-w_{\mathrm{dm}})}{(1+3w_{\mathrm{dm}})r_0+1+3w}`$ **or** $`\delta\ge0`$ |
| 10 | $`-\frac{2(w-w_{\mathrm{dm}})r_0}{(1+3w_{\mathrm{dm}})r_0+1+3w}\le\delta\le0`$ |

`ALLOW_NEGATIVE_ENERGIES=False` enforces this filter.

## Future Big-Rip conditions

A Big Rip means divergence at $`a\to\infty`$ in finite future proper time.
A finite-scale-factor singularity is a different outcome. Reality, global
component positivity and future behavior are separate checks: allowing a
negative component does not automatically determine whether a Big Rip occurs.

### Qualified table from the supplied clarified PDF

For comparison with the PDF, its no-future-Big-Rip conditions are reproduced
in the original notation below. **Rows 1–8 are the PDF's qualified working-summary
conditions, not unconditional tests over the full unrestricted parameter space.**
They presume its ordinary hierarchy, appropriate regular future branch and
surviving leading coefficients. The clarified PDF itself warns against treating
these rows as independently established, positivity-unrestricted classifications.
Kosmulator therefore uses the more general mode checks in the following section.

| Model | No-future-Big-Rip condition as tabulated in the PDF |
|---:|---|
| 1 | $`\delta_{\mathrm{dm}}(1+w)-(1+w_{\mathrm{dm}})\delta_{\mathrm{de}}\le(1+w_{\mathrm{dm}})(1+w)`$ |
| 2 | $`\delta(w-w_{\mathrm{dm}})\le(1+w_{\mathrm{dm}})(1+w)`$ |
| 3 | $`\delta(2+w_{\mathrm{dm}}+w)\le(1+w_{\mathrm{dm}})(1+w)`$ |
| 4 | $`w\ge-1`$ |
| 5 | $`\delta\ge-1-w`$ |
| 6 | $`w\ge-1`$ |
| 7 | $`w\ge-1`$ |
| 8 | $`\delta\ge\frac{(w-w_{\mathrm{dm}})(1+w)}{1+w_{\mathrm{dm}}}`$ |
| 9 | $`w\ge-1`$ **or** $`(w<-1\ \text{and}\ \delta\le-1-w_{\mathrm{dm}})`$ |
| 10 | $`\delta\le1+w`$ on its future-regular branch |

### Conditions actually applied to Models 1–8

For Models 1–5, Kosmulator adds the two displayed dark densities and examines
the largest surviving power of $`a`$. A positive leading coefficient with a
positive power gives a Big Rip; a nonpositive power gives no Big Rip. An
exactly vanishing coefficient removes that mode, so the next surviving mode
must be used. A negative leading total is reported as `negative_future_total`,
not mislabeled as a Big Rip. At a repeated root, the sign of the leading
$`\ln a`$ coefficient is included; power zero with at most one factor of
$`\ln a`$ does not give finite future proper time.

For Model 1 the two generic powers of the total density are

```math
\frac32(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}-w-w_{\mathrm{dm}}-2+\Delta),
\qquad
\frac32(\delta_{\mathrm{dm}}-\delta_{\mathrm{de}}-w-w_{\mathrm{dm}}-2-\Delta).
```

Their coefficients are precisely the two coefficients of the dark-sector
terms in $`H_1^2/H_0^2`$ above. Models 2–5 use their corresponding coupling
substitutions. Thus the explicit formulas specify both the coefficient and
power checks without introducing new fitted parameters or eigenmode notation.

For Models 6–8, the implemented surviving future powers are:

| Model and branch | Power of $`a`$ in the leading future dark-sector density |
|---|---|
| 6, $`w-w_{\mathrm{dm}}+\delta<0`$ | $`-3(1+w)`$ |
| 6, $`w-w_{\mathrm{dm}}+\delta>0`$ | $`-3(1+w_{\mathrm{dm}})`$ |
| 6, $`w-w_{\mathrm{dm}}+\delta=0`$ | $`-3\left(1+\frac{w_{\mathrm{dm}}r_0+w}{1+r_0}\right)`$ |
| 7, $`w-w_{\mathrm{dm}}+\delta r_0\ne0`$ | $`-3(1+w)`$ |
| 7, $`w-w_{\mathrm{dm}}+\delta r_0=0`$ | $`-3\left[1+w_{\mathrm{dm}}-\frac{(w-w_{\mathrm{dm}})\delta}{w-w_{\mathrm{dm}}-\delta}\right]`$ |
| 8 | $`-3\left[1+w+\frac{(w-w_{\mathrm{dm}})\delta}{w-w_{\mathrm{dm}}-\delta}\right]`$ |

Within the selected real branches, a positive power gives a Big Rip and a
nonpositive power gives no Big Rip. This is why the simple Model 6 condition
$`w\ge-1`$ cannot be applied regardless of the sign of $`w-w_{\mathrm{dm}}+\delta`$.

### Models 9–10: full future classification

Under the Model 9–10 assumptions stated above, without additionally imposing
global component positivity:

| Model | Conditions | Future classification |
|---:|---|---|
| 9 | $`w<-1`$ and $`\delta>-1-w_{\mathrm{dm}}`$ | Big Rip |
| 9 | $`w\ge-1`$ or $`\delta\le-1-w_{\mathrm{dm}}`$ | No Big Rip on the corresponding future-regular solution |
| 10 | $`\delta\le1+w`$ | No Big Rip |
| 10 | $`1+w<\delta<2/3`$ | Big Rip |
| 10 | $`\delta=2/3`$ | Big Rip on the exceptional limiting solution |
| 10 | $`\delta>2/3`$ | Finite-scale-factor singularity, not a Big Rip at infinity |

The last row is classified independently, although the selected global
real-domain prior already excludes it from sampling.
`ALLOW_BIG_RIP=False` accepts only `no_big_rip`; unclassified futures,
negative future total density and finite-scale singularities also fail this filter.

## Exceptional limits and implementation notes

The code evaluates equivalent, numerically stable expressions instead of
substituting zero into the denominators of the generic formulas above.
No production code is changed by this presentation.

- **Models 1–3:** repeated real roots and $`w=w_{\mathrm{dm}}`$ are evaluated
  with the finite matrix-exponential limit where the selected real domain permits.
  The generic density expressions above are not to be evaluated as $`0/0`$.
- **Model 4**, at $`\delta+w-w_{\mathrm{dm}}=0`$:

```math
\rho_{\mathrm{dm}}=\rho_{\mathrm{dm},0}a^{-3(1+w)},\qquad
\rho_{\mathrm{de}}=
[\rho_{\mathrm{de},0}-3\delta\rho_{\mathrm{dm},0}\ln a]a^{-3(1+w)}.
```

- **Model 5**, at $`\delta+w-w_{\mathrm{dm}}=0`$:

```math
\rho_{\mathrm{dm}}=
[\rho_{\mathrm{dm},0}+3\delta\rho_{\mathrm{de},0}\ln a]a^{-3(1+w_{\mathrm{dm}})},\qquad
\rho_{\mathrm{de}}=\rho_{\mathrm{de},0}a^{-3(1+w_{\mathrm{dm}})}.
```

- **Model 6**, at $`\delta+w-w_{\mathrm{dm}}=0`$, has constant density ratio:

```math
\rho_{\mathrm{dm}}=\rho_{\mathrm{dm},0}
a^{-3\left(1+\frac{w_{\mathrm{dm}}r_0+w}{1+r_0}\right)},\qquad
\rho_{\mathrm{de}}=\rho_{\mathrm{de},0}
a^{-3\left(1+\frac{w_{\mathrm{dm}}r_0+w}{1+r_0}\right)}.
```

- **Models 7–8:** the sufficient real branches exclude the singular denominators;
  no extension through $`\delta=w-w_{\mathrm{dm}}`$ is claimed.
- **Model 9:** $`\delta=-2/3`$ is handled by its separate continuous limit;
  $`2(w-w_{\mathrm{dm}})+\delta(1+3w)=0`$ uses the logarithmic density-ratio solution.
- **Model 10:** $`\delta=2/3`$ is handled by its separate continuous limit;
  $`2(w-w_{\mathrm{dm}})+\delta(1+3w_{\mathrm{dm}})=0`$ uses the logarithmic density-ratio solution.

All ten models have the uncoupled limit

```math
\begin{aligned}
\rho_{\mathrm{dm}}&=\rho_{\mathrm{dm},0}a^{-3(1+w_{\mathrm{dm}})},\qquad
\rho_{\mathrm{de}}=\rho_{\mathrm{de},0}a^{-3(1+w)},\\
H(z)&=H_0\left[\Omega_{\mathrm{bm},0}(1+z)^3+\Omega_{\mathrm{r},0}(1+z)^4
+\Omega_{\mathrm{dm},0}(1+z)^{3(1+w_{\mathrm{dm}})}
+\Omega_{\mathrm{de},0}(1+z)^{3(1+w)}\right]^{1/2}.
\end{aligned}
```

### Free/fixed EOS and doom-factor scope

| `FREE_W` | `FREE_W_DM` | Free EOS | Fixed EOS |
|---|---|---|---|
| True | False | $`w`$ | $`w_{\mathrm{dm}}=0`$ |
| False | False | none | $`w=-1`$, $`w_{\mathrm{dm}}=0`$ |
| True | True | $`w`$, $`w_{\mathrm{dm}}`$ | none |
| False | True | $`w_{\mathrm{dm}}`$ | $`w=-1`$ |

`ALLOW_DOOM_FACTOR_INSTABILITIES=False` is supported only for Models 1–8
with free $`w`$ and fixed $`w_{\mathrm{dm}}=0`$. For nonzero interaction the
implemented sufficient criteria are:

| Model | Cold-DM doom criterion |
|---:|---|
| 1 | $`\delta_{\mathrm{de}}>0`$ and $`w<-1`$ |
| 2, 3, 4, 7 | $`w<-1`$ |
| 5, 6, 8 | $`\delta(1+w)<0`$ |

Zero interaction passes the pointwise doom test. Warm/free DM EOS, fixed
$`w=-1`$ and Models 9–10 are unclassified configurations: enforcing this prior
raises a configuration error. Setting the switch to `True` disables this
optional prior; it does not establish perturbative stability.

### CLASS mapping and validation scope

In Python, `w_dm` represents $`w_{\mathrm{dm}}`$, `w` represents $`w`$,
`Omega_b` represents $`\Omega_{\mathrm{bm},0}`$, and
`Omega_m - Omega_b` supplies $`\Omega_{\mathrm{dm},0}`$.
Python density routines return densities in present critical-density units;
the formulas above use the PDF's physical-density notation.

All ten models share `Class/IDE_background`; the warm EOS is passed as
`w_dm_ide`. Its dark-sector pressure obeys

```math
\begin{aligned}
p_{\mathrm{dark}}&=w_{\mathrm{dm}}\rho_{\mathrm{dm}}+w\rho_{\mathrm{de}},\\
\frac{dp_{\mathrm{dark}}}{d\ln a}
&=-3w_{\mathrm{dm}}(1+w_{\mathrm{dm}})\rho_{\mathrm{dm}}
-3w(1+w)\rho_{\mathrm{de}}+(w_{\mathrm{dm}}-w)\frac{Q}{H}.
\end{aligned}
```

The low-redshift Python default is $`\Omega_{\mathrm{r},0}=0`$. CLASS retains
calibrated photons and neutrinos, including massive-neutrino evolution.
Comparison with CLASS must match the density/radiation normalization; massive
neutrinos cannot be represented by a single constant $`a^{-4}`$ component at all epochs.
This is a homogeneous-background and standard-thermodynamics sound-horizon
extension, not a warm IDE perturbation or particle free-streaming calculation.

See [validation evidence](../validation/WARM_DM.md),
[convergence policy](../reproducibility/SAIP2026_IDE/CONVERGENCE.md), and
[the known CLASS BBN-table limitation](../validation/PERFORMANCE_CHECKPOINT.md).
The original eight-model reference is cited in the [repository README](../README.md).
The companion LaTeX was used to transcribe the generic formulas; the clarified
PDF supplies the updated Models 9–10 condition tables and future classifications.
