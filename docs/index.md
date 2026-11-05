# Welcome to Kosmulator's documentation!

```{image} _static/kosmulator_logo.png
:alt: Kosmulator logo
:width: 260px
:align: center
```

This is the documentation of Kosmulator {{ version }}.

Kosmulator is a Python package that uses **zeus** and **emcee (the MCMC Hammer)** to run efficient, vectorised Markov chain Monte Carlo (MCMC) inference for **modified gravity** and **alternative cosmological models**.

It is written so that a model can be tested against data with very little code: you write the expansion rate $E(z) = H(z)/H_0$ of your model in **one Python function**, give it a name and a list of parameters, choose the data sets, and run. For every model and every combination of data, Kosmulator then:

1. adds the parameters the data need ($H_0$, $r_d$, supernova calibrations, CMB parameters and nuisances) with their priors;
2. finds a starting point and samples the posterior with zeus or emcee;
3. stops once the chains have converged (chain length against the autocorrelation time, effective sample size and split-$\hat R$);
4. finds the best fit and computes $\chi^2$, AIC, BIC, AICc, DIC and WAIC;
5. compares the model with a reference model (usually ΛCDM), with $\Delta\chi^2$ as a significance and a plain-language interpretation of each criterion;
6. draws corner plots, best-fit plots and convergence plots, and writes LaTeX-ready tables; and
7. lists exactly what to cite for the data and codes that were used.

Kosmulator includes 32 data sets:

1. Type Ia supernovae: JLA, Pantheon, Pantheon+, Pantheon+SH0ES, Union3 and DES-Y5;
2. baryon acoustic oscillations from DESI DR1 and DR2;
3. cosmic chronometers and an $H(z)$ compilation;
4. growth of structure, $f\sigma_8(z)$ and $f(z)$;
5. Big Bang nucleosynthesis: primordial deuterium and a BBN prior on $\Omega_b h^2$;
6. the cosmic microwave background: Planck 2018 and PR4, ACT DR6, SPT-3G D1, the P-ACT and CMB-SPA combinations, CMB lensing, and DESI DR2's compressed CMB.

For the CMB, Kosmulator builds and caches its own copy of CLASS for each model, so a modified CLASS can sit next to the standard one.

Kosmulator reproduces published analyses: with DESI DR2 and Pantheon+ it gives the same $w_0w_a$CDM preference over ΛCDM as the DESI collaboration ([Tutorial 1](tutorials/desi_w0wa.md)).

## Contents

```{toctree}
:maxdepth: 2

getting_started
running
tags
options
outputs
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/desi_w0wa
tutorials/own_model
tutorials/cmb_model
```

```{toctree}
:maxdepth: 1
:caption: Reference

reference/advanced_installation
reference/structure
reference/troubleshooting
reference/citing
reference/licence
```

## Citing Kosmulator

If you use Kosmulator in your work, please cite the software paper (Hough, Rugg, Sahlu & Abebe 2026, SAGS2025 Proceedings, pp. 7 - 19, [doi:10.62869/001c.172550](https://doi.org/10.62869/001c.172550); [arXiv:2602.08424](https://arxiv.org/abs/2602.08424)) and the original paper of the code (Hough, Abebe & Ferreira 2020, [EPJC 80, 787](https://doi.org/10.1140/epjc/s10052-020-8342-7)), together with the original papers of the data you use. Every run lists these for you; see [Citing Kosmulator](reference/citing.md).
