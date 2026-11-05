# Data set tags

Every data set is chosen by a **tag** in the `observations` list of `Kosmulator.py`. All data files are included in `Observations/` (only the Planck PR4 CamSpec data are downloaded on first use). `Observations/DATA_SOURCES.md` lists every file, where it comes from, its licence status and the papers to cite.

::::{admonition} Cite the original papers
:class: warning
Remember to cite the **original papers** of the data you use. Every run writes `citations.md` and `citations.bib` with exactly the references for the data and codes of that run (see [Citing Kosmulator](reference/citing.md)).
::::

## Late-time data

### Type Ia supernovae

| Tag | Data |
|---|---|
| `JLA` | SDSS-II/SNLS3 Joint Light-curve Analysis, 740 SNe (Betoule et al. 2014), the official likelihood: $\alpha$ and $\beta$ sampled, absolute magnitudes (with the host-mass step) fitted analytically, covariance rebuilt for each $\alpha, \beta$ |
| `JLA_legacy` | The old 359-SN file used as "JLA" before October 2026, kept only to reproduce earlier runs |
| `Pantheon` | Pantheon, 1048 SNe (Scolnic et al. 2018), statistical errors only unless `PANTHEON_SYS_COV = "full"` |
| `PantheonP` | Pantheon+ with its statistical and systematic covariance (Brout et al. 2022) |
| `PantheonPS` | Pantheon+ with the SH0ES Cepheid calibration (Riess et al. 2022): an $H_0$ anchor |
| `Union3` | Union3 / UNITY1.5 binned distance moduli (Rubin et al. 2025) |
| `DESY5` | DES-SN5YR Hubble diagram (DES Collaboration 2024) |

::::{dropdown} Supernovae and $H_0$
Uncalibrated supernovae (all tags except `PantheonPS`) measure only the shape of the distance-redshift relation: the magnitude offset is marginalised analytically and absorbs $5\log_{10} H_0$. Without BAO or an $H_0$ anchor (`CC`, `OHD`, `PantheonPS` or CMB data) in the group, $H_0$ is therefore not sampled by default (`sample_H0_uncalibrated_SNe` in [Configuring a run](options.md#physics-and-likelihood-switches)). The $\chi^2$, AIC, BIC and degrees of freedom do not depend on this switch.
::::

### Baryon acoustic oscillations

| Tag | Data |
|---|---|
| `DESI_DR2` | DESI DR2, 13 measurements with the published covariance (Abdul Karim et al. 2025) |
| `DESI_DR1` | DESI DR1, all 12 measurements with the published covariance (DESI 2024 VI) |
| `BAO` | The 10 anisotropic DESI DR1 measurements ($D_M/r_d$ and $D_H/r_d$), without the BGS and QSO $D_V/r_d$ points |

::::{admonition} Renamed tags
:class: note
Before October 2026 the `BAO` and `DESI_DR1` tags were the other way round: older results labelled `BAO` are the 12-point DR1 set, and older `DESI_DR1` results the 10-point set.
::::

::::{dropdown} How Kosmulator treats the sound horizon $r_d$
BAO measure distances in units of $r_d$. What happens to $r_d$ depends on the rest of the group:

| Group | $r_d$ |
|---|---|
| BAO with an early-time calibrator (`BBN_DH`, `BBN_DH_AlterBBN`, `BBN_PryMordial`, `ThetaStar_CMB` or primary CMB) | computed with CLASS from the background at every point (`rd_background`), not sampled |
| BAO alone, or with uncalibrated supernovae and/or growth data only | fixed at 147.5 Mpc; $H_0$ then measures $h\,r_d$, and $c/(H_0 r_d)$ is reported as well |
| BAO with an $H_0$ anchor (`CC`, `OHD`, `PantheonPS`, CMB lensing) but no early-time calibrator | sampled as a free parameter |

The start-up log says which case applies to each group, and the statistics tables carry the label (for example `[rd: fixed 147.5 Mpc]`).
::::

### Expansion rate and growth of structure

| Tag | Data |
|---|---|
| `CC` | Cosmic chronometers $H(z)$: Kosmulator's compilation, every row checked against its paper (source after `#` in `CC.dat`), with the correlations of the DESI points of Loubser (2025) |
| `OHD` | $H(z)$ compilation of Sharov & Vasiliev (2018): 31 chronometer and 20 BAO points |
| `f_sigma_8` | $f\sigma_8(z)$ from redshift-space distortions, 66 points (Skara & Perivolaropoulos 2020) |
| `f` | Growth rate $f(z)$: Perenon et al. (2019) and Avila et al. (2022) |

The growth data use `growth_model` ([Configuring a run](options.md#physics-and-likelihood-switches)): either $f = \Omega_m(z)^\gamma$ with $\gamma$ sampled, or the model's own linear growth with its $G_\mathrm{eff}/G$.

## Early-time data

### Big Bang nucleosynthesis

| Tag | Data |
|---|---|
| `BBN_DH` | Primordial deuterium, the 12 quasar systems of the PDG 2025 review; predicted D/H from a power law fitted to the PRyMordial table |
| `BBN_DH_AlterBBN` | The same data, with D/H predicted by AlterBBN (a grid made with AlterBBN v1.4 is included) |
| `BBN_PryMordial` | A Gaussian prior $\Omega_b h^2 = 0.02218 \pm 0.00055$ (Schöneberg 2024) |

Every BBN backend adds a 4.1% theory error (nuclear rates). BBN uses the run's `nu_N_eff`, the same $N_\mathrm{eff}$ as CLASS.

::::{dropdown} AlterBBN and the shipped grid
`BBN_DH_AlterBBN` reads `Observations/BBN/bbn_grid.npz` (AlterBBN v1.4, $\Omega_b h^2$ = 0.0150 - 0.0310, $N_\mathrm{eff}$ = 3.044 and 3.046, $\tau_n$ = 879.4 s), so AlterBBN itself is needed only for live calls or a new grid. Priors outside the grid (another $N_\mathrm{eff}$, a sampled $N_\mathrm{eff}$ or $\tau_n$, a wider $\Omega_b h^2$ range) make Kosmulator build a new grid with AlterBBN, which replaces that file (`git checkout Observations/BBN/bbn_grid.npz` restores it). Without AlterBBN it falls back to the approximation and says so. Installing AlterBBN is described in [Advanced installation](reference/advanced_installation.md).
::::

### Cosmic microwave background

**Planck**

| Tag | Data |
|---|---|
| `CMB_hil` | Planck 2018 high-$\ell$ TT,TE,EE (plik; plik_lite with `cmb_nuisance = "lite"`) |
| `CMB_hil_TT` | Planck 2018 high-$\ell$ TT |
| `CMB_hil_PR4` | Planck PR4 (NPIPE) high-$\ell$ TT,TE,EE with CamSpec, DESI DR2's baseline; data downloaded on first use (0.5 GB) |
| `CMB_lowl_TT` | Planck 2018 low-$\ell$ TT (Commander, $\ell$ = 2 - 29) |
| `CMB_lowl` | Planck 2018 low-$\ell$ EE (SimAll) |
| `CMB_lowl_sroll2` | Planck low-$\ell$ EE from the SRoll2 maps |
| `CMB_lensing` | Planck 2018 lensing reconstruction (the CMB-marginalised version when no primary CMB is in the group) |

Planck's "TT,TE,EE+lowE" is `["CMB_hil", "CMB_lowl_TT", "CMB_lowl"]`; Kosmulator warns when a group has `CMB_hil` or `CMB_hil_TT` without `CMB_lowl_TT`.

**Ground-based CMB and combinations**

| Tag | Data |
|---|---|
| `CMB_ACT` | ACT DR6 TT/TE/EE (ACT-lite, foreground-marginalised) |
| `CMB_SPT` | SPT-3G D1 T&E (SPT-lite, foreground-marginalised) |
| `CMB_planck_ACTcut` | Planck 2018 plik_lite cut at the ACT DR6 multipoles |
| `CMB_PACT` | P-ACT: `CMB_lowl_TT` + `CMB_lowl_sroll2` + `CMB_planck_ACTcut` + `CMB_ACT` |
| `CMB_SPA` | CMB-SPA: P-ACT + `CMB_SPT` |
| `CMB_lensing_ACTPlanck` | ACT DR6 + Planck PR4 lensing power spectrum |
| `CMB_lensing_SPT` | SPT-3G MUSE lensing |

`CMB_ACT` and `CMB_SPT` alone get a Gaussian prior on $\tau_\mathrm{reio}$ unless a low-$\ell$ EE likelihood is in the group.

**Compressed CMB**

| Tag | Data |
|---|---|
| `ThetaStar_CMB` | DESI DR2's compressed CMB: a Gaussian prior on $(\theta_*, \omega_b, \omega_{bc})$ from the Planck PR4 early-universe chains, with $\theta_*$ from each model's own background. It calibrates $r_d$ like the BBN prior. |

::::{dropdown} What CMB data add to a run
Groups with CMB data sample $H_0$, the physical densities $\Omega_b h^2$ and $\Omega_\mathrm{dm} h^2$ (in place of $\Omega_m$, which is then derived), $\ln(10^{10}A_s)$, $n_s$ and $\tau_\mathrm{reio}$ next to the model's own parameters, plus the likelihoods' nuisance parameters (Planck's foregrounds and calibrations with their priors, the ACT and SPT calibrations). The CMB spectra come from one lensed CLASS run per point, with the model's own CLASS build (`Class/<model>`); a model fitted to CMB data needs a CMB helper function (see [Your model with CMB data](tutorials/cmb_model.md)). These groups always run emcee, through the worker pool.
::::

## Example combinations

```python
observations = [
    ["DESI_DR2", "PantheonP"],                        # BAO + supernovae (r_d fixed, h r_d measured)
    ["DESI_DR2", "PantheonP", "BBN_PryMordial"],      # ... with BBN calibrating r_d
    ["CC", "DESI_DR2", "PantheonPS"],                 # with H_0 anchors (r_d free)
    ["DESI_DR2", "ThetaStar_CMB"],                    # DESI DR2 with the compressed CMB
    ["CMB_hil", "CMB_lowl_TT", "CMB_lowl"],           # Planck 2018 TT,TE,EE+lowE
    ["CMB_SPA", "CMB_lensing_ACTPlanck", "DESI_DR2"], # CMB-SPA + lensing + DESI DR2
]
```
