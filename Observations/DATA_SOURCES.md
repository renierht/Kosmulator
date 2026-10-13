# Data sources, licences and what to cite

Written from `Kosmulator_main/citations.py` (`python -m Kosmulator_main.citations --write-data-sources`); every reference was checked against its arXiv page and the journal record (13 October 2026). The BibTeX of every reference is in `Observations/references.bib` (its key in brackets after each reference). Each run also writes `Statistical_analysis_tables/<suffix>/citations.md` with what that run used, and its BibTeX in `citations.bib`.

Kosmulator includes every data set it supports, with its source and the papers to cite, as is common practice for published cosmological data. Several sources have no licence file; if a collaboration asks for its data to be removed, it will be (the SPT-3G files can be restored with `python -m Kosmulator_main.get_spt_data`). Cite Kosmulator and the original papers of the data you use.

## Kosmulator

- R. T. Hough, R. Rugg, S. Sahlu, A. Abebe (2026), Kosmulator: A Python framework for cosmological inference with MCMC, arXiv:2602.08424 [`hough26`]
- R. T. Hough, A. Abebe, S. E. S. Ferreira (2020), Viability tests of f(R)-gravity models with Supernovae Type 1A data, Eur. Phys. J. C 80, 787, arXiv:1911.05983, doi:10.1140/epjc/s10052-020-8342-7 [`hough20`]

## Likelihood data

| Tag(s) | Data | Files in `Observations/` | Source | Licence status |
|---|---|---|---|---|
| `CC` | Cosmic chronometers H(z) | `CC.dat`, `CC_corr.txt` | Kosmulator's compilation; every row checked against its paper (source after '#' in CC.dat); correlations of the Loubser (2025) DESI points in CC_corr.txt | values compiled from the published papers |
| `OHD` | H(z) compilation (31 chronometer + 20 BAO points) | `OHD.dat` | the compilation of Sharov & Vasiliev (2018), taken from that paper | values compiled from the published papers |
| `JLA` | SDSS-II/SNLS3 Joint Light-curve Analysis (740 SNe) | `JLA/` | CosmoMC/Cobaya distribution, github.com/CobayaSampler/sn_data (JLA/, commit 61d9643); release page https://supernovae.in2p3.fr/sdss_snls_jla/ReadMe.html | no licence file at the source |
| `JLA_legacy` | Old 359-SN distance moduli (kept to reproduce earlier runs) | `JLA_legacy.dat` | taken from the papers cited | values compiled from the published papers |
| `Pantheon` | Pantheon SNe Ia (1048) | `Pantheon.dat` | the Pantheon sample of Scolnic et al. (2018), taken from that paper's release (github.com/dscolnic/Pantheon) | no licence file at the source |
| `PantheonP`, `PantheonPS` | Pantheon+ (PantheonPS: with the SH0ES Cepheid calibration) | `PantheonP.dat`, `PantheonP.cov` | Pantheon+ release, github.com/PantheonPlusSH0ES/DataRelease (Pantheon+SH0ES.dat and its STAT+SYS covariance; Kosmulator's copy: format matches) | no licence file at the source |
| `DESY5` | DES-SN5YR Hubble diagram | `DESY5.dat`, `DESY5_covsys_000.txt` | DES-SN5YR, taken from the release of the papers cited, github.com/des-science/DES-SN5YR (DES-SN5YR_HD.csv and covsys_000) | no licence file at the source |
| `Union3` | Union3 / UNITY1.5 binned distance moduli | `Union3.txt`, `Union3_mag_covmat.txt` | Union3 of Rubin et al. (2025), taken from that paper's release, github.com/rubind/union3_release | MIT (github.com/rubind/union3_release) |
| `BAO` | DESI DR1 BAO, the 10 anisotropic points | `BAO.txt`, `BAO_invcov.txt` | values of DESI 2024 VI (Table 1), two decimals; see BAO.txt's header | CC BY 4.0 (DESI data licence, https://data.desi.lbl.gov/doc/acknowledgments/) |
| `DESI_DR1` | DESI DR1 BAO, all 12 measurements | `DESI_DR1.txt`, `DESI_DR1_cov.txt` | DESI 2024 likelihood release as distributed in github.com/CobayaSampler/bao_data (desi_2024_gaussian_bao_ALL_GCcomb); see DESI_DR1.txt's header | CC BY 4.0 (DESI data licence, https://data.desi.lbl.gov/doc/acknowledgments/) |
| `DESI_DR2` | DESI DR2 BAO | `DESI_DR2_synced.txt`, `DESI_DR2_covtot.txt` | the values and covariance of DESI DR2 Results II (Abdul Karim et al. 2025), taken from that paper | values of DESI DR2 Results II; DESI's CC BY 4.0 data licence covers its public data releases (EDR, DR1) |
| `f` | Growth rate f(z) | `f.dat` | rows 1 - 3 Perenon et al. (2019), Table 1; rows 4 - 14 Avila et al. (2022), Table 1 (f.dat header) | values compiled from the published papers |
| `f_sigma_8` | f sigma_8(z) from redshift-space distortions (66 points) | `f_sigma_8.dat` | Skara & Perivolaropoulos (2020), Appendix B, Table VI, an update of Kazantzidis & Perivolaropoulos (2018), Table II (f_sigma_8.dat header) | values compiled from the published papers |
| `BBN_DH`, `BBN_DH_AlterBBN` | Primordial deuterium D/H (the 12 quasar systems of the PDG 2025 review) | `constants.BBN_DH_PDG_SYSTEMS`, `BBN/ (AlterBBN grids made by Kosmulator)` | PDG BBN review (2025 update); predicted D/H(omega_b) from a power law fitted to the PRyMordial table (BBN_DH) or from AlterBBN (BBN_DH_AlterBBN) | values compiled from the published papers |
| `BBN_PryMordial`, `BBN_prior` | BBN prior on Omega_b h^2 = 0.02218 +- 0.00055 (LCDM) | `Kosmulator_main/Config.py` | the value of Schoeneberg (2024) (PDG abundances, PRyMordial, conservative nuclear rates) | values compiled from the published papers |
| `CMB_hil`, `CMB_hil_TT` | Planck 2018 high-l TT,TE,EE / TT (plik, plik_lite) | `plik_rd12_HM_v22b_TTTEEE.clik`, `plik_rd12_HM_v22_TT.clik`, `plik_lite_v22_TTTEEE.clik`, `plik_lite_v22_TT.clik` | Planck Legacy Archive, COM_Likelihood_Data-baseline_R3.00 | no licence stated (ESA Planck Legacy Archive) |
| `CMB_lowl`, `CMB_lowl_TT` | Planck 2018 low-l EE (SimAll) and TT (Commander) | `simall_100x143_offlike5_EE_Aplanck_B.clik`, `commander_dx12_v3_2_29.clik` | Planck Legacy Archive, COM_Likelihood_Data-baseline_R3.00 | no licence stated (ESA Planck Legacy Archive) |
| `CMB_lensing` | Planck 2018 lensing (raw and CMB-marginalised) | `smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing`, `smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_CMBmarged.clik_lensing` | Planck Legacy Archive, COM_Likelihood_Data-baseline_R3.00 | no licence stated (ESA Planck Legacy Archive) |
| `CMB_lowl_sroll2` | Planck low-l EE from the SRoll2 maps (probability table) | `sroll2_prob_table.txt` | Cobaya's planck_native_data release (planck_sroll2_lowE.zip), github.com/CobayaSampler/planck_native_data | no licence file at the source |
| `CMB_planck_ACTcut` | Planck 2018 plik_lite cut at the ACT DR6 multipoles | `plik_lite_v22_TTTEEE.clik` | Planck Legacy Archive; the cut as ACT DR6's PlanckActCut | no licence stated (ESA Planck Legacy Archive) |
| `CMB_ACT` | ACT DR6 TT/TE/EE (ACT-lite, foreground-marginalised) | `candl/ACT_DR6_CMB_only_v0/` | candl_data 2.0.0 (commit c4eb8fb), Observations/candl/README.md | MIT (candl_data; its licence is Observations/candl/LICENCE_candl_data.txt) |
| `CMB_SPT` | SPT-3G D1 T&E (SPT-lite, foreground-marginalised) | `candl/SPT3G_D1_TnE_lite/` | spt_candl_data 3.0.1 (commit f22611a), Observations/candl/README.md | no licence file (spt_candl_data, commit f22611a) |
| `CMB_lensing_ACTPlanck` | ACT DR6 + Planck PR4 lensing power spectrum (actplanck_baseline) | `act_dr6_lenslike/` | LAMBDA, ACT_dr6_likelihood_v1.2.tgz (Observations/act_dr6_lenslike/README.md); the binned corrections file is Kosmulator's | no licence stated; the LAMBDA page asks to cite Qu et al. and Madhavacheril et al. |
| `CMB_lensing_SPT` | SPT-3G MUSE lensing (phi phi band powers) | `spt3g_muse_lensing/` | muse_3g_like_march_2025.zip, https://pole.uchicago.edu/public/data/ge25/ (Observations/spt3g_muse_lensing/README.md) | no licence file (muse_3g_like_march_2025.zip) |

Combinations: `CMB_PACT` = `CMB_lowl_TT` + `CMB_lowl_sroll2` + `CMB_planck_ACTcut` + `CMB_ACT`; `CMB_SPA` = the same + `CMB_SPT`. Cite their parts.

## What to cite, per tag

### CC: Cosmic chronometers H(z)

- with the option `constants.CC_SYS_COV` also:
  - M. Moresco et al. (2020), Setting the Stage for Cosmic Chronometers. II. Impact of Stellar Population Synthesis Models Systematics and Full Covariance Matrix, ApJ 898, 82, arXiv:2003.07362, doi:10.3847/1538-4357/ab9eb0 [`moresco20`]
- the papers of the CC.dat rows used:
  - Zhang et al. 2014, arXiv:1207.4541
  - Jimenez et al. 2003, astro-ph/0302560
  - Wang, Lei & Fan 2026, arXiv:2601.07345
  - Simon et al. 2005, astro-ph/0412269
  - Moresco et al. 2012, arXiv:1201.3609
  - Moresco et al. 2016, arXiv:1601.01701
  - Loubser 2025, arXiv:2511.02730
  - Ratsimbazafy et al. 2017, arXiv:1702.00418
  - Stern et al. 2010, arXiv:0907.3149
  - Loubser et al. 2025, arXiv:2506.03836
  - Moresco 2015, arXiv:1503.01116
- Note: Cite the paper of every row used (listed per run, and in CC.dat).

### OHD: H(z) compilation (31 chronometer + 20 BAO points)

- G. S. Sharov, V. O. Vasiliev (2018), How predictions of cosmological models depend on Hubble parameter data sets, Math. Model. Geom. 6, 1, arXiv:1807.07323, doi:10.26456/mmg/2018-611 [`sharov18`]
- Note: Also cite the original papers of the points, listed in Sharov & Vasiliev (2018).

### JLA: SDSS-II/SNLS3 Joint Light-curve Analysis (740 SNe)

- M. Betoule et al. (2014), Improved cosmological constraints from a joint analysis of the SDSS-II and SNLS supernova samples, A&A 568, A22, arXiv:1401.4064, doi:10.1051/0004-6361/201423413 [`betoule14`]

### JLA_legacy: Old 359-SN distance moduli (kept to reproduce earlier runs)

- M. Hicken et al. (2009), CfA3: 185 Type Ia Supernova Light Curves from the CfA, ApJ 700, 331, arXiv:0901.4787, doi:10.1088/0004-637X/700/1/331 [`hicken09`]
- J. D. Neill et al. (2009), The Local Hosts of Type Ia Supernovae, ApJ 707, 1449, arXiv:0911.0690, doi:10.1088/0004-637X/707/2/1449 [`neill09`]
- A. Conley et al. (2011), Supernova Constraints and Systematic Uncertainties from the First Three Years of the Supernova Legacy Survey, ApJS 192, 1, arXiv:1104.1443, doi:10.1088/0067-0049/192/1/1 [`conley11`]
- Note: Prefer JLA.

### Pantheon: Pantheon SNe Ia (1048)

- D. M. Scolnic et al. (2018), The Complete Light-curve Sample of Spectroscopically Confirmed SNe Ia from Pan-STARRS1 and Cosmological Constraints from the Combined Pantheon Sample, ApJ 859, 101, arXiv:1710.00845, doi:10.3847/1538-4357/aab9bb [`scolnic18`]

### PantheonP, PantheonPS: Pantheon+ (PantheonPS: with the SH0ES Cepheid calibration)

- D. Scolnic et al. (2022), The Pantheon+ Analysis: The Full Data Set and Light-curve Release, ApJ 938, 113, arXiv:2112.03863, doi:10.3847/1538-4357/ac8b7a [`scolnic22`]
- D. Brout et al. (2022), The Pantheon+ Analysis: Cosmological Constraints, ApJ 938, 110, arXiv:2202.04077, doi:10.3847/1538-4357/ac8e04 [`brout22`]
- with `PantheonPS` also:
  - A. G. Riess et al. (2022), A Comprehensive Measurement of the Local Value of the Hubble Constant with 1 km/s/Mpc Uncertainty from the Hubble Space Telescope and the SH0ES Team, ApJL 934, L7, arXiv:2112.04510, doi:10.3847/2041-8213/ac5c5b [`riess22`]

### DESY5: DES-SN5YR Hubble diagram

- DES Collaboration, T. M. C. Abbott et al. (2024), The Dark Energy Survey: Cosmology Results with ~1500 New High-redshift Type Ia Supernovae Using the Full 5 yr Data Set, ApJL 973, L14, arXiv:2401.02929, doi:10.3847/2041-8213/ad6f9f [`des24`]
- B. O. Sanchez et al. (2024), The Dark Energy Survey Supernova Program: Light Curves and 5 Yr Data Release, ApJ 975, 5, arXiv:2406.05046, doi:10.3847/1538-4357/ad739a [`sanchez24`]

### Union3: Union3 / UNITY1.5 binned distance moduli

- D. Rubin et al. (2025), Union Through UNITY: Cosmology with 2,000 SNe Using a Unified Bayesian Framework, ApJ 986, 231, arXiv:2311.12098, doi:10.3847/1538-4357/adc0a5 [`rubin25`]

### BAO: DESI DR1 BAO, the 10 anisotropic points

- DESI Collaboration, A. G. Adame et al. (2025), DESI 2024 VI: Cosmological Constraints from the Measurements of Baryon Acoustic Oscillations, JCAP 02 (2025) 021, arXiv:2404.03002, doi:10.1088/1475-7516/2025/02/021 [`desi24_VI`]
- DESI Collaboration, M. Abdul Karim et al. (2026), Data Release 1 of the Dark Energy Spectroscopic Instrument, AJ 171, 285, arXiv:2503.14745, doi:10.3847/1538-3881/ae4c43 [`desi_dr1`]
- Acknowledgement: DESI's data licence (CC BY 4.0, for its public data releases) asks publications that use DESI data to cite the paper of the data release used (DR1: DESI Collaboration 2026, AJ 171, 285) and to include DESI's acknowledgement text, given at https://data.desi.lbl.gov/doc/acknowledgments/ ; whoever redistributes the data says what was changed (Kosmulator's DESI files: see their headers).

### DESI_DR1: DESI DR1 BAO, all 12 measurements

- DESI Collaboration, A. G. Adame et al. (2025), DESI 2024 VI: Cosmological Constraints from the Measurements of Baryon Acoustic Oscillations, JCAP 02 (2025) 021, arXiv:2404.03002, doi:10.1088/1475-7516/2025/02/021 [`desi24_VI`]
- DESI Collaboration, M. Abdul Karim et al. (2026), Data Release 1 of the Dark Energy Spectroscopic Instrument, AJ 171, 285, arXiv:2503.14745, doi:10.3847/1538-3881/ae4c43 [`desi_dr1`]
- Acknowledgement: DESI's data licence (CC BY 4.0, for its public data releases) asks publications that use DESI data to cite the paper of the data release used (DR1: DESI Collaboration 2026, AJ 171, 285) and to include DESI's acknowledgement text, given at https://data.desi.lbl.gov/doc/acknowledgments/ ; whoever redistributes the data says what was changed (Kosmulator's DESI files: see their headers).

### DESI_DR2: DESI DR2 BAO

- DESI Collaboration, M. Abdul Karim et al. (2025), DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints, Phys. Rev. D 112, 083515, arXiv:2503.14738, doi:10.1103/tr6y-kpc6 [`desi25_dr2`]
- Note: DESI_DR2.txt holds the same values with another error column and is not read.
- Acknowledgement: DESI's data licence (CC BY 4.0, for its public data releases) asks publications that use DESI data to cite the paper of the data release used (DR1: DESI Collaboration 2026, AJ 171, 285) and to include DESI's acknowledgement text, given at https://data.desi.lbl.gov/doc/acknowledgments/ ; whoever redistributes the data says what was changed (Kosmulator's DESI files: see their headers).

### f: Growth rate f(z)

- L. Perenon, J. Bel, R. Maartens, A. de la Cruz-Dombriz (2019), Optimising growth of structure constraints on modified gravity, JCAP 06 (2019) 020, arXiv:1901.11063, doi:10.1088/1475-7516/2019/06/020 [`perenon19`]
- F. Avila, A. Bernui, A. Bonilla, R. C. Nunes (2022), Inferring S8(z) and gamma(z) with cosmic growth rate measurements using machine learning, Eur. Phys. J. C 82, 594, arXiv:2201.07829, doi:10.1140/epjc/s10052-022-10561-0 [`avila22`]
- Note: Also cite the survey papers of the points (listed in those tables).

### f_sigma_8: f sigma_8(z) from redshift-space distortions (66 points)

- F. Skara, L. Perivolaropoulos (2020), Tension of the E_G statistic and RSD data with Planck/LCDM and implications for weakening gravity, Phys. Rev. D 101, 063521, arXiv:1911.10609, doi:10.1103/PhysRevD.101.063521 [`skara20`]
- L. Kazantzidis, L. Perivolaropoulos (2018), Evolution of the f sigma_8 tension with the Planck15/LCDM determination and implications for modified gravity theories, Phys. Rev. D 97, 103503, arXiv:1803.01337, doi:10.1103/PhysRevD.97.103503 [`kazantzidis18`]
- Note: Also cite the survey papers of the points (listed in Table VI).

### BBN_DH, BBN_DH_AlterBBN: Primordial deuterium D/H (the 12 quasar systems of the PDG 2025 review)

- B. D. Fields, P. Molaro, S. Sarkar, Big Bang Nucleosynthesis, review in S. Navas et al. (Particle Data Group), Phys. Rev. D 110, 030001 (2024) and 2025 update, doi:10.1103/PhysRevD.110.030001 [`pdg_bbn`]
- R. J. Cooke, M. Pettini, R. A. Jorgenson, M. T. Murphy, C. C. Steidel (2014), Precision measures of the primordial abundance of deuterium, ApJ 781, 31, arXiv:1308.3240, doi:10.1088/0004-637X/781/1/31 [`cooke14`]
- Note: Also cite the papers of the 12 D/H systems (listed in the PDG review); the code that predicted D/H is listed under Codes.

### BBN_PryMordial, BBN_prior: BBN prior on Omega_b h^2 = 0.02218 +- 0.00055 (LCDM)

- N. Schoeneberg (2024), The 2024 BBN baryon abundance update, JCAP 06 (2024) 006, arXiv:2401.15054, doi:10.1088/1475-7516/2024/06/006 [`schoneberg24`]
- A.-K. Burns, T. M. P. Tait, M. Valli (2024), PRyMordial: The First Three Minutes, Within and Beyond the Standard Model, Eur. Phys. J. C 84, 86, arXiv:2307.07061, doi:10.1140/epjc/s10052-024-12442-0 [`burns24`]

### CMB_hil, CMB_hil_TT: Planck 2018 high-l TT,TE,EE / TT (plik, plik_lite)

- Planck Collaboration, N. Aghanim et al. (2020), Planck 2018 results. V. CMB power spectra and likelihoods, A&A 641, A5, arXiv:1907.12875, doi:10.1051/0004-6361/201936386 [`planck18_V`]
- Planck Collaboration, N. Aghanim et al. (2020), Planck 2018 results. VI. Cosmological parameters, A&A 641, A6, arXiv:1807.06209, doi:10.1051/0004-6361/201833910 [`planck18_VI`]

### CMB_lowl, CMB_lowl_TT: Planck 2018 low-l EE (SimAll) and TT (Commander)

- Planck Collaboration, N. Aghanim et al. (2020), Planck 2018 results. V. CMB power spectra and likelihoods, A&A 641, A5, arXiv:1907.12875, doi:10.1051/0004-6361/201936386 [`planck18_V`]

### CMB_lensing: Planck 2018 lensing (raw and CMB-marginalised)

- Planck Collaboration, N. Aghanim et al. (2020), Planck 2018 results. VIII. Gravitational lensing, A&A 641, A8, arXiv:1807.06210, doi:10.1051/0004-6361/201833886 [`planck18_VIII`]

### CMB_lowl_sroll2: Planck low-l EE from the SRoll2 maps (probability table)

- L. Pagano et al. (2020), Reionization optical depth determination from Planck HFI data with ten percent accuracy, A&A 635, A99, arXiv:1908.09856, doi:10.1051/0004-6361/201936630 [`pagano20`]

### CMB_planck_ACTcut: Planck 2018 plik_lite cut at the ACT DR6 multipoles

- Planck Collaboration, N. Aghanim et al. (2020), Planck 2018 results. V. CMB power spectra and likelihoods, A&A 641, A5, arXiv:1907.12875, doi:10.1051/0004-6361/201936386 [`planck18_V`]
- T. Louis et al. (2025), The Atacama Cosmology Telescope: DR6 Power Spectra, Likelihoods and LCDM Parameters, JCAP 11 (2025) 062, arXiv:2503.14452, doi:10.1088/1475-7516/2025/11/062 [`louis25`]

### CMB_ACT: ACT DR6 TT/TE/EE (ACT-lite, foreground-marginalised)

- T. Louis et al. (2025), The Atacama Cosmology Telescope: DR6 Power Spectra, Likelihoods and LCDM Parameters, JCAP 11 (2025) 062, arXiv:2503.14452, doi:10.1088/1475-7516/2025/11/062 [`louis25`]
- E. Calabrese et al. (2025), The Atacama Cosmology Telescope: DR6 Constraints on Extended Cosmological Models, JCAP 11 (2025) 063, arXiv:2503.14454, doi:10.1088/1475-7516/2025/11/063 [`calabrese25`]
- S. Naess et al. (2025), The Atacama Cosmology Telescope: DR6 Maps, JCAP 11 (2025) 061, arXiv:2503.14451, doi:10.1088/1475-7516/2025/11/061 [`naess25`]

### CMB_SPT: SPT-3G D1 T&E (SPT-lite, foreground-marginalised)

- E. Camphuis et al. (2026), SPT-3G D1: CMB temperature and polarization power spectra and cosmology from 2019 and 2020 observations of the SPT-3G Main field, Phys. Rev. D 113, 083504, arXiv:2506.20707, doi:10.1103/7wt3-9v2y [`camphuis26`]
- L. Balkenhol (2025), Compressed 'CMB-lite' Likelihoods Using Automatic Differentiation, Open J. Astrophys., arXiv:2412.00826, doi:10.33232/001c.129886 [`balkenhol25`]
- W. Quan et al. (2026), SPT-3G D1: Maps of the millimeter-wave sky from 2019 and 2020 observations of the SPT-3G Main field, arXiv:2603.20163 [`quan26`]
- Note: SPT-lite is Appendix F of Camphuis et al.; Quan et al. (D1 maps) as spt_candl_data lists.

### CMB_lensing_ACTPlanck: ACT DR6 + Planck PR4 lensing power spectrum (actplanck_baseline)

- F. J. Qu et al. (2024), The Atacama Cosmology Telescope: A Measurement of the DR6 CMB Lensing Power Spectrum and its Implications for Structure Growth, ApJ 962, 112, arXiv:2304.05202, doi:10.3847/1538-4357/acfe06 [`qu24`]
- M. S. Madhavacheril et al. (2024), The Atacama Cosmology Telescope: DR6 Gravitational Lensing Map and Cosmological Parameters, ApJ 962, 113, arXiv:2304.05203, doi:10.3847/1538-4357/acff5f [`madhavacheril24`]
- J. Carron, M. Mirmelstein, A. Lewis (2022), CMB lensing from Planck PR4 maps, JCAP 09 (2022) 039, arXiv:2206.07773, doi:10.1088/1475-7516/2022/09/039 [`carron22`]

### CMB_lensing_SPT: SPT-3G MUSE lensing (phi phi band powers)

- F. Ge et al. (2025), Cosmology From CMB Lensing and Delensed EE Power Spectra Using 2019-2020 SPT-3G Polarization Data, Phys. Rev. D 111, 083534, arXiv:2411.06000, doi:10.1103/PhysRevD.111.083534 [`ge25`]

## Plot-only data

- Planck 2018 binned power spectra drawn on bestfit.png (`CMB_TT.dat`, `CMB_TE.dat`, `CMB_EE.dat`): Planck Legacy Archive (format of COM_PowerSpect_CMB-*-full_R3.01); no licence stated (ESA Planck Legacy Archive).
  - Planck Collaboration, N. Aghanim et al. (2020), Planck 2018 results. V. CMB power spectra and likelihoods, A&A 641, A5, arXiv:1907.12875, doi:10.1051/0004-6361/201936386 [`planck18_V`]

## Codes

### Kosmulator

- R. T. Hough, R. Rugg, S. Sahlu, A. Abebe (2026), Kosmulator: A Python framework for cosmological inference with MCMC, arXiv:2602.08424 [`hough26`]
- R. T. Hough, A. Abebe, S. E. S. Ferreira (2020), Viability tests of f(R)-gravity models with Supernovae Type 1A data, Eur. Phys. J. C 80, 787, arXiv:1911.05983, doi:10.1140/epjc/s10052-020-8342-7 [`hough20`]

### emcee (ensemble sampler)

- D. Foreman-Mackey, D. W. Hogg, D. Lang, J. Goodman (2013), emcee: The MCMC Hammer, PASP 125, 306, arXiv:1202.3665, doi:10.1086/670067 [`foremanmackey13`]

### emcee's DE move and Kosmulator's snooker move (emcee_moves = "de_snooker")

- C. J. F. ter Braak (2006), A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution: easy Bayesian computing for real parameter spaces, Stat. Comput. 16, 239, doi:10.1007/s11222-006-8769-1 [`terbraak06`]
- C. J. F. ter Braak, J. A. Vrugt (2008), Differential Evolution Markov Chain with snooker updater and fewer chains, Stat. Comput. 18, 435, doi:10.1007/s11222-008-9104-9 [`terbraak08`]

### zeus (ensemble slice sampling)

- M. Karamanis, F. Beutler, J. A. Peacock (2021), zeus: A Python implementation of Ensemble Slice Sampling for efficient Bayesian parameter inference, MNRAS 508, 3589, arXiv:2105.03468, doi:10.1093/mnras/stab2867 [`karamanis21`]
- M. Karamanis, F. Beutler (2021), Ensemble Slice Sampling, Stat. Comput. 31, 61, arXiv:2002.06212, doi:10.1007/s11222-021-10038-2 [`karamanis21b`]

### CLASS (CMB spectra; r_d)

- J. Lesgourgues (2011), The Cosmic Linear Anisotropy Solving System (CLASS) I: Overview, arXiv:1104.2932 [`lesgourgues11`]
- D. Blas, J. Lesgourgues, T. Tram (2011), The Cosmic Linear Anisotropy Solving System (CLASS) II: Approximation schemes, JCAP 07 (2011) 034, arXiv:1104.2933, doi:10.1088/1475-7516/2011/07/034 [`blas11`]

### r_d from the Eisenstein & Hu (1998) fit (CLASS could not be imported)

- D. J. Eisenstein, W. Hu (1998), Baryonic Features in the Matter Transfer Function, ApJ 496, 605, arXiv:astro-ph/9709112, doi:10.1086/305424 [`eisenstein98`]

### HMcode-2020 in CLASS (cmb_nonlinear = "hmcode")

- A. Mead, S. Brieden, T. Troester, C. Heymans (2021), HMcode-2020: Improved modelling of non-linear cosmological power spectra with baryonic feedback, MNRAS 502, 1401, arXiv:2009.01858, doi:10.1093/mnras/stab082 [`mead21`]

### HMcode 2015 in CLASS (cmb_nonlinear = "hmcode", hmcode_version = "2015")

- A. Mead, J. Peacock, C. Heymans, S. Joudaki, A. Heavens (2015), An accurate halo model for fitting non-linear cosmological power spectra and baryonic feedback models, MNRAS 454, 1958, arXiv:1505.07833, doi:10.1093/mnras/stv2036 [`mead15`]

### Halofit in CLASS (cmb_nonlinear = "halofit")

- R. Takahashi, M. Sato, T. Nishimichi, A. Taruya, M. Oguri (2012), Revising the Halofit Model for the Nonlinear Matter Power Spectrum, ApJ 761, 152, arXiv:1208.2701, doi:10.1088/0004-637X/761/2/152 [`takahashi12`]

### Planck likelihood code (clik or clipy, github.com/benabed/clipy)

- Planck Collaboration, N. Aghanim et al. (2020), Planck 2018 results. V. CMB power spectra and likelihoods, A&A 641, A5, arXiv:1907.12875, doi:10.1051/0004-6361/201936386 [`planck18_V`]

### candl (ACT and SPT likelihoods)

- L. Balkenhol, C. Trendafilova, K. Benabed, S. Galli (2024), candl: Cosmic Microwave Background Analysis with a Differentiable Likelihood, A&A 686, A10, arXiv:2401.13433, doi:10.1051/0004-6361/202449432 [`balkenhol24`]

### act_dr6_lenslike (ACT DR6 + Planck PR4 lensing likelihood code)

- F. J. Qu et al. (2024), The Atacama Cosmology Telescope: A Measurement of the DR6 CMB Lensing Power Spectrum and its Implications for Structure Growth, ApJ 962, 112, arXiv:2304.05202, doi:10.3847/1538-4357/acfe06 [`qu24`]
- M. S. Madhavacheril et al. (2024), The Atacama Cosmology Telescope: DR6 Gravitational Lensing Map and Cosmological Parameters, ApJ 962, 113, arXiv:2304.05203, doi:10.3847/1538-4357/acff5f [`madhavacheril24`]

### AlterBBN

- A. Arbey (2012), AlterBBN: A program for calculating the BBN abundances of the elements in alternative cosmologies, Comput. Phys. Commun. 183, 1822, arXiv:1106.1363, doi:10.1016/j.cpc.2012.03.018 [`arbey12`]
- A. Arbey, J. Auffinger, K. P. Hickerson, E. S. Jenssen (2020), AlterBBN v2: A public code for calculating Big-Bang nucleosynthesis constraints in alternative cosmologies, Comput. Phys. Commun. 248, 106982, arXiv:1806.11095, doi:10.1016/j.cpc.2019.106982 [`arbey20`]

### PRyMordial (the D/H table behind BBN_DH's power law)

- A.-K. Burns, T. M. P. Tait, M. Valli (2024), PRyMordial: The First Three Minutes, Within and Beyond the Standard Model, Eur. Phys. J. C 84, 86, arXiv:2307.07061, doi:10.1140/epjc/s10052-024-12442-0 [`burns24`]

### GetDist (corner plots, marginal statistics)

- A. Lewis (2025), GetDist: a Python package for analysing Monte Carlo samples, JCAP 08 (2025) 025, arXiv:1910.13970, doi:10.1088/1475-7516/2025/08/025 [`lewis19`]

## Acknowledgements the data's licences ask for

- DESI's data licence (CC BY 4.0, for its public data releases) asks publications that use DESI data to cite the paper of the data release used (DR1: DESI Collaboration 2026, AJ 171, 285) and to include DESI's acknowledgement text, given at https://data.desi.lbl.gov/doc/acknowledgments/ ; whoever redistributes the data says what was changed (Kosmulator's DESI files: see their headers).
