"""
What to cite and acknowledge: every data set and code Kosmulator uses.

- Observations/DATA_SOURCES.md (written from this file): for each observation tag, its files,
  where they come from, the licence status of the source, the papers to cite and any
  acknowledgement the data's licence asks for.
- At the end of a run, Statistical_analysis_tables/<suffix>/citations.md lists what that run
  used: Kosmulator, every data set of its groups, and the codes (samplers, CLASS, likelihood
  codes, BBN codes, GetDist).

Every reference was checked against its arXiv page and the journal record (13 October 2026);
the README's reference list is the same set.

    python -m Kosmulator_main.citations --write-data-sources     # regenerate DATA_SOURCES.md
    python -m Kosmulator_main.citations --check                  # is DATA_SOURCES.md up to date?
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence

from Kosmulator_main import constants as K

# ---------------------------------------------------------------------------
# References (checked: arXiv page for title, authors, DOI; journal from the publisher, INSPIRE
# or Crossref)
# ---------------------------------------------------------------------------
REFS: Dict[str, str] = {
    # Kosmulator
    "hough26": "R. T. Hough, R. Rugg, S. Sahlu, A. Abebe (2026), Kosmulator: A Python framework for cosmological "
               "inference with MCMC, arXiv:2602.08424",
    "hough20": "R. T. Hough, A. Abebe, S. E. S. Ferreira (2020), Viability tests of f(R)-gravity models with "
               "Supernovae Type 1A data, Eur. Phys. J. C 80, 787, arXiv:1911.05983, doi:10.1140/epjc/s10052-020-8342-7",
    # Samplers, analysis
    "foremanmackey13": "D. Foreman-Mackey, D. W. Hogg, D. Lang, J. Goodman (2013), emcee: The MCMC Hammer, "
                       "PASP 125, 306, arXiv:1202.3665, doi:10.1086/670067",
    "karamanis21": "M. Karamanis, F. Beutler, J. A. Peacock (2021), zeus: A Python implementation of Ensemble Slice "
                   "Sampling for efficient Bayesian parameter inference, MNRAS 508, 3589, arXiv:2105.03468, "
                   "doi:10.1093/mnras/stab2867",
    "karamanis21b": "M. Karamanis, F. Beutler (2021), Ensemble Slice Sampling, Stat. Comput. 31, 61, "
                    "arXiv:2002.06212, doi:10.1007/s11222-021-10038-2",
    "terbraak06": "C. J. F. ter Braak (2006), A Markov Chain Monte Carlo version of the genetic algorithm "
                  "Differential Evolution: easy Bayesian computing for real parameter spaces, Stat. Comput. 16, 239, "
                  "doi:10.1007/s11222-006-8769-1",
    "terbraak08": "C. J. F. ter Braak, J. A. Vrugt (2008), Differential Evolution Markov Chain with snooker updater "
                  "and fewer chains, Stat. Comput. 18, 435, doi:10.1007/s11222-008-9104-9",
    "lewis19": "A. Lewis (2025), GetDist: a Python package for analysing Monte Carlo samples, JCAP 08 (2025) 025, "
               "arXiv:1910.13970, doi:10.1088/1475-7516/2025/08/025",
    # Boltzmann code, non-linear power
    "lesgourgues11": "J. Lesgourgues (2011), The Cosmic Linear Anisotropy Solving System (CLASS) I: Overview, "
                     "arXiv:1104.2932",
    "blas11": "D. Blas, J. Lesgourgues, T. Tram (2011), The Cosmic Linear Anisotropy Solving System (CLASS) II: "
              "Approximation schemes, JCAP 07 (2011) 034, arXiv:1104.2933, doi:10.1088/1475-7516/2011/07/034",
    "eisenstein98": "D. J. Eisenstein, W. Hu (1998), Baryonic Features in the Matter Transfer Function, ApJ 496, 605, "
                    "arXiv:astro-ph/9709112, doi:10.1086/305424",
    "mead21": "A. Mead, S. Brieden, T. Troester, C. Heymans (2021), HMcode-2020: Improved modelling of non-linear "
              "cosmological power spectra with baryonic feedback, MNRAS 502, 1401, arXiv:2009.01858, "
              "doi:10.1093/mnras/stab082",
    "takahashi12": "R. Takahashi, M. Sato, T. Nishimichi, A. Taruya, M. Oguri (2012), Revising the Halofit Model for "
                   "the Nonlinear Matter Power Spectrum, ApJ 761, 152, arXiv:1208.2701, doi:10.1088/0004-637X/761/2/152",
    # Planck
    "planck18_V": "Planck Collaboration, N. Aghanim et al. (2020), Planck 2018 results. V. CMB power spectra and "
                  "likelihoods, A&A 641, A5, arXiv:1907.12875, doi:10.1051/0004-6361/201936386",
    "planck18_VI": "Planck Collaboration, N. Aghanim et al. (2020), Planck 2018 results. VI. Cosmological parameters, "
                   "A&A 641, A6, arXiv:1807.06209, doi:10.1051/0004-6361/201833910",
    "planck18_VIII": "Planck Collaboration, N. Aghanim et al. (2020), Planck 2018 results. VIII. Gravitational "
                     "lensing, A&A 641, A8, arXiv:1807.06210, doi:10.1051/0004-6361/201833886",
    "pagano20": "L. Pagano et al. (2020), Reionization optical depth determination from Planck HFI data with ten "
                "percent accuracy, A&A 635, A99, arXiv:1908.09856, doi:10.1051/0004-6361/201936630",
    "carron22": "J. Carron, M. Mirmelstein, A. Lewis (2022), CMB lensing from Planck PR4 maps, JCAP 09 (2022) 039, "
                "arXiv:2206.07773, doi:10.1088/1475-7516/2022/09/039",
    # ACT
    "naess25": "S. Naess et al. (2025), The Atacama Cosmology Telescope: DR6 Maps, JCAP 11 (2025) 061, "
               "arXiv:2503.14451, doi:10.1088/1475-7516/2025/11/061",
    "louis25": "T. Louis et al. (2025), The Atacama Cosmology Telescope: DR6 Power Spectra, Likelihoods and LCDM "
               "Parameters, JCAP 11 (2025) 062, arXiv:2503.14452, doi:10.1088/1475-7516/2025/11/062",
    "calabrese25": "E. Calabrese et al. (2025), The Atacama Cosmology Telescope: DR6 Constraints on Extended "
                   "Cosmological Models, JCAP 11 (2025) 063, arXiv:2503.14454, doi:10.1088/1475-7516/2025/11/063",
    "qu24": "F. J. Qu et al. (2024), The Atacama Cosmology Telescope: A Measurement of the DR6 CMB Lensing Power "
            "Spectrum and its Implications for Structure Growth, ApJ 962, 112, arXiv:2304.05202, "
            "doi:10.3847/1538-4357/acfe06",
    "madhavacheril24": "M. S. Madhavacheril et al. (2024), The Atacama Cosmology Telescope: DR6 Gravitational Lensing "
                       "Map and Cosmological Parameters, ApJ 962, 113, arXiv:2304.05203, doi:10.3847/1538-4357/acff5f",
    # SPT
    "camphuis26": "E. Camphuis et al. (2026), SPT-3G D1: CMB temperature and polarization power spectra and cosmology "
                  "from 2019 and 2020 observations of the SPT-3G Main field, Phys. Rev. D 113, 083504, "
                  "arXiv:2506.20707, doi:10.1103/7wt3-9v2y",
    "balkenhol25": "L. Balkenhol (2025), Compressed 'CMB-lite' Likelihoods Using Automatic Differentiation, "
                   "Open J. Astrophys., arXiv:2412.00826, doi:10.33232/001c.129886",
    "quan26": "W. Quan et al. (2026), SPT-3G D1: Maps of the millimeter-wave sky from 2019 and 2020 observations of "
              "the SPT-3G Main field, arXiv:2603.20163",
    "ge25": "F. Ge et al. (2025), Cosmology From CMB Lensing and Delensed EE Power Spectra Using 2019-2020 SPT-3G "
            "Polarization Data, Phys. Rev. D 111, 083534, arXiv:2411.06000, doi:10.1103/PhysRevD.111.083534",
    "balkenhol24": "L. Balkenhol, C. Trendafilova, K. Benabed, S. Galli (2024), candl: Cosmic Microwave Background "
                   "Analysis with a Differentiable Likelihood, A&A 686, A10, arXiv:2401.13433, "
                   "doi:10.1051/0004-6361/202449432",
    # DESI
    "desi24_VI": "DESI Collaboration, A. G. Adame et al. (2025), DESI 2024 VI: Cosmological Constraints from the "
                 "Measurements of Baryon Acoustic Oscillations, JCAP 02 (2025) 021, arXiv:2404.03002, "
                 "doi:10.1088/1475-7516/2025/02/021",
    "desi25_dr2": "DESI Collaboration, M. Abdul Karim et al. (2025), DESI DR2 Results II: Measurements of Baryon "
                  "Acoustic Oscillations and Cosmological Constraints, Phys. Rev. D 112, 083515, arXiv:2503.14738, "
                  "doi:10.1103/tr6y-kpc6",
    "desi_dr1": "DESI Collaboration, M. Abdul Karim et al. (2026), Data Release 1 of the Dark Energy Spectroscopic "
                "Instrument, AJ 171, 285, arXiv:2503.14745, doi:10.3847/1538-3881/ae4c43",
    # Supernovae
    "betoule14": "M. Betoule et al. (2014), Improved cosmological constraints from a joint analysis of the SDSS-II "
                 "and SNLS supernova samples, A&A 568, A22, arXiv:1401.4064, doi:10.1051/0004-6361/201423413",
    "scolnic18": "D. M. Scolnic et al. (2018), The Complete Light-curve Sample of Spectroscopically Confirmed SNe Ia "
                 "from Pan-STARRS1 and Cosmological Constraints from the Combined Pantheon Sample, ApJ 859, 101, "
                 "arXiv:1710.00845, doi:10.3847/1538-4357/aab9bb",
    "scolnic22": "D. Scolnic et al. (2022), The Pantheon+ Analysis: The Full Data Set and Light-curve Release, "
                 "ApJ 938, 113, arXiv:2112.03863, doi:10.3847/1538-4357/ac8b7a",
    "brout22": "D. Brout et al. (2022), The Pantheon+ Analysis: Cosmological Constraints, ApJ 938, 110, "
               "arXiv:2202.04077, doi:10.3847/1538-4357/ac8e04",
    "riess22": "A. G. Riess et al. (2022), A Comprehensive Measurement of the Local Value of the Hubble Constant "
               "with 1 km/s/Mpc Uncertainty from the Hubble Space Telescope and the SH0ES Team, ApJL 934, L7, "
               "arXiv:2112.04510, doi:10.3847/2041-8213/ac5c5b",
    "des24": "DES Collaboration, T. M. C. Abbott et al. (2024), The Dark Energy Survey: Cosmology Results with ~1500 "
             "New High-redshift Type Ia Supernovae Using the Full 5 yr Data Set, ApJL 973, L14, arXiv:2401.02929, "
             "doi:10.3847/2041-8213/ad6f9f",
    "sanchez24": "B. O. Sanchez et al. (2024), The Dark Energy Survey Supernova Program: Light Curves and 5 Yr Data "
                 "Release, ApJ 975, 5, arXiv:2406.05046, doi:10.3847/1538-4357/ad739a",
    "rubin25": "D. Rubin et al. (2025), Union Through UNITY: Cosmology with 2,000 SNe Using a Unified Bayesian "
               "Framework, ApJ 986, 231, arXiv:2311.12098, doi:10.3847/1538-4357/adc0a5",
    "hicken09": "M. Hicken et al. (2009), CfA3: 185 Type Ia Supernova Light Curves from the CfA, ApJ 700, 331, "
                "arXiv:0901.4787, doi:10.1088/0004-637X/700/1/331",
    "neill09": "J. D. Neill et al. (2009), The Local Hosts of Type Ia Supernovae, ApJ 707, 1449, "
               "arXiv:0911.0690, doi:10.1088/0004-637X/707/2/1449",
    "conley11": "A. Conley et al. (2011), Supernova Constraints and Systematic Uncertainties from the First Three "
                "Years of the Supernova Legacy Survey, ApJS 192, 1, arXiv:1104.1443, doi:10.1088/0067-0049/192/1/1",
    # Expansion rate, growth
    "moresco20": "M. Moresco et al. (2020), Setting the Stage for Cosmic Chronometers. II. Impact of Stellar "
                 "Population Synthesis Models Systematics and Full Covariance Matrix, ApJ 898, 82, arXiv:2003.07362, "
                 "doi:10.3847/1538-4357/ab9eb0",
    "sharov18": "G. S. Sharov, V. O. Vasiliev (2018), How predictions of cosmological models depend on Hubble "
                "parameter data sets, Math. Model. Geom. 6, 1, arXiv:1807.07323, doi:10.26456/mmg/2018-611",
    "perenon19": "L. Perenon, J. Bel, R. Maartens, A. de la Cruz-Dombriz (2019), Optimising growth of structure "
                 "constraints on modified gravity, JCAP 06 (2019) 020, arXiv:1901.11063, "
                 "doi:10.1088/1475-7516/2019/06/020",
    "avila22": "F. Avila, A. Bernui, A. Bonilla, R. C. Nunes (2022), Inferring S8(z) and gamma(z) with cosmic growth "
               "rate measurements using machine learning, Eur. Phys. J. C 82, 594, arXiv:2201.07829, "
               "doi:10.1140/epjc/s10052-022-10561-0",
    "skara20": "F. Skara, L. Perivolaropoulos (2020), Tension of the E_G statistic and RSD data with Planck/LCDM and "
               "implications for weakening gravity, Phys. Rev. D 101, 063521, arXiv:1911.10609, "
               "doi:10.1103/PhysRevD.101.063521",
    "kazantzidis18": "L. Kazantzidis, L. Perivolaropoulos (2018), Evolution of the f sigma_8 tension with the "
                     "Planck15/LCDM determination and implications for modified gravity theories, Phys. Rev. D 97, "
                     "103503, arXiv:1803.01337, doi:10.1103/PhysRevD.97.103503",
    # BBN
    "pdg_bbn": "B. D. Fields, P. Molaro, S. Sarkar, Big Bang Nucleosynthesis, review in S. Navas et al. (Particle "
               "Data Group), Phys. Rev. D 110, 030001 (2024) and 2025 update, doi:10.1103/PhysRevD.110.030001",
    "cooke14": "R. J. Cooke, M. Pettini, R. A. Jorgenson, M. T. Murphy, C. C. Steidel (2014), Precision measures of "
               "the primordial abundance of deuterium, ApJ 781, 31, arXiv:1308.3240, doi:10.1088/0004-637X/781/1/31",
    "burns24": "A.-K. Burns, T. M. P. Tait, M. Valli (2024), PRyMordial: The First Three Minutes, Within and Beyond "
               "the Standard Model, Eur. Phys. J. C 84, 86, arXiv:2307.07061, doi:10.1140/epjc/s10052-024-12442-0",
    "arbey12": "A. Arbey (2012), AlterBBN: A program for calculating the BBN abundances of the elements in "
               "alternative cosmologies, Comput. Phys. Commun. 183, 1822, arXiv:1106.1363, "
               "doi:10.1016/j.cpc.2012.03.018",
    "arbey20": "A. Arbey, J. Auffinger, K. P. Hickerson, E. S. Jenssen (2020), AlterBBN v2: A public code for "
               "calculating Big-Bang nucleosynthesis constraints in alternative cosmologies, Comput. Phys. Commun. "
               "248, 106982, arXiv:1806.11095, doi:10.1016/j.cpc.2019.106982",
}

# ---------------------------------------------------------------------------
# Licence statuses of the sources (checked 13 October 2026)
# ---------------------------------------------------------------------------
LICENCES: Dict[str, str] = {
    "cc_by_desi": "CC BY 4.0 (DESI data licence, https://data.desi.lbl.gov/doc/acknowledgments/)",
    "mit_candl": "MIT (candl_data; its licence is Observations/candl/LICENCE_candl_data.txt)",
    "mit_union3": "MIT (github.com/rubind/union3_release)",
    "none_planck": "no licence stated (ESA Planck Legacy Archive)",
    "none_lambda": "no licence stated; the LAMBDA page asks to cite Qu et al. and Madhavacheril et al.",
    "none_repo": "no licence file at the source",
    "none_spt": "no licence file (spt_candl_data, commit f22611a)",
    "none_muse": "no licence file (muse_3g_like_march_2025.zip)",
    "cc_by_desi_dr2": "values of DESI DR2 Results II; DESI's CC BY 4.0 data licence covers its public data releases "
                      "(EDR, DR1)",
    "unknown": "unknown (origin not documented)",
    "papers": "values compiled from the published papers",
    "kosmulator": "Kosmulator's own (MIT, with the repository)",
}

# ---------------------------------------------------------------------------
# Acknowledgements the data's licence asks for (only those stated by the provider)
# ---------------------------------------------------------------------------
ACKS: Dict[str, str] = {
    "desi": "DESI's data licence (CC BY 4.0, for its public data releases) asks publications that use DESI data to "
            "cite the paper of the data release used (DR1: DESI Collaboration 2026, AJ 171, 285) and to include "
            "DESI's acknowledgement text, given at https://data.desi.lbl.gov/doc/acknowledgments/ ; whoever "
            "redistributes the data says what was changed (Kosmulator's DESI files: see their headers).",
}

# ---------------------------------------------------------------------------
# Data sets: tag(s) -> files, source, licence, papers
# ---------------------------------------------------------------------------
DATA: List[Dict[str, object]] = [
    dict(tags=("CC",), name="Cosmic chronometers H(z)",
         files=("CC.dat", "CC_corr.txt"), licence="papers",
         source="Kosmulator's compilation; every row checked against its paper (source after '#' in CC.dat); "
                "correlations of the Loubser (2025) DESI points in CC_corr.txt",
         refs=("moresco20",), rows_from="CC.dat",
         note="Cite the paper of every row used (listed per run, and in CC.dat). Moresco et al. (2020) for the "
              "optional SPS-model systematic covariance."),
    dict(tags=("OHD",), name="H(z) compilation (31 chronometer + 20 BAO points)",
         files=("OHD.dat",), licence="papers", source="compilation of Sharov & Vasiliev (2018); origin of the file "
                                                       "not recorded",
         refs=("sharov18",), note="Also cite the original papers of the points, listed in Sharov & Vasiliev (2018)."),
    dict(tags=("JLA",), name="SDSS-II/SNLS3 Joint Light-curve Analysis (740 SNe)",
         files=("JLA/",), licence="none_repo",
         source="CosmoMC/Cobaya distribution, github.com/CobayaSampler/sn_data (JLA/, commit 61d9643); release "
                "page https://supernovae.in2p3.fr/sdss_snls_jla/ReadMe.html",
         refs=("betoule14",)),
    dict(tags=("JLA_legacy",), name="Old 359-SN distance moduli (kept to reproduce earlier runs)",
         files=("JLA_legacy.dat",), licence="unknown", source="origin not documented",
         refs=("hicken09", "neill09", "conley11"), note="References as previously listed; prefer JLA."),
    dict(tags=("Pantheon",), name="Pantheon SNe Ia (1048)", files=("Pantheon.dat",), licence="none_repo",
         source="Pantheon release (github.com/dscolnic/Pantheon); Kosmulator's copy: origin not recorded",
         refs=("scolnic18",)),
    dict(tags=("PantheonP", "PantheonPS"), name="Pantheon+ (PantheonPS: with the SH0ES Cepheid calibration)",
         files=("PantheonP.dat", "PantheonP.cov"), licence="none_repo",
         source="Pantheon+ release, github.com/PantheonPlusSH0ES/DataRelease (Pantheon+SH0ES.dat and its STAT+SYS "
                "covariance; Kosmulator's copy: format matches)",
         refs=("scolnic22", "brout22"), refs_by_tag={"PantheonPS": ("riess22",)}),
    dict(tags=("DESY5",), name="DES-SN5YR Hubble diagram", files=("DESY5.dat", "DESY5_covsys_000.txt"),
         licence="none_repo", source="DES-SN5YR release, github.com/des-science/DES-SN5YR (DES-SN5YR_HD.csv and "
                                     "covsys_000; Kosmulator's copy: format matches)",
         refs=("des24", "sanchez24")),
    dict(tags=("Union3",), name="Union3 / UNITY1.5 binned distance moduli", files=("Union3.txt",
                                                                                      "Union3_mag_covmat.txt"),
         licence="mit_union3", source="Union3 release, github.com/rubind/union3_release; Kosmulator's copy: origin "
                                      "not recorded", refs=("rubin25",)),
    dict(tags=("BAO",), name="DESI DR1 BAO, the 10 anisotropic points", files=("BAO.txt", "BAO_invcov.txt"),
         licence="cc_by_desi", source="values of DESI 2024 VI (Table 1), two decimals; see BAO.txt's header",
         refs=("desi24_VI", "desi_dr1"), ack=("desi",)),
    dict(tags=("DESI_DR1",), name="DESI DR1 BAO, all 12 measurements", files=("DESI_DR1.txt", "DESI_DR1_cov.txt"),
         licence="cc_by_desi", source="DESI 2024 likelihood release as distributed in github.com/CobayaSampler/"
                                      "bao_data (desi_2024_gaussian_bao_ALL_GCcomb); see DESI_DR1.txt's header",
         refs=("desi24_VI", "desi_dr1"), ack=("desi",)),
    dict(tags=("DESI_DR2",), name="DESI DR2 BAO", files=("DESI_DR2.txt", "DESI_DR2_covtot.txt"),
         licence="cc_by_desi_dr2", source="DESI DR2 BAO (values and covariance of DESI DR2 Results II); Kosmulator's "
                                      "copy: origin not recorded",
         refs=("desi25_dr2",), ack=("desi",)),
    dict(tags=("f",), name="Growth rate f(z)", files=("f.dat",), licence="papers",
         source="rows 1 - 3 Perenon et al. (2019), Table 1; rows 4 - 14 Avila et al. (2022), Table 1 (f.dat header)",
         refs=("perenon19", "avila22"), note="Also cite the survey papers of the points (listed in those tables)."),
    dict(tags=("f_sigma_8",), name="f sigma_8(z) from redshift-space distortions (66 points)",
         files=("f_sigma_8.dat",), licence="papers",
         source="Skara & Perivolaropoulos (2020), Appendix B, Table VI, an update of Kazantzidis & Perivolaropoulos "
                "(2018), Table II (f_sigma_8.dat header)",
         refs=("skara20", "kazantzidis18"), note="Also cite the survey papers of the points (listed in Table VI)."),
    dict(tags=("BBN_DH", "BBN_DH_AlterBBN", "BBN_PryMordial", "BBN_prior"),
         name="Primordial deuterium D/H (12 quasar systems, PDG 2025) and BBN predictions",
         files=("constants.BBN_DH_PDG_SYSTEMS", "BBN/ (grids made by Kosmulator)"), licence="papers",
         source="PDG BBN review (2025 update); D/H(omega_b) from AlterBBN (BBN_DH_AlterBBN) or a power law fitted "
                "to the PRyMordial table",
         refs=("pdg_bbn", "cooke14", "burns24"), refs_by_tag={"BBN_DH_AlterBBN": ("arbey12", "arbey20")},
         note="Also cite the papers of the 12 D/H systems (listed in the PDG review)."),
    dict(tags=("CMB_hil", "CMB_hil_TT"), name="Planck 2018 high-l TT,TE,EE / TT (plik, plik_lite)",
         files=("plik_rd12_HM_v22b_TTTEEE.clik", "plik_rd12_HM_v22_TT.clik", "plik_lite_v22_TTTEEE.clik",
                "plik_lite_v22_TT.clik"), licence="none_planck",
         source="Planck Legacy Archive, COM_Likelihood_Data-baseline_R3.00", refs=("planck18_V", "planck18_VI")),
    dict(tags=("CMB_lowl", "CMB_lowl_TT"), name="Planck 2018 low-l EE (SimAll) and TT (Commander)",
         files=("simall_100x143_offlike5_EE_Aplanck_B.clik", "commander_dx12_v3_2_29.clik"), licence="none_planck",
         source="Planck Legacy Archive, COM_Likelihood_Data-baseline_R3.00", refs=("planck18_V",)),
    dict(tags=("CMB_lensing",), name="Planck 2018 lensing (raw and CMB-marginalised)",
         files=("smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing",
                "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_CMBmarged.clik_lensing"), licence="none_planck",
         source="Planck Legacy Archive, COM_Likelihood_Data-baseline_R3.00", refs=("planck18_VIII",)),
    dict(tags=("CMB_lowl_sroll2",), name="Planck low-l EE from the SRoll2 maps (probability table)",
         files=("sroll2_prob_table.txt",), licence="none_repo",
         source="Cobaya's planck_native_data release (planck_sroll2_lowE.zip), github.com/CobayaSampler/"
                "planck_native_data", refs=("pagano20",)),
    dict(tags=("CMB_planck_ACTcut",), name="Planck 2018 plik_lite cut at the ACT DR6 multipoles",
         files=("plik_lite_v22_TTTEEE.clik",), licence="none_planck",
         source="Planck Legacy Archive; the cut as ACT DR6's PlanckActCut", refs=("planck18_V", "louis25")),
    dict(tags=("CMB_ACT",), name="ACT DR6 TT/TE/EE (ACT-lite, foreground-marginalised)",
         files=("candl/ACT_DR6_CMB_only_v0/",), licence="mit_candl",
         source="candl_data 2.0.0 (commit c4eb8fb), Observations/candl/README.md",
         refs=("louis25", "calabrese25", "naess25")),
    dict(tags=("CMB_SPT",), name="SPT-3G D1 T&E (SPT-lite, foreground-marginalised)",
         files=("candl/SPT3G_D1_TnE_lite/",), licence="none_spt",
         source="spt_candl_data 3.0.1 (commit f22611a), Observations/candl/README.md",
         refs=("camphuis26", "balkenhol25", "quan26"),
         note="SPT-lite is Appendix F of Camphuis et al.; Quan et al. (D1 maps) as spt_candl_data lists."),
    dict(tags=("CMB_lensing_ACTPlanck",), name="ACT DR6 + Planck PR4 lensing power spectrum (actplanck_baseline)",
         files=("act_dr6_lenslike/",), licence="none_lambda",
         source="LAMBDA, ACT_dr6_likelihood_v1.2.tgz (Observations/act_dr6_lenslike/README.md); the binned "
                "corrections file is Kosmulator's",
         refs=("qu24", "madhavacheril24", "carron22")),
    dict(tags=("CMB_lensing_SPT",), name="SPT-3G MUSE lensing (phi phi band powers)",
         files=("spt3g_muse_lensing/",), licence="none_muse",
         source="muse_3g_like_march_2025.zip, https://pole.uchicago.edu/public/data/ge25/ "
                "(Observations/spt3g_muse_lensing/README.md)", refs=("ge25",)),
]

# Plot-only data (the best-fit figures), not likelihoods
PLOT_DATA: List[Dict[str, object]] = [
    dict(name="Planck 2018 binned power spectra drawn on bestfit.png", files=("CMB_TT.dat", "CMB_TE.dat",
                                                                               "CMB_EE.dat"),
         licence="none_planck", source="Planck Legacy Archive (format of COM_PowerSpect_CMB-*-full_R3.01)",
         refs=("planck18_V",)),
]

# ---------------------------------------------------------------------------
# Codes: when they are used, and what to cite
# ---------------------------------------------------------------------------
CODES: Dict[str, Dict[str, object]] = {
    "kosmulator": dict(name="Kosmulator", refs=("hough26", "hough20")),
    "emcee": dict(name="emcee (ensemble sampler)", refs=("foremanmackey13",)),
    "de_snooker": dict(name="emcee's DE move and Kosmulator's snooker move (emcee_moves = \"de_snooker\")",
                       refs=("terbraak06", "terbraak08")),
    "zeus": dict(name="zeus (ensemble slice sampling)", refs=("karamanis21", "karamanis21b")),
    "class": dict(name="CLASS (CMB spectra; r_d)", refs=("lesgourgues11", "blas11")),
    "eh98": dict(name="r_d from the Eisenstein & Hu (1998) fit (CLASS was not loaded)", refs=("eisenstein98",)),
    "hmcode": dict(name="HMcode-2020 in CLASS (cmb_nonlinear = \"hmcode\")", refs=("mead21",)),
    "halofit": dict(name="Halofit in CLASS (cmb_nonlinear = \"halofit\")", refs=("takahashi12",)),
    "clik": dict(name="Planck likelihood code (clik or clipy, github.com/benabed/clipy)", refs=("planck18_V",)),
    "candl": dict(name="candl (ACT and SPT likelihoods)", refs=("balkenhol24",)),
    "act_dr6_lenslike": dict(name="act_dr6_lenslike (ACT DR6 + Planck PR4 lensing likelihood code)",
                             refs=("qu24", "madhavacheril24")),
    "alterbbn": dict(name="AlterBBN", refs=("arbey12", "arbey20")),
    "prymordial": dict(name="PRyMordial (the D/H table behind the BBN power law and BBN_PryMordial)",
                       refs=("burns24",)),
    "getdist": dict(name="GetDist (corner plots, marginal statistics)", refs=("lewis19",)),
}

#: The samplers that ran in this process (Kosmulator_MCMC records them), for the run's citations
ENGINES_USED: set = set()

_PLANCK_TAGS = ("CMB_hil", "CMB_hil_TT", "CMB_lowl", "CMB_lowl_TT", "CMB_lensing", "CMB_planck_ACTcut")


def _components(tags: Iterable[str]) -> List[str]:
    """The tags of a group with the combinations (CMB_PACT, CMB_SPA) expanded, each once."""
    out: List[str] = []
    for t in tags:
        parts = list(K.cmb_components(str(t))) if hasattr(K, "cmb_components") else [str(t)]
        for c in parts or [str(t)]:
            if c not in out:
                out.append(c)
    return out


def entry_for(tag: str) -> Optional[Dict[str, object]]:
    for e in DATA:
        if tag in e["tags"]:
            return e
    return None


def refs_for_tag(tag: str) -> List[str]:
    e = entry_for(tag)
    if e is None:
        return []
    keys = list(e["refs"])
    keys += list(dict(e.get("refs_by_tag", {})).get(tag, ()))
    return keys


def cc_row_sources(path: Optional[str] = None) -> List[str]:
    """The source of every row of CC.dat ('Author year, arXiv:...'), each once, in file order."""
    path = path or os.path.join(K.OBSERVATIONS_BASE, "CC.dat")
    found: Dict[str, str] = {}                  # author-year -> identifier ('' if none seen yet)
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        return []
    for line in lines:
        if line.lstrip().startswith("#") or "#" not in line:
            continue
        c = line.split("#", 1)[1].strip()
        m = re.match(r"(.+?\b(?:19|20)\d{2})\b", c)
        if not m:
            continue
        who = m.group(1).strip()
        ident = re.search(r"(arXiv:\d{4}\.\d{4,5}|astro-ph/\d{7})", c)
        if not found.get(who):
            found[who] = ident.group(1) if ident else found.get(who, "")
    return [who + (f", {i}" if i else "") for who, i in found.items()]


def codes_used(observations: Sequence[Sequence[str]], engines: Iterable[str] = (), settings: Optional[dict] = None,
               class_loaded: Optional[bool] = None) -> List[str]:
    """The codes a run with these groups used (engines: the samplers that ran; class_loaded: whether CLASS
    computed r_d for BAO, by default whether classy is loaded in this process)."""
    settings = dict(settings or {})
    tags = set(_components(t for g in observations for t in g))
    used = ["kosmulator"]
    engines = set(engines) or {"emcee"}
    if "emcee" in engines:
        used.append("emcee")
        if str(settings.get("emcee_moves", getattr(K, "EMCEE_MOVES", "stretch"))) == "de_snooker":
            used.append("de_snooker")
    if "zeus" in engines:
        used.append("zeus")
    cmb = any(t.startswith("CMB_") for t in tags)
    bao = bool(tags & {"BAO", "DESI_DR1", "DESI_DR2"})
    if class_loaded is None:
        class_loaded = "classy" in sys.modules
    if bao and not (cmb or class_loaded):
        used.append("eh98")
    if cmb or (bao and class_loaded):
        used.append("class")
        nl = settings.get("cmb_nonlinear", getattr(K, "CMB_NONLINEAR", None)) if cmb else None
        if nl == "hmcode" and str(settings.get("hmcode_version", getattr(K, "HMCODE_VERSION", "2020"))) == "2020":
            used.append("hmcode")
        elif nl == "halofit":
            used.append("halofit")
    if tags & set(_PLANCK_TAGS):
        used.append("clik")
    if tags & {"CMB_ACT", "CMB_SPT"}:
        used.append("candl")
    if "CMB_lensing_ACTPlanck" in tags:
        used.append("act_dr6_lenslike")
    if "BBN_DH_AlterBBN" in tags:
        used.append("alterbbn")
    if tags & {"BBN_DH", "BBN_PryMordial"}:
        used.append("prymordial")
    used.append("getdist")
    return used


def _fmt_refs(keys: Iterable[str], indent: str = "  ") -> List[str]:
    return [f"{indent}- {REFS[k]}" for k in keys]


def run_citations_markdown(models: Sequence[str], observations: Sequence[Sequence[str]], suffix: str = "",
                           engines: Iterable[str] = (), settings: Optional[dict] = None,
                           class_loaded: Optional[bool] = None) -> str:
    """What to cite and acknowledge for one run (markdown)."""
    tags = _components(t for g in observations for t in g)
    out = [f"# What to cite for this run{f' ({suffix})' if suffix else ''}", "",
           f"Written by Kosmulator on {time.strftime('%Y-%m-%d %H:%M')}. Models: {', '.join(map(str, models))}. "
           f"Groups: {'; '.join(' + '.join(map(str, g)) for g in observations)}.",
           "Cite the papers below for what this run used. Sources and licence status of every data file: "
           "Observations/DATA_SOURCES.md.", ""]
    used = codes_used(observations, engines, settings, class_loaded)
    out += ["## Kosmulator", ""] + _fmt_refs(CODES["kosmulator"]["refs"]) + [""]
    out += ["## Data", ""]
    acks: List[str] = []
    for t in tags:
        e = entry_for(t)
        if e is None:
            out += [f"### {t}", "", "  - (no entry: see Observations/DATA_SOURCES.md)", ""]
            continue
        out += [f"### {t}: {e['name']}", ""] + _fmt_refs(refs_for_tag(t))
        if e.get("rows_from") == "CC.dat":
            rows = cc_row_sources()
            if rows:
                out += ["  - the papers of the CC.dat rows:"] + [f"    - {r}" for r in rows]
        if e.get("note"):
            out += [f"  - Note: {e['note']}"]
        out.append("")
        for a in e.get("ack", ()):
            if a not in acks:
                acks.append(a)
    out += ["## Codes", ""]
    for c in used:
        if c == "kosmulator":
            continue
        out += [f"### {CODES[c]['name']}", ""] + _fmt_refs(CODES[c]["refs"]) + [""]
    if acks:
        out += ["## Acknowledgements the data's licences ask for", ""] + [f"- {ACKS[a]}" for a in acks] + [""]
    return "\n".join(out).rstrip() + "\n"


def write_run_citations(suffix: str, models: Sequence[str], observations: Sequence[Sequence[str]],
                        engines: Iterable[str] = (), settings: Optional[dict] = None) -> Optional[str]:
    """Write Statistical_analysis_tables/<suffix>/citations.md; returns its path (None if it fails)."""
    try:
        folder = os.path.join("Statistical_analysis_tables", suffix or "default_run")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "citations.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(run_citations_markdown(models, observations, suffix, engines, settings))
        return path
    except Exception as e:                       # never stop a finished run over this file
        print(f"[citations] not written ({e})")
        return None


# ---------------------------------------------------------------------------
# Observations/DATA_SOURCES.md
# ---------------------------------------------------------------------------
def data_sources_markdown() -> str:
    out = ["# Data sources, licences and what to cite", "",
           "Written from `Kosmulator_main/citations.py` (`python -m Kosmulator_main.citations --write-data-sources`); "
           "every reference was checked against its arXiv page and the journal record (13 October 2026). Each run "
           "also writes `Statistical_analysis_tables/<suffix>/citations.md` with what that run used.", "",
           "Kosmulator includes every data set it supports, with its source and the papers to cite, as is common "
           "practice for published cosmological data. Several sources have no licence file; if a collaboration asks "
           "for its data to be removed, it will be (the SPT-3G files can be restored with "
           "`python -m Kosmulator_main.get_spt_data`). Cite Kosmulator and the original papers of the data you use.",
           "", "## Kosmulator", ""] + _fmt_refs(CODES["kosmulator"]["refs"], "") + [""]
    out += ["## Likelihood data", "",
            "| Tag(s) | Data | Files in `Observations/` | Source | Licence status |", "|---|---|---|---|---|"]
    for e in DATA:
        out.append(f"| {', '.join('`%s`' % t for t in e['tags'])} | {e['name']} | "
                   f"{', '.join('`%s`' % f for f in e['files'])} | {e['source']} | {LICENCES[e['licence']]} |")
    out += ["", "Combinations: `CMB_PACT` = `CMB_lowl_TT` + `CMB_lowl_sroll2` + `CMB_planck_ACTcut` + `CMB_ACT`; "
                "`CMB_SPA` = the same + `CMB_SPT`. Cite their parts.", "", "## What to cite, per tag", ""]
    for e in DATA:
        out += [f"### {', '.join(e['tags'])}: {e['name']}", ""] + _fmt_refs(e["refs"], "")
        for t, ks in dict(e.get("refs_by_tag", {})).items():
            out += [f"- with `{t}` also:"] + _fmt_refs(ks, "  ")
        if e.get("rows_from") == "CC.dat":
            out += ["- the papers of the CC.dat rows used:"] + [f"  - {r}" for r in cc_row_sources()]
        if e.get("note"):
            out += [f"- Note: {e['note']}"]
        for a in e.get("ack", ()):
            out += [f"- Acknowledgement: {ACKS[a]}"]
        out.append("")
    out += ["## Plot-only data", ""]
    for e in PLOT_DATA:
        out += [f"- {e['name']} ({', '.join('`%s`' % f for f in e['files'])}): {e['source']}; "
                f"{LICENCES[e['licence']]}."] + _fmt_refs(e["refs"], "  ")
    out += ["", "## Codes", ""]
    for c in CODES.values():
        out += [f"### {c['name']}", ""] + _fmt_refs(c["refs"], "") + [""]
    out += ["## Acknowledgements the data's licences ask for", ""] + [f"- {a}" for a in ACKS.values()]
    return "\n".join(out).rstrip() + "\n"


def data_sources_path() -> str:
    return os.path.join(K.OBSERVATIONS_BASE, "DATA_SOURCES.md")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="What to cite: Observations/DATA_SOURCES.md from the registry.")
    ap.add_argument("--write-data-sources", action="store_true", help="write Observations/DATA_SOURCES.md")
    ap.add_argument("--check", action="store_true", help="exit 1 if DATA_SOURCES.md is not up to date")
    a = ap.parse_args(argv)
    text = data_sources_markdown()
    if a.write_data_sources:
        with open(data_sources_path(), "w", encoding="utf-8") as fh:
            fh.write(text)
        print("written:", data_sources_path())
    if a.check:
        try:
            same = open(data_sources_path(), encoding="utf-8").read() == text
        except OSError:
            same = False
        print("DATA_SOURCES.md is", "up to date" if same else "out of date")
        return 0 if same else 1
    if not (a.write_data_sources or a.check):
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
