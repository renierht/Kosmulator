# Planck PR4 CamSpec data (`CMB_hil_PR4`)

This folder holds the data of `CMB_hil_PR4`, the Planck PR4 (NPIPE) high-ℓ TT, TE, EE likelihood with
CamSpec (Rosenberg, Gratton & Efstathiou 2022, MNRAS 517, 4620, arXiv:2205.10869), DESI DR2's baseline
high-ℓ CMB. The data are **not in Kosmulator's repository**: their covariance alone is 509 MB, and GitHub
takes files of at most 100 MB. Only this README is in git.

## How they get here

The first run with `CMB_hil_PR4` that finds no data downloads them at start-up, before sampling, and says
why and from where. By hand:

    python -m Kosmulator_main.planck_pr4 --get      # download, unpack, check (md5)
    python -m Kosmulator_main.planck_pr4 --check    # check (md5) and build the inverse covariance

Source: `CamSpec_NPIPE.zip` (85.4 MB, 0.5 GB unpacked) of the `planck_native_data` release v1, where
Cobaya's authors publish the Planck likelihood data:
https://github.com/CobayaSampler/planck_native_data/releases/download/v1/CamSpec_NPIPE.zip.
Cobaya itself is not needed. The zip unpacks to `CamSpec_NPIPE/` here.

If you already have these data (for example in Cobaya's packages, `<packages>/data/planck_NPIPE_CamSpec/
CamSpec_NPIPE`), point Kosmulator at them instead of downloading:

    export KOSM_CAMSPEC_PR4_DIR=/path/to/CamSpec_NPIPE

On a cluster whose compute nodes have no internet, run `--get` once on the login node.

## Files Kosmulator reads (md5, checked after a download and by `--check`)

| File | md5 |
|---|---|
| `CamSpec_NPIPE_12_6_cl.dataset` | `f95aac05fddcd93562ee0bfe43cf58df` |
| `like_NPIPE_12.6_unified_cov.bin` | `843063c3ac30984fe4f2470d0ce38d8f` |
| `like_NPIPE_12.6_unified_spectra.txt` | `e0e22a85436a399380a65e855520e2ae` |
| `like_NPIPE_12.6_unified_data_ranges.txt` | `71607015ae0e6d0debc5d6d15e06373e` |

The first use also writes `CamSpec_NPIPE/.kosmulator_cache/covinv_TTTEEE_<key>.npy` (786 MB, about a
minute and 1 GB of memory once): the inverse of the used 9915 x 9915 covariance block, which every
process of a run maps read-only. When this folder cannot be written it goes to
`~/.cache/kosmulator/camspec_pr4` (or `KOSM_CACHE_DIR`).

## Licence and citation

The release has no licence file; Kosmulator downloads the data from their source and does not
redistribute them. Cite Rosenberg, Gratton & Efstathiou (2022) and the Planck PR4 maps (Planck
Collaboration, A&A 643, A42, 2020); `Observations/DATA_SOURCES.md` and each run's `citations.md` list them.
