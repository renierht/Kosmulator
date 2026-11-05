# ACT DR6 and SPT-3G D1 for candl (CMB_ACT, CMB_SPT, CMB_PACT, CMB_SPA)

The two foreground-marginalised ("lite") CMB data sets that `Kosmulator_main/ground_cmb.py` gives to
candl, as unchanged copies, so that Kosmulator needs only the candl code:

```bash
pip install candl-like          # tested with candl-like 2.2.0 (SPT-3G D1 T&E needs candl >= 2.0.3)
```

`candl_data` and `spt_candl_data` do not have to be installed (if they are, Kosmulator still reads the
copies here; the packages are used only when a folder below is missing). No Cobaya is involved.

| Folder | Source | In the repository | Used by |
|---|---|---|---|
| `ACT_DR6_CMB_only_v0/` | `candl_data/ACT_DR6_CMB_only_v0/` of [candl_data](https://github.com/Lbalkenhol/candl_data) 2.0.0 (commit `c4eb8fb`, 10 Oct 2025; `.DS_Store` left out). The data set `candl_data.ACT_DR6_TTTEEE`. | yes (MIT licence) | `CMB_ACT`, `CMB_PACT`, `CMB_SPA` |
| `SPT3G_D1_TnE_lite/` | `spt_candl_data/SPT3G_D1_TnE_v0/lite/` of [spt_candl_data](https://github.com/SouthPoleTelescope/spt_candl_data) 3.0.1 (commit `f22611a`, 28 Sep 2026). The data set `spt_candl_data.SPT3G_D1_TnE_lite`. | yes (no licence file; see below) | `CMB_SPT`, `CMB_SPA` |
| `tests/` | The packages' reference tests for these two data sets (`ACT_DR6_TTTEEE_test.*` from candl_data; `SPT3G_D1_TnE_lite_test.yaml` and `SPT3G_D1_TnE_test_spec.txt` from spt_candl_data). | yes | the verification kit (`GROUNDCMB`) |

## Restoring the SPT-3G D1 files

If the SPT files are missing or changed in your copy, this downloads them again from
github.com/SouthPoleTelescope/spt_candl_data (commit `f22611a`) and checks each against the md5 below:

```bash
python -m Kosmulator_main.get_spt_data
```

## Checksums

md5 of the data (LF line endings, as in the packages); the three ACT window files are identical, as in
candl_data:

```
520265f5abd29416b803353b143319d6  ACT_DR6_CMB_only_v0/ACT_DR6_CMB_only.yaml
16d8b394d15eb3fdc37e1f33f62e64bc  ACT_DR6_CMB_only_v0/ACT_DR6_CMB_only_bdp.txt
272434b1f24cff7876512c79bcd74c82  ACT_DR6_CMB_only_v0/ACT_DR6_CMB_only_cov.txt
3dc6db7f3d7c94c942eac08f22c22488  ACT_DR6_CMB_only_v0/windows/{TT,TE,EE}_lxl_window_functions.txt
6d18cec0786a13555456976d34acfa0d  SPT3G_D1_TnE_lite/SPT3G_D1_TnE_lite.yaml
7ab63d52d032f1c03a9baaab96ca8620  SPT3G_D1_TnE_lite/SPT3G_D1_TnE_lite_bdp.txt
d3b399a7e5f0d457c3d535e45643b1fc  SPT3G_D1_TnE_lite/SPT3G_D1_TnE_lite_cov.txt
f14cbf8d99939972248db20c9aa1eaee  SPT3G_D1_TnE_lite/windows/TT_lxl_window_functions.txt
b29b89ed9d1a48b7e42eb0a77f035392  SPT3G_D1_TnE_lite/windows/TE_lxl_window_functions.txt
324a1c65cef8645b775f9f3e54843f43  SPT3G_D1_TnE_lite/windows/EE_lxl_window_functions.txt
```

The yaml files read their data relative to their own folder, so the folders must keep their layout.

## Licences and citations

- candl_data is under the MIT licence (`LICENCE_candl_data.txt`, its copyright notice kept as the licence
  asks).
- spt_candl_data has no licence file (commit `f22611a`); its README asks users to cite the papers of the
  likelihoods used and the candl paper. Kosmulator includes these public data with their source and
  citations, as is common practice for published cosmological data; if the SPT collaboration asks
  otherwise, they will be removed and `get_spt_data` becomes the way to obtain them.

Cite, when these data are used:

- candl: L. Balkenhol et al. 2024, A&A 686, A10, arXiv:2401.13433.
- ACT DR6: Naess et al. 2025, JCAP 11 (2025) 061, arXiv:2503.14451; Louis et al. 2025, JCAP 11 (2025) 062,
  arXiv:2503.14452; Calabrese et al. 2025, JCAP 11 (2025) 063, arXiv:2503.14454 (as listed in
  `ACT_DR6_CMB_only.yaml`).
- SPT-3G D1 T&E: Camphuis et al. 2026, Phys. Rev. D 113, 083504, arXiv:2506.20707 (the SPT-lite likelihood
  is its Appendix F); Balkenhol 2025, Open J. Astrophys., doi:10.33232/001c.129886, arXiv:2412.00826 (the
  CMB-lite compression); Quan et al. 2026, arXiv:2603.20163 (the D1 maps, listed by spt_candl_data).

Every data set Kosmulator includes, with what to cite: `Observations/DATA_SOURCES.md`.
