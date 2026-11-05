# Citing Kosmulator

Kosmulator is an actively developed research framework. If you use it in your work, please cite the software paper, published in the proceedings of the South African Gravity Society Conference 2025 ([doi:10.62869/001c.172550](https://doi.org/10.62869/001c.172550); [arXiv:2602.08424](https://arxiv.org/abs/2602.08424)):

```bibtex
@INPROCEEDINGS{Hough2026Kosmulator,
       author = {{Hough}, Renier T. and {Rugg}, Robert and {Sahlu}, Shambel and {Abebe}, Amare},
        title = "{Kosmulator: A Python Framework for Cosmological Inference With MCMC}",
    booktitle = {SAGS2025 Proceedings of the South African Gravity Society Conference 2025},
    publisher = {Axiom Academic Publishers},
      chapter = {2},
        pages = {7--19},
         year = 2026,
          doi = {10.62869/001c.172550},
archivePrefix = {arXiv},
       eprint = {2602.08424},
 primaryClass = {astro-ph.CO}
}
```

and the paper of the original implementation of the code:

```bibtex
@ARTICLE{2020EPJC...80..787H,
       author = {{Hough}, R.~T. and {Abebe}, A. and {Ferreira}, S.~E.~S.},
        title = "{Viability tests of f(R)-gravity models with Supernovae Type 1A data}",
      journal = {European Physical Journal C},
     keywords = {General Relativity and Quantum Cosmology, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2020,
        month = aug,
       volume = {80},
       number = {8},
          eid = {787},
        pages = {787},
          doi = {10.1140/epjc/s10052-020-8342-7},
archivePrefix = {arXiv},
       eprint = {1911.05983},
 primaryClass = {gr-qc},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020EPJC...80..787H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## The data and codes you used

**Remember to cite the original papers of the data you use.** Kosmulator makes this easy:

- Every run writes `Statistical_analysis_tables/<suffix>/citations.md`: what to cite for exactly the data sets, samplers and codes that run used, with the acknowledgements some data licences ask for (DESI's), and their BibTeX in `citations.bib` next to it. Its path is printed at the end of the run.
- [`Observations/DATA_SOURCES.md`](https://github.com/renierht/Kosmulator/blob/main/Observations/DATA_SOURCES.md) lists every data set Kosmulator includes: its files, where they come from, the licence status of the source and the papers to cite.
- [`Observations/references.bib`](https://github.com/renierht/Kosmulator/blob/main/Observations/references.bib) holds the BibTeX of every reference in `DATA_SOURCES.md`.

For example, the run of [Tutorial 1](../tutorials/desi_w0wa.md) lists Kosmulator's two papers, DESI DR2 (with DESI's acknowledgement request), the two Pantheon+ papers, zeus and ensemble slice sampling, and GetDist.
