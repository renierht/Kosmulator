# Citing Kosmulator

Kosmulator is an actively developed research framework. If you use it in your work, please cite the software paper:

```bibtex
@ARTICLE{2026arXiv260208424H,
       author = {{Hough}, Renier T. and {Rugg}, Robert and {Sahlu}, Shambel and {Abebe}, Amare},
        title = "{Kosmulator: A Python framework for cosmological inference with MCMC}",
      journal = {arXiv e-prints},
     keywords = {Cosmology and Nongalactic Astrophysics, General Relativity and Quantum Cosmology},
         year = 2026,
        month = feb,
          eid = {arXiv:2602.08424},
        pages = {arXiv:2602.08424},
          doi = {10.48550/arXiv.2602.08424},
archivePrefix = {arXiv},
       eprint = {2602.08424},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260208424H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
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
