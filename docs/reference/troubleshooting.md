# Troubleshooting

Start with `kosmulator-doctor`: it checks the installation and gives, for anything missing, the command that fixes it. The sections below cover the messages people meet most often.

::::{dropdown} `Missing prior for 'x'. Please add prior_limits['x'] = (low, high).`
A model or data set samples a parameter that has no flat prior in `prior_limits` in `Kosmulator.py`. Add it, and preferably a value in `reference_values` as well (see [Configuring a run](../running.md#priors-and-reference-values)).
::::

::::{dropdown} `Model 'x' vectorised: False`
The model function did not accept an array of redshifts (or was too slow on one). It still runs, but point by point through the worker pool, with emcee. Write it with NumPy array operations, beginning with `z = _asarray(z)` and ending with `return _scalar_or_array(out)` (see [Tutorial 2](../tutorials/own_model.md#1-write-ez)).
::::

::::{dropdown} `... reached N steps without meeting the convergence rule ... Treat these results as preliminary; increase nsteps.`
The chain is not long enough yet; the message says which condition failed and roughly how many more steps are needed. Raise `nsteps` in `Kosmulator.py` and continue with `--resume` (see [Tutorial 1](../tutorials/desi_w0wa.md#4-a-chain-that-has-not-converged)).
::::

::::{dropdown} `RuntimeError: latex was not able to process the following string`
A LaTeX package is missing. The most common is `cm-super` (the message then contains `Missing cm-super package, required by Matplotlib`); on Ubuntu or WSL: `sudo apt install cm-super dvipng`. Or leave out `--latex_enabled`. See [Installation](../getting_started.md#checking-the-installation).
::::

::::{dropdown} `Another Kosmulator run is writing to 'MCMC_Chains/...'`
Two runs cannot write to the same `--output_suffix` at the same time. Use another suffix or wait for the first run to finish.
::::

::::{dropdown} CLASS does not build
The compiler output is in `Class/<model>/build.log`. Build one model by itself with `kosmulator-doctor --build-class <model>` to see the error straight away. A missing compiler, `make` or Python headers are reported by `kosmulator-doctor`; with the conda environment they come from conda-forge.
::::

::::{dropdown} `No model-specific CMB helper (x_CMB). Using generic CLASS fallback.`
The model has no `<model>_CMB` function, so its CMB spectra are computed with stock ΛCDM settings that ignore the model's parameters. Add a helper (see [Tutorial 3](../tutorials/cmb_model.md)).
::::

::::{dropdown} `CMB_lowl through clipy needs NumPy >= 2.0`
clipy's low-$\ell$ EE likelihood uses a NumPy 2 function. Update NumPy (the conda environment already has NumPy 2), or install the compiled CLIK ([Advanced installation](advanced_installation.md)).
::::

::::{dropdown} A run on a cluster without MPI tries to build an MPI pool
Kosmulator takes the environment variables `SLURM_NTASKS`, `PMI_SIZE`, `PMI_RANK` and `OMPI_COMM_WORLD_SIZE` as a sign of an MPI run. For a Slurm job without MPI start it with `env -u SLURM_NTASKS -u PMI_SIZE -u PMI_RANK python Kosmulator.py ...` (see [Clusters and long runs](../running.md#in-parallel-and-on-clusters)).
::::

::::{dropdown} Everything is very slow on WSL
Check that Kosmulator is in your Linux home folder (`~/`), not on `/mnt/c` or `/mnt/d`.
::::

::::{dropdown} Compiled CLIK imports but a likelihood does not load
Usually one of: a missing library of `libclik.so` (check with `ldd`), a wrong `CLIKROOT`, `PYTHONPATH` or `LD_LIBRARY_PATH`, or an incomplete `.clik` folder.
::::

Still stuck? Open an issue on [GitHub](https://github.com/renierht/Kosmulator/issues) with the output of `kosmulator-doctor` and the run log.
