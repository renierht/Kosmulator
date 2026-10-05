# Scientific validation

- [VALIDATION.md](VALIDATION.md) records the analytical and numerical checks of the IDE CLASS background implementation, including zero-coupling limits and sound horizons.
- [validate_paper_postprocessing.py](validate_paper_postprocessing.py) checks the statistical results using the four original SAIP analysis chains.

## Validate the original paper results

This script requires the original four chain files, the Kosmulator dependencies and prepared matching CLASS binaries. Run from the repository root:

    MPLBACKEND=Agg python -u validation/validate_paper_postprocessing.py --chain-root /absolute/path/to/MCMC_Chains

Replace the example path with the directory containing the original chain subdirectories. The validator loads the chains read-only, checks their statistics and likelihood polishing, and writes validation outputs. It does not generate replacement chains.

For generating and analysing independent chains, use the [SAIP reproduction guide](../reproducibility/SAIP2026_IDE/REPRODUCE.md). For automated software checks, see [tests/](../tests/README.md).
