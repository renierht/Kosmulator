# Automated tests

These tests check configuration, existing-chain loading, statistical calculations, physical boundaries and plotting metadata. They complement the scientific checks in [validation/](../validation/README.md).

Run from the repository root in your Kosmulator environment:

    MPLBACKEND=Agg python -m unittest discover -s tests -p 'test*.py' -v

The automated suite is not a reproduction of the four original paper chains. To generate and analyse your own chains, follow the [SAIP reproduction guide](../reproducibility/SAIP2026_IDE/REPRODUCE.md).
