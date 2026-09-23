# Allosteric–orthosteric preprocessing v1

This package constructs a two-class, exact UniProt/full-InChIKey Uncontrolled dataset from ASD, GtoPdb, KLIFS, and the frozen K03 BRENDA extraction. It contains no decoys and performs no protein matching or ligand-property grouping.

## Reproduction order

1. `parse_gtop_orthosteric_references.py`
2. `resolve_brenda_exact_structures.py --verify-existing` (or `--run` only when exact Tuna resolution is intentionally repeated)
3. `build_uncontrolled_allosteric_orthosteric.py`
4. `validate_uncontrolled_allosteric_orthosteric.py --write-results`
5. `finalize_uncontrolled_package.py`
6. `validate_uncontrolled_allosteric_orthosteric.py --validate-only`
