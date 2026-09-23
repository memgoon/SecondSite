# Source provenance and validation scope

Analysis implementations are byte-identical to the retained files in the verified v1 source snapshot. `provenance/SOURCE_INDEX.tsv` records original relative paths, stage assignments and hashes. `EXCLUDED_FILES.tsv` lists the intentionally omitted files; it does not contain their source code.

Fresh checks validate source hashes, Python and shell syntax, eight-stage membership, exclusion of the dedicated website-export and manuscript-figure packages, and final archive integrity. Package unit tests are rerun. Earlier scientific synthetic test results are retained separately with explicit provenance; they are not presented as newly executed scientific analyses.

This is a publication source snapshot, not a portable runtime distribution. Original workspace paths, external dependencies and historical experiment contracts remain in the source. Portability and dependency audit tables identify known references; dynamic imports are not exhaustively resolved. No analysis is rerun and no network or GitHub operation is performed when preparing this subset.

Some historical package documentation and integrated report functions mention figures or tabular exports. They are preserved with their analysis modules for provenance; the dedicated publication-figure and SecondSite export packages are intentionally absent. No website deployment configuration or manuscript figure assets are distributed.
