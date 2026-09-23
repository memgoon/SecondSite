#!/usr/bin/env python3
"""Finalize checkpoint 4 after exact-master build and cached-author validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
REPORTS = PACKAGE / "reports"
MANIFESTS = PACKAGE / "manifests"
VALIDATION = PACKAGE / "validation"

BUILD_SUMMARY = VALIDATION / "CHECKPOINT4A_BUILD_SUMMARY.json"
AUTHOR_SUMMARY = VALIDATION / "CHECKPOINT4B_LOCAL_AUTHOR_VALIDATION.json"
MASTER = DATA / "BIOLIP_EXACT_OBSERVATION_MASTER.tsv.gz"
AUTHOR_AUDIT = DATA / "CHECKPOINT4_LOCAL_AUTHOR_MAPPING_VALIDATION.tsv.gz"
ELIGIBILITY = DATA / "BIOLIP_SITE_MAPPING_ELIGIBILITY.tsv.gz"
ELIGIBILITY_COUNTS = DATA / "CHECKPOINT4_FINAL_ELIGIBILITY_COUNTS.tsv"
INPUT_HASHES = MANIFESTS / "CHECKPOINT4C_INPUT_HASHES.tsv"
REPORT = REPORTS / "CHECKPOINT4_EXACT_OBSERVATION_MAPPING.md"
VALIDATION_OUT = VALIDATION / "CHECKPOINT4_VALIDATION.json"

EXPECTED_ROWS = 989058
MIN_CANONICAL_ONLY_QUERY_COVERAGE = 0.50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def file_record(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    for path in (BUILD_SUMMARY, AUTHOR_SUMMARY, MASTER, AUTHOR_AUDIT):
        if not path.is_file():
            raise FileNotFoundError(path)
    build = json.loads(BUILD_SUMMARY.read_text())
    author = json.loads(AUTHOR_SUMMARY.read_text())
    if build.get("status") != "validated" or author.get("status") != "validated":
        raise RuntimeError("checkpoint 4a/4b inputs are not validated")
    for record in build.get("outputs", {}).values():
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint-4a output hash mismatch: {path}")
    author_record = author["output"]
    if sha256(Path(author_record["path"])) != author_record["sha256"]:
        raise RuntimeError("checkpoint-4b audit hash mismatch")

    con = duckdb.connect(database=":memory:")
    master_sql = (
        "read_csv_auto(?, delim='\\t', header=true, compression='gzip', "
        "all_varchar=true, sample_size=-1)"
    )
    audit_sql = (
        "read_csv_auto(?, delim='\\t', header=true, compression='gzip', "
        "all_varchar=true, sample_size=-1)"
    )
    source_rows = int(con.execute(f"SELECT count(*) FROM {master_sql}", [str(MASTER)]).fetchone()[0])
    if source_rows != EXPECTED_ROWS:
        raise RuntimeError(f"master row count is {source_rows}, expected {EXPECTED_ROWS}")
    duplicate_observation_ids = int(
        con.execute(
            f"SELECT count(*) - count(DISTINCT observation_id) FROM {master_sql}",
            [str(MASTER)],
        ).fetchone()[0]
    )
    if duplicate_observation_ids:
        raise RuntimeError("observation_id is not unique")
    duplicate_audit_ids = int(
        con.execute(
            f"SELECT count(*) - count(DISTINCT observation_id) FROM {audit_sql}",
            [str(AUTHOR_AUDIT)],
        ).fetchone()[0]
    )
    if duplicate_audit_ids:
        raise RuntimeError("local author audit observation_id is not unique")

    temporary = ELIGIBILITY.with_name(ELIGIBILITY.name + ".tmp")
    query = f"""
        COPY (
          WITH master AS (
            SELECT * FROM {master_sql}
          ), audit AS (
            SELECT observation_id, validation_status AS local_author_validation_status
            FROM {audit_sql}
          )
          SELECT
            CAST(master.source_ordinal AS BIGINT) AS source_ordinal,
            master.observation_id,
            master.pdb_id,
            master.receptor_chain,
            master.ligand_ccd,
            master.uniprot_resolved,
            master.site_mapping_status AS checkpoint4a_site_mapping_status,
            lower(master.mapping_usable_for_site_clustering) = 'true' AS checkpoint4a_mapping_usable,
            CAST(master.canonical_alignment_query_coverage AS DOUBLE) AS canonical_alignment_query_coverage,
            coalesce(audit.local_author_validation_status, 'not_in_local_paired_cache') AS local_author_validation_status,
            CASE
              WHEN lower(master.mapping_usable_for_site_clustering) <> 'true' THEN false
              WHEN master.site_mapping_status = 'mapped_canonical_complete_sifts_unavailable_or_partial'
                   AND CAST(master.canonical_alignment_query_coverage AS DOUBLE) < {MIN_CANONICAL_ONLY_QUERY_COVERAGE}
                THEN false
              WHEN audit.local_author_validation_status IN
                   ('complete_position_disagreement', 'partial_disagreement') THEN false
              ELSE true
            END AS final_site_mapping_usable,
            CASE
              WHEN lower(master.mapping_usable_for_site_clustering) <> 'true'
                THEN 'checkpoint4a_' || master.site_mapping_status
              WHEN master.site_mapping_status = 'mapped_canonical_complete_sifts_unavailable_or_partial'
                   AND CAST(master.canonical_alignment_query_coverage AS DOUBLE) < {MIN_CANONICAL_ONLY_QUERY_COVERAGE}
                THEN 'canonical_only_query_coverage_below_0.50'
              WHEN audit.local_author_validation_status = 'complete_position_disagreement'
                THEN 'direct_author_complete_position_disagreement'
              WHEN audit.local_author_validation_status = 'partial_disagreement'
                THEN 'direct_author_partial_position_disagreement'
              ELSE 'eligible'
            END AS final_eligibility_reason
          FROM master
          LEFT JOIN audit USING (observation_id)
          ORDER BY CAST(master.source_ordinal AS BIGINT)
        ) TO '{temporary}' (HEADER, DELIMITER '\t', COMPRESSION GZIP)
    """
    con.execute(query, [str(MASTER), str(AUTHOR_AUDIT)])
    os.replace(temporary, ELIGIBILITY)

    eligibility_scan = (
        "read_csv_auto(?, delim='\\t', header=true, compression='gzip', "
        "all_varchar=true, sample_size=-1)"
    )
    eligibility_rows = int(
        con.execute(f"SELECT count(*) FROM {eligibility_scan}", [str(ELIGIBILITY)]).fetchone()[0]
    )
    if eligibility_rows != EXPECTED_ROWS:
        raise RuntimeError("eligibility ledger row count mismatch")
    counts = con.execute(
        f"""
        SELECT final_eligibility_reason, count(*) AS rows
        FROM {eligibility_scan}
        GROUP BY 1 ORDER BY rows DESC, final_eligibility_reason
        """,
        [str(ELIGIBILITY)],
    ).fetchdf()
    final_usable = int(
        con.execute(
            f"SELECT count(*) FROM {eligibility_scan} WHERE lower(final_site_mapping_usable)='true'",
            [str(ELIGIBILITY)],
        ).fetchone()[0]
    )
    con.close()
    write_tsv(counts, ELIGIBILITY_COUNTS)

    input_records = pd.DataFrame(
        [
            {"asset_id": "checkpoint4a_summary", **file_record(BUILD_SUMMARY)},
            {"asset_id": "checkpoint4a_master", **file_record(MASTER)},
            {"asset_id": "checkpoint4b_summary", **file_record(AUTHOR_SUMMARY)},
            {"asset_id": "checkpoint4b_author_audit", **file_record(AUTHOR_AUDIT)},
        ]
    )
    write_tsv(input_records, INPUT_HASHES)

    site_counts = build["site_mapping_status_counts"]
    agreement_fraction = author.get("exact_agreement_fraction_among_directly_complete")
    report = f"""# Checkpoint 4: exact BioLiP observation and residue mapping

Status: **validated**

## Observation master

- Source rows preserved: **{source_rows:,}**
- Unique natural observations: **{build['unique_natural_observations']:,}**
- Duplicate natural keys / excess rows: **{build['duplicate_natural_keys']:,} / {build['duplicate_excess_rows']:,}**
- Unique PDBs: **{build['unique_pdbs']:,}**
- Deduplicated receptor-sequence records: **{build['unique_receptor_sequence_records']:,}**
- Raw and offset-derived source rows matched byte-for-byte: **yes**

The master remains at source-row grain: PDB, receptor chain, binding-site code,
ligand CCD, ligand chain, ligand serial/auth sequence ID, binding residues and
source ordinal.  Repeated long receptor sequences are stored once in a separate
sequence manifest.

## Residue mapping

Primary path: BioLiP observed-sequence position -> CIF polymer label position ->
bulk SIFTS segment -> UniProt residue.  Direct alignment to the cached canonical
UniProt sequence is recorded as a consistency check/rescue.

- Binding residues: **{build['binding_residues']:,}**
- Binding residues mapped before final QC: **{build['mapped_binding_residues']:,}**
- Rows marked usable by checkpoint 4a: **{build['site_clustering_usable_rows']:,}**
- Rows usable after canonical-coverage and local-author QC: **{final_usable:,}**

Checkpoint-4a site status counts:
{os.linesep.join(f'- `{key}`: {value:,}' for key, value in sorted(site_counts.items(), key=lambda item: (-item[1], item[0])))}

## Independent local coordinate check

For observations whose PDB already had a paired local mmCIF and SIFTS JSON, the
sequence route was compared with the independent route raw author residue ->
mmCIF label_seq -> local SIFTS JSON -> UniProt.

- Eligible observations on the paired-cache subset: **{author['eligible_observations_on_paired_cache']:,}**
- Rows with a complete direct-author mapping: **{author['directly_complete_rows']:,}**
- Exact full agreement: **{author['exact_full_agreement_rows']:,}**
- Complete disagreements: **{author['complete_position_disagreement_rows']:,}**
- Exact agreement among directly complete rows: **{agreement_fraction:.2%}**

Any complete or partial position disagreement is excluded in the final
eligibility ledger.  A canonical-only rescue is also excluded when less than
50% of the observed receptor sequence aligns, because a small local match in a
fusion/multicomponent chain is not sufficient for automatic site clustering.

## Final eligibility reasons

{os.linesep.join(f'- `{row.final_eligibility_reason}`: {int(row.rows):,}' for row in counts.itertuples(index=False))}

## Retained audit flags

- Resolved UniProt accession differs from the raw BioLiP annotation:
  **{build['uniprot_resolution_status_counts'].get('resolved_conflicts_with_raw', 0):,} rows**.
  The raw accession is retained.  These are not automatically discarded when
  a unique SIFTS mapping, or SIFTS plus canonical-sequence agreement, resolves
  the observed chain to a different accession.
- CCD absent from the frozen wwPDB compound cache:
  **{build['chemical_mapping_status_counts'].get('ccd_not_in_wwpdb_cache', 0):,} rows**.
- CCD present but without an InChIKey:
  **{build['chemical_mapping_status_counts'].get('wwpdb_inchikey_missing', 0):,} row**.

The last two categories can retain valid site coordinates but cannot enter a
chemical ranking until the ligand identity is resolved.  Site-mapping and
chemical-mapping eligibility therefore remain separate.

## Scope boundary

This checkpoint fixes observation identity and UniProt residue coordinates.  It
does not calculate an allosteric score and does not yet merge observations into
global site clusters.  Biological-assembly/interface geometry is evaluated
when coordinate-dependent site features are constructed; it is not inferred
from sequence mapping alone.
"""
    atomic_text(REPORT, report)

    outputs = {
        "eligibility_ledger": file_record(ELIGIBILITY),
        "eligibility_counts": file_record(ELIGIBILITY_COUNTS),
        "input_hashes": file_record(INPUT_HASHES),
        "report": file_record(REPORT),
    }
    validation = {
        "checkpoint": 4,
        "status": "validated",
        "scope": "exact BioLiP source observations and residue-level UniProt site mapping",
        "source_rows": source_rows,
        "observation_id_duplicates": duplicate_observation_ids,
        "eligibility_rows": eligibility_rows,
        "final_site_mapping_usable_rows": final_usable,
        "minimum_canonical_only_query_coverage": MIN_CANONICAL_ONLY_QUERY_COVERAGE,
        "local_author_mapping_validation": author,
        "checkpoint4a_build_summary": build,
        "final_eligibility_counts": {
            str(row.final_eligibility_reason): int(row.rows) for row in counts.itertuples(index=False)
        },
        "outputs": outputs,
    }
    atomic_json(VALIDATION_OUT, validation)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
