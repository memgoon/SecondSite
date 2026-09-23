#!/usr/bin/env python3
"""K03: extract general reported kinetic parameters from BRENDA 2026.1.

The parser is deliberately independent of the prior 25,333-pair overlap.  It
streams the complete BRENDA JSON archive, preserves every source parameter
record, and emits normalized experiment, participant, condition, reference,
entity-mapping, and parameter tables.  Values reported only in comments
(S0.5 and Hill coefficients) are retained as lower-confidence explicit-text
extractions; they are never promoted to Q1/Q2 automatically.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import platform
import re
import ssl
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path


PARSER_VERSION = "k03_brenda_parameters_v1.0"
SOURCE_DATABASE = "BRENDA"
EXPECTED_RELEASE = "2026.1"
SPARQL_ENDPOINT = "https://sparql.dsmz.de/api/brenda"
BRENDA_DOWNLOAD_PAGE = "https://www.brenda-enzymes.org/download.php"
BRENDA_DATAFIELDS_PAGE = "https://www.brenda-enzymes.org/datafields.php"

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "analysis/kinetic_allostery_model_v1"
DEFAULT_ARCHIVE = (
    ROOT
    / "analysis/expanded_allosteric_database/raw/brenda_2026_1/"
    "brenda_2026_1.json.tar.gz"
)

PARAMETER_FIELDS = {
    "km_value": ("Km", "mM", "substrate"),
    "turnover_number": ("kcat", "1/s", "substrate"),
    "kcat_km_value": ("kcat_over_Km", "mM/s", "substrate"),
    "ki_value": ("Ki", "mM", "inhibitor"),
}

FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
UNIPROT_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9]{5,9}(?:-\d+)?$")
MUTATION_PATTERN = re.compile(r"(?<![A-Za-z0-9])([A-Z][0-9]{1,5}[A-Z])(?![A-Za-z0-9])")
S05_PATTERN = re.compile(
    r"(?i)(?:\bS\s*(?:0[\.,]5|\(\s*0[\.,]5\s*\))|"
    r"\bhalf[- ]saturation(?:\s+constant)?)"
    r"\s*(?:value(?:value)?|constant)?\s*(?:of|is|=|:)?\s*"
    r"(?P<relation><=|>=|<|>|~|ca\.?|approximately|about)?\s*"
    r"(?P<value>" + FLOAT_PATTERN + r")\s*"
    r"(?P<unit>nM|uM|µM|μM|mM|M)?\b"
)
NH_PATTERN = re.compile(
    r"(?i)(?:\bHill\s+(?:coefficient|number|constant)|\bnH\b)"
    r"\s*(?:of|is|=|:)?\s*"
    r"(?P<relation><=|>=|<|>|~|ca\.?|approximately|about)?\s*"
    r"(?P<value>" + FLOAT_PATTERN + r")\b"
)


def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def norm_name(value):
    value = clean_text(value).casefold()
    return value.replace("−", "-").replace("–", "-").replace("—", "-")


def join_values(values):
    return ";".join(sorted({clean_text(x) for x in values if clean_text(x)}))


def stable_id(prefix, *parts):
    text = "|".join(clean_text(x) for x in parts)
    return "{}_{}".format(prefix, hashlib.sha256(text.encode("utf-8")).hexdigest()[:20])


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_count(path):
    path = Path(path)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_tsv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def iter_brenda_data(stream, chunk_size=8 * 1024 * 1024):
    """Yield EC/object pairs below the top-level BRENDA data object."""

    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False

    def read_more():
        nonlocal buffer, eof
        chunk = stream.read(chunk_size)
        if chunk:
            buffer += chunk
            return True
        eof = True
        return False

    marker = '"data"'
    while True:
        index = buffer.find(marker, position)
        if index >= 0:
            brace = buffer.find("{", index + len(marker))
            if brace >= 0:
                position = brace + 1
                break
        if eof:
            raise ValueError("BRENDA JSON has no top-level data object")
        if len(buffer) > 64:
            buffer = buffer[-64:]
            position = 0
        read_more()

    while True:
        while True:
            while position < len(buffer) and buffer[position] in " \t\r\n,":
                position += 1
            if position < len(buffer):
                break
            if not read_more():
                raise ValueError("Unexpected EOF within BRENDA data object")
        if buffer[position] == "}":
            return
        while True:
            try:
                key, key_end = decoder.raw_decode(buffer, position)
                break
            except json.JSONDecodeError:
                if not read_more():
                    raise
        position = key_end
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                break
            if not read_more():
                raise ValueError("Unexpected EOF after BRENDA data key")
        if buffer[position] != ":":
            raise ValueError("Expected colon after BRENDA data key {}".format(key))
        position += 1
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        while True:
            try:
                value, value_end = decoder.raw_decode(buffer, position)
                break
            except json.JSONDecodeError:
                if not read_more():
                    raise
        yield key, value
        buffer = buffer[value_end:]
        position = 0


def archive_member_and_release(tar_path):
    with tarfile.open(str(tar_path), "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.name.endswith(".json")]
        if len(members) != 1:
            raise ValueError("Expected exactly one JSON member; found {}".format(len(members)))
        raw = archive.extractfile(members[0])
        if raw is None:
            raise ValueError("Could not open BRENDA JSON member")
        prefix = raw.read(4096).decode("utf-8", errors="replace")
        match = re.search(r'"release"\s*:\s*"([^"]+)"', prefix)
        if not match:
            raise ValueError("Could not determine BRENDA release")
        return members[0].name, match.group(1)


def brenda_json_stream(tar_path):
    with tarfile.open(str(tar_path), "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.name.endswith(".json")]
        if len(members) != 1:
            raise ValueError("Expected exactly one JSON member; found {}".format(len(members)))
        raw = archive.extractfile(members[0])
        if raw is None:
            raise ValueError("Could not open BRENDA JSON member")
        with raw, io.TextIOWrapper(raw, encoding="utf-8") as text_handle:
            yield from iter_brenda_data(text_handle)


def ssl_context():
    cafile = Path("/etc/ssl/certs/ca-certificates.crt")
    if cafile.exists():
        return ssl.create_default_context(cafile=str(cafile))
    return ssl.create_default_context()


def fetch_compound_structure_map(cache_path, metadata_path, retrieval_date):
    query = """PREFIX d3o: <https://purl.dsmz.de/schema/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?name ?inchikey WHERE {
  ?compound a d3o:ChemicalCompound ;
            rdfs:label ?name ;
            d3o:hasStructure ?structure .
  ?structure d3o:hasInChIKey ?inchikey .
  FILTER(STRLEN(STR(?inchikey)) > 0)
}"""
    url = SPARQL_ENDPOINT + "?" + urllib.parse.urlencode({"query": query})
    error_messages = []
    payload = None
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Heesu-Allostery-K03-BRENDA/1.0"},
            )
            with urllib.request.urlopen(
                request, timeout=600, context=ssl_context()
            ) as response:
                payload = json.load(response)
            break
        except Exception as exc:
            error_messages.append("attempt_{}:{}".format(attempt, clean_text(exc)))
            if attempt < 3:
                time.sleep(2 ** attempt)
    if payload is None:
        write_json(
            metadata_path,
            {
                "status": "fetch_failed",
                "endpoint": SPARQL_ENDPOINT,
                "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "retrieval_date": retrieval_date,
                "errors": error_messages,
            },
        )
        return False

    rows = []
    for binding in payload.get("results", {}).get("bindings", []):
        name = clean_text((binding.get("name") or {}).get("value", ""))
        inchikey = clean_text((binding.get("inchikey") or {}).get("value", ""))
        inchikey = inchikey.replace("InChIKey=", "", 1).strip()
        if name and inchikey:
            rows.append(
                {
                    "compound_name": name,
                    "normalized_name": norm_name(name),
                    "standard_inchikey": inchikey,
                    "connectivity_key": inchikey.split("-")[0],
                }
            )
    unique = {
        (r["normalized_name"], r["standard_inchikey"], r["compound_name"]): r
        for r in rows
    }
    rows = sorted(
        unique.values(),
        key=lambda r: (r["normalized_name"], r["standard_inchikey"], r["compound_name"]),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound_name",
                "normalized_name",
                "standard_inchikey",
                "connectivity_key",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        metadata_path,
        {
            "status": "success",
            "endpoint": SPARQL_ENDPOINT,
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "retrieval_date": retrieval_date,
            "mapping_rows": len(rows),
            "errors_before_success": error_messages,
        },
    )
    return True


def load_compound_structure_map(cache_path):
    mapping = defaultdict(set)
    names = defaultdict(set)
    if not Path(cache_path).exists():
        return mapping, names, 0
    with gzip.open(cache_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            mapping[row["normalized_name"]].add(row["standard_inchikey"])
            names[row["normalized_name"]].add(row["compound_name"])
    return mapping, names, sum(len(v) for v in mapping.values())


def extract_participant_names(raw_value):
    names = [clean_text(x) for x in re.findall(r"\{([^{}]+)\}", raw_value or "")]
    return [x for x in names if x]


def canonical_float(value):
    return "{:.12g}".format(float(value))


def parse_reported_value(raw_value):
    raw_value = clean_text(raw_value)
    numeric_text = raw_value.split("{", 1)[0].strip()
    result = {
        "raw_numeric_text": numeric_text or "NA_not_reported",
        "numeric_value": "NA_not_single_value",
        "numeric_value_min": "NA_not_range",
        "numeric_value_max": "NA_not_range",
        "numeric_uncertainty": "NA_not_reported",
        "relation_operator": "=",
        "numeric_parse_status": "unparsed",
    }
    if not numeric_text:
        result["numeric_parse_status"] = "missing_numeric_text"
        return result
    if re.fullmatch(r"-999(?:\.0+)?", numeric_text):
        result["numeric_parse_status"] = "sentinel_minus_999_additional_information"
        return result

    text = numeric_text.replace("−", "-").replace("±", "+/-")
    relation_match = re.match(
        r"^\s*(<=|>=|<|>|~|ca\.?|approx(?:imately)?\.?|about)?\s*(.*)$",
        text,
        flags=re.I,
    )
    relation = clean_text(relation_match.group(1) if relation_match else "")
    body = clean_text(relation_match.group(2) if relation_match else text)
    if relation:
        result["relation_operator"] = (
            "~" if relation.casefold().startswith(("ca", "approx", "about")) else relation
        )

    plusminus = re.fullmatch(
        r"({0})\s*\+/-\s*({0})".format(FLOAT_PATTERN), body
    )
    if plusminus:
        value = float(plusminus.group(1))
        uncertainty = abs(float(plusminus.group(2)))
        result.update(
            {
                "numeric_value": canonical_float(value),
                "numeric_uncertainty": canonical_float(uncertainty),
                "numeric_parse_status": "parsed_value_with_uncertainty",
            }
        )
        if value <= 0:
            result["numeric_parse_status"] = "nonpositive_value"
        return result

    value_range = re.fullmatch(
        r"({0})\s*(?:-|to)\s*({0})".format(FLOAT_PATTERN), body, flags=re.I
    )
    if value_range:
        low = float(value_range.group(1))
        high = float(value_range.group(2))
        if low > high:
            low, high = high, low
        result.update(
            {
                "numeric_value_min": canonical_float(low),
                "numeric_value_max": canonical_float(high),
                "relation_operator": "range",
                "numeric_parse_status": "parsed_range",
            }
        )
        if low <= 0 or high <= 0:
            result["numeric_parse_status"] = "nonpositive_range"
        return result

    single = re.fullmatch(r"({})".format(FLOAT_PATTERN), body)
    if single:
        value = float(single.group(1))
        result.update(
            {
                "numeric_value": canonical_float(value),
                "numeric_parse_status": "parsed_single_value",
            }
        )
        if value <= 0:
            result["numeric_parse_status"] = "nonpositive_value"
        return result

    numbers = re.findall(FLOAT_PATTERN, body)
    if len(numbers) == 1 and re.fullmatch(
        r"(?i)\s*(?:value\s*)?" + FLOAT_PATTERN + r"\s*", body
    ):
        value = float(numbers[0])
        result.update(
            {
                "numeric_value": canonical_float(value),
                "numeric_parse_status": "parsed_single_value",
            }
        )
        if value <= 0:
            result["numeric_parse_status"] = "nonpositive_value"
        return result

    result["numeric_parse_status"] = (
        "multiple_or_complex_values_unparsed" if len(numbers) > 1 else "numeric_format_unparsed"
    )
    return result


def normalize_relation(value):
    value = clean_text(value)
    if value.casefold().startswith(("ca", "approx", "about")):
        return "~"
    return value or "="


def explicit_secondary_parameters(comment):
    comment = clean_text(comment)
    rows = []
    for match in S05_PATTERN.finditer(comment):
        raw_unit = clean_text(match.group("unit")) or "NA_not_reported"
        value = float(match.group("value"))
        normalized = "NA_missing_or_unsupported_unit"
        normalized_unit = "NA_missing_or_unsupported_unit"
        factor = {
            "nm": 1e-6,
            "um": 1e-3,
            "µm": 1e-3,
            "μm": 1e-3,
            "mm": 1.0,
            "m": 1e3,
        }.get(raw_unit.casefold())
        if factor is not None:
            normalized = canonical_float(value * factor)
            normalized_unit = "mM"
        rows.append(
            {
                "parameter_type": "S0.5",
                "raw_value": clean_text(match.group(0)),
                "raw_numeric_text": match.group("value"),
                "numeric_value": canonical_float(value),
                "numeric_value_min": "NA_not_range",
                "numeric_value_max": "NA_not_range",
                "numeric_uncertainty": "NA_not_reported",
                "relation_operator": normalize_relation(match.group("relation")),
                "original_unit": raw_unit,
                "normalized_value": normalized,
                "normalized_value_min": "NA_not_range",
                "normalized_value_max": "NA_not_range",
                "normalized_unit": normalized_unit,
                "numeric_parse_status": (
                    "parsed_comment_value_and_unit"
                    if factor is not None
                    else "parsed_comment_value_missing_unit"
                ),
                "parameter_origin": "comment_explicit_secondary",
            }
        )
    for match in NH_PATTERN.finditer(comment):
        value = float(match.group("value"))
        rows.append(
            {
                "parameter_type": "nH",
                "raw_value": clean_text(match.group(0)),
                "raw_numeric_text": match.group("value"),
                "numeric_value": canonical_float(value),
                "numeric_value_min": "NA_not_range",
                "numeric_value_max": "NA_not_range",
                "numeric_uncertainty": "NA_not_reported",
                "relation_operator": normalize_relation(match.group("relation")),
                "original_unit": "dimensionless",
                "normalized_value": canonical_float(value),
                "normalized_value_min": "NA_not_range",
                "normalized_value_max": "NA_not_range",
                "normalized_unit": "dimensionless",
                "numeric_parse_status": "parsed_comment_value",
                "parameter_origin": "comment_explicit_secondary",
            }
        )
    return rows


def parse_conditions(comment):
    comment = clean_text(comment)
    ph_matches = re.findall(
        r"(?i)\bpH\s*(?:=|of|at)?\s*({})".format(FLOAT_PATTERN), comment
    )
    temp_matches = re.findall(
        r"(?i)(-?\d+(?:\.\d+)?)\s*(?:°|deg(?:ree)?s?\s*)?C\b", comment
    )
    ph_values = sorted({canonical_float(x) for x in ph_matches})
    temp_values = sorted({canonical_float(x) for x in temp_matches})
    concentration_mentions = re.findall(
        r"(?i)\b(?:\d+(?:\.\d+)?|\.\d+)\s*(?:nM|uM|µM|μM|mM|M)\b",
        comment,
    )
    incubation_mentions = re.findall(
        r"(?i)\b(?:\d+(?:\.\d+)?|\.\d+)\s*(?:s|sec(?:onds?)?|min(?:utes?)?|h|hr|hours?)\b",
        comment,
    )
    has_buffer = bool(
        re.search(r"(?i)\bbuffer\b|\bTris\b|\bHEPES\b|\bphosphate\b|\bMES\b", comment)
    )
    has_ionic = bool(re.search(r"(?i)ionic strength|\bNaCl\b|\bKCl\b", comment))
    has_enzyme_conc = bool(re.search(r"(?i)(?:enzyme|protein) concentration", comment))
    has_substrate_conc = bool(re.search(r"(?i)substrate concentration", comment))
    has_modifier_conc = bool(re.search(r"(?i)(?:inhibitor|activator|modifier) concentration", comment))
    has_cofactor = bool(re.search(r"(?i)\bcofactor\b|\bNADP?H?\b|\bATP\b|\bADP\b", comment))
    has_detergent = bool(re.search(r"(?i)\bdetergent\b|\bTriton\b|\bTween\b", comment))
    if ph_values and temp_values:
        completeness = "ph_and_temperature_reported"
    elif ph_values or temp_values:
        completeness = "one_of_ph_or_temperature_reported"
    elif comment:
        completeness = "free_text_only_no_ph_temperature"
    else:
        completeness = "no_condition_text"
    return {
        "raw_condition_text": comment or "NA_not_reported",
        "ph_values": ";".join(ph_values) if ph_values else "NA_not_reported",
        "temperature_c_values": ";".join(temp_values) if temp_values else "NA_not_reported",
        "buffer_context": comment if has_buffer else "NA_not_reported",
        "ionic_strength_context": comment if has_ionic else "NA_not_reported",
        "enzyme_concentration_context": comment if has_enzyme_conc else "NA_not_reported",
        "substrate_concentration_context": comment if has_substrate_conc else "NA_not_reported",
        "modifier_concentration_context": comment if has_modifier_conc else "NA_not_reported",
        "cofactor_context": comment if has_cofactor else "NA_not_reported",
        "detergent_context": comment if has_detergent else "NA_not_reported",
        "all_concentration_mentions": join_values(concentration_mentions) or "NA_not_reported",
        "incubation_time_mentions": join_values(incubation_mentions) or "NA_not_reported",
        "condition_completeness": completeness,
        "ph_missing": str(not bool(ph_values)).lower(),
        "temperature_missing": str(not bool(temp_values)).lower(),
    }


def protein_mapping(ec, protein_id, protein_obj):
    if not protein_obj:
        status = "missing_protein_object"
        accessions = []
    else:
        accessions = sorted({clean_text(x) for x in protein_obj.get("accessions") or [] if clean_text(x)})
        valid = [x for x in accessions if UNIPROT_PATTERN.match(x)]
        if len(accessions) == 1 and len(valid) == 1:
            status = "exact_single_uniprot"
        elif len(accessions) > 1 and len(valid) == len(accessions):
            status = "multiple_uniprot_accessions_ambiguous"
        elif accessions:
            status = "invalid_or_nonstandard_accession"
        else:
            status = "no_uniprot_accession"
    return {
        "protein_mapping_id": stable_id("K03PROT", ec, protein_id),
        "source_database": SOURCE_DATABASE,
        "source_release": EXPECTED_RELEASE,
        "brenda_ec_number": ec,
        "brenda_protein_id": clean_text(protein_id) or "NA_not_reported",
        "uniprot_accessions": join_values(accessions) or "NA_not_reported",
        "uniprot_isoform_status": (
            "isoform_accession_present"
            if any("-" in x for x in accessions)
            else "canonical_or_unspecified"
        ),
        "organism": clean_text((protein_obj or {}).get("organism", "")) or "NA_not_reported",
        "protein_comment": clean_text((protein_obj or {}).get("comment", "")) or "NA_not_reported",
        "protein_reference_ids": join_values((protein_obj or {}).get("references") or []) or "NA_not_reported",
        "mapping_method": "BRENDA_JSON_protein_accessions",
        "mapping_status": status,
        "mapping_confidence": "exact" if status == "exact_single_uniprot" else "unresolved_or_ambiguous",
        "sequence_hash": "NA_deferred_to_K07",
        "sequence_cluster_id": "NA_deferred_to_K07",
    }


def compound_mapping_for_name(name, structure_map):
    normalized = norm_name(name)
    keys = sorted(structure_map.get(normalized, set()))
    connectivities = sorted({x.split("-")[0] for x in keys})
    placeholder = normalized in {"", "more", "?", "unknown", "additional information"}
    if placeholder:
        status = "placeholder_not_a_resolved_compound"
        reason = "source_placeholder"
    elif len(keys) == 1:
        status = "exact_unique_full_inchikey"
        reason = "exact_normalized_name_unique_in_official_BRENDA_structure_map"
    elif len(keys) > 1:
        status = "multiple_full_inchikeys_ambiguous"
        reason = "normalized_name_maps_to_multiple_official_BRENDA_structures"
    else:
        status = "no_structure_mapping"
        reason = "normalized_name_absent_from_official_BRENDA_structure_map"
    return {
        "compound_mapping_id": stable_id("K03CMP", normalized or "NA"),
        "original_name": clean_text(name) or "NA_not_reported",
        "normalized_name": normalized or "NA_not_reported",
        "canonical_smiles": "NA_not_available_from_mapping_endpoint",
        "full_inchikeys": join_values(keys) or "NA_not_mapped",
        "connectivity_keys": join_values(connectivities) or "NA_not_mapped",
        "full_inchikey_count": len(keys),
        "connectivity_key_count": len(connectivities),
        "mapping_method": "exact_normalized_name_to_official_BRENDA_SPARQL_structure",
        "mapping_status": status,
        "mapping_reason": reason,
        "stereochemistry_status": (
            "single_full_inchikey"
            if len(keys) == 1
            else "unresolved_or_multiple_full_inchikeys"
        ),
        "salt_mapping_status": "NA_not_assessed",
        "bemis_murcko_scaffold": "NA_deferred_to_K07",
        "close_analogue_group": "NA_deferred_to_K07",
    }


def build_variant_index(obj):
    index = defaultdict(list)
    for entry in obj.get("protein_variants") or []:
        proteins = [clean_text(x) for x in entry.get("proteins") or []]
        references = [clean_text(x) for x in entry.get("references") or []]
        text = " | ".join(
            x
            for x in [clean_text(entry.get("value", "")), clean_text(entry.get("comment", ""))]
            if x
        )
        if not text:
            continue
        for protein_id in proteins:
            for reference_id in references or ["NA_no_reference"]:
                index[(protein_id, reference_id)].append(text)
    return index


def construct_context(entry_comment, protein_comment, linked_variants):
    texts = [x for x in [clean_text(entry_comment), clean_text(protein_comment)] if x]
    texts.extend(clean_text(x) for x in linked_variants if clean_text(x))
    combined = " | ".join(texts)
    lower = combined.casefold()
    has_mutant = bool(
        MUTATION_PATTERN.search(combined)
        or re.search(r"\bmutant\b|\bmutation\b|site[- ]directed|variant", lower)
    )
    has_wild = bool(re.search(r"wild[- ]type|\bwild type\b|\bwt\b", lower))
    if has_mutant and has_wild:
        status = "mixed_wild_type_and_mutant_comparison"
    elif has_mutant:
        status = "mutant_or_variant"
    elif has_wild:
        status = "wild_type"
    elif "recombinant" in lower:
        status = "recombinant_construct_unspecified"
    else:
        status = "construct_unspecified"
    return status, join_values(MUTATION_PATTERN.findall(combined)) or "NA_not_reported", combined or "NA_not_reported"


def reference_row(ec, reference_id, obj, raw_locator, retrieval_date):
    ref = (obj.get("reference") or {}).get(str(reference_id), {})
    source_publication_id = "BRENDA:{}:{}:REF:{}".format(EXPECTED_RELEASE, ec, reference_id)
    pmid = clean_text(ref.get("pmid", "")) or "NA_not_reported"
    title = clean_text(ref.get("title", "")) or "NA_not_reported"
    doi = clean_text(ref.get("doi", "")) or "NA_not_reported"
    family_basis = pmid if pmid != "NA_not_reported" else title
    return {
        "source_publication_id": source_publication_id,
        "source_database": SOURCE_DATABASE,
        "source_release": EXPECTED_RELEASE,
        "brenda_ec_number": ec,
        "brenda_reference_id": clean_text(reference_id),
        "pmid": pmid,
        "doi": doi,
        "title": title,
        "authors": join_values(ref.get("authors") or []) or "NA_not_reported",
        "journal": clean_text(ref.get("journal", "")) or "NA_not_reported",
        "publication_year": clean_text(ref.get("year", "")) or "NA_not_reported",
        "volume": clean_text(ref.get("vol", "")) or "NA_not_reported",
        "pages": clean_text(ref.get("pages", "")) or "NA_not_reported",
        "publication_family_id": stable_id("K03PUB", family_basis),
        "mapping_status": "exact_reference_object" if ref else "missing_reference_object",
        "raw_source_locator": raw_locator + "/reference/" + clean_text(reference_id),
        "retrieval_date": retrieval_date,
    }


def quality_tier(
    parameter_origin,
    parse_status,
    protein_status,
    compound_status,
    has_reference,
    condition_completeness,
    normalized_unit,
):
    reasons = []
    valid_parse = parse_status in {
        "parsed_single_value",
        "parsed_value_with_uncertainty",
        "parsed_range",
        "parsed_comment_value_and_unit",
        "parsed_comment_value",
    }
    if not valid_parse:
        reasons.append("unusable_numeric_parse:" + parse_status)
    if protein_status != "exact_single_uniprot":
        reasons.append("protein_mapping:" + protein_status)
    if compound_status != "exact_unique_full_inchikey":
        reasons.append("compound_mapping:" + compound_status)
    if normalized_unit.startswith("NA_"):
        reasons.append("missing_or_unsupported_unit")
    if reasons:
        return "EXCLUDE", join_values(reasons)
    if parameter_origin == "comment_explicit_secondary":
        return "Q3", "explicit_comment_parameter_requires_secondary_text_validation"
    if not has_reference:
        return "Q3", "no_source_reference_id"
    if condition_completeness == "ph_and_temperature_reported":
        return "Q1", "exact_entities_numeric_unit_reference_ph_temperature"
    return "Q2", "exact_entities_numeric_unit_reference_incomplete_conditions"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brenda-tar", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out-root", type=Path, default=PROJECT)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-compound-map", action="store_true")
    args = parser.parse_args()

    started = time.time()
    retrieval_date = date.today().isoformat()
    out_root = args.out_root.resolve()
    data_dir = out_root / "data/k03_brenda"
    cache_dir = out_root / "cache/k03_brenda"
    report_dir = out_root / "reports"
    manifest_dir = out_root / "manifests"
    checksum_dir = out_root / "checksums"
    status_dir = out_root / "work/status"
    for directory in [data_dir, cache_dir, report_dir, manifest_dir, checksum_dir, status_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if not args.brenda_tar.exists():
        raise FileNotFoundError(args.brenda_tar)
    member_name, release = archive_member_and_release(args.brenda_tar)
    if release != EXPECTED_RELEASE:
        raise ValueError("Expected BRENDA {}, found {}".format(EXPECTED_RELEASE, release))

    compound_cache = cache_dir / "K03_BRENDA_COMPOUND_STRUCTURE_MAP.tsv.gz"
    compound_fetch_metadata = cache_dir / "K03_BRENDA_COMPOUND_MAP_FETCH.json"
    if args.refresh_compound_map and compound_cache.exists():
        compound_cache.unlink()
    if not compound_cache.exists() and not args.offline:
        fetch_compound_structure_map(compound_cache, compound_fetch_metadata, retrieval_date)
    if args.offline and not compound_fetch_metadata.exists():
        write_json(
            compound_fetch_metadata,
            {
                "status": "offline_cache_present" if compound_cache.exists() else "offline_cache_missing",
                "endpoint": SPARQL_ENDPOINT,
                "retrieval_date": retrieval_date,
            },
        )
    structure_map, structure_names, structure_mapping_rows = load_compound_structure_map(compound_cache)

    paths = {
        "ledger": data_dir / "K03_BRENDA_SOURCE_RECORD_LEDGER.tsv",
        "experiments": data_dir / "K03_BRENDA_EXPERIMENTS.tsv",
        "parameters": data_dir / "K03_BRENDA_PARAMETERS.tsv",
        "participants": data_dir / "K03_BRENDA_PARTICIPANTS.tsv",
        "conditions": data_dir / "K03_BRENDA_CONDITIONS.tsv",
        "references": data_dir / "K03_BRENDA_REFERENCES.tsv",
        "proteins": data_dir / "K03_BRENDA_PROTEIN_MAPPINGS.tsv",
        "compounds": data_dir / "K03_BRENDA_COMPOUND_MAPPINGS.tsv",
    }

    ledger_fields = [
        "source_database", "source_release", "source_record_id", "brenda_ec_number",
        "source_value_field", "source_field_row_index", "base_parameter_type", "raw_value",
        "raw_comment", "brenda_protein_ids", "brenda_reference_ids", "participant_names",
        "participant_count", "base_numeric_parse_status", "derived_parameter_count",
        "protein_mapping_summary", "compound_mapping_summary", "normalization_disposition",
        "reason_codes", "raw_source_locator", "retrieval_date", "parser_version",
    ]
    experiment_fields = [
        "source_database", "source_release", "source_record_id", "source_publication_ids",
        "experiment_id", "series_id", "assay_series_id", "condition_id", "brenda_ec_number",
        "recommended_enzyme_name", "brenda_protein_id", "protein_mapping_id",
        "uniprot_accessions", "protein_mapping_status", "organism", "construct_status",
        "mutation_tokens", "construct_context", "linked_variant_context",
        "variant_link_status", "biochemical_cellular_binding_class", "assay_type",
        "readout_direction", "raw_source_locator", "retrieval_date", "parser_version",
        "row_provenance_status",
    ]
    parameter_fields = [
        "source_database", "source_release", "source_record_id", "source_publication_ids",
        "parameter_id", "experiment_id", "assay_series_id", "brenda_ec_number",
        "protein_mapping_id", "uniprot_accessions", "protein_mapping_status",
        "participant_ids", "participant_names", "participant_role", "compound_mapping_status",
        "full_inchikeys", "connectivity_keys", "parameter_type", "parameter_origin",
        "source_value_field", "raw_value", "raw_numeric_text", "numeric_value",
        "numeric_value_min", "numeric_value_max", "numeric_uncertainty", "original_unit",
        "relation_operator", "normalized_value", "normalized_value_min",
        "normalized_value_max", "normalized_unit", "numeric_parse_status",
        "measured_fitted_reported_status", "kinetic_law", "rate_equation",
        "raw_condition_text", "condition_completeness", "quality_tier", "quality_reason_codes",
        "mechanism_label", "allostery_claim_limit", "raw_source_locator", "retrieval_date",
        "parser_version", "row_provenance_status",
    ]
    participant_fields = [
        "source_database", "source_release", "source_record_id", "participant_id",
        "compound_mapping_id", "participant_role", "original_name", "source_identifier",
        "canonical_smiles", "full_inchikeys", "connectivity_keys", "mapping_method",
        "mapping_status", "mapping_reason", "stereochemistry_status", "salt_mapping_status",
        "raw_source_locator", "retrieval_date", "parser_version",
    ]
    condition_fields = [
        "source_database", "source_release", "source_record_id", "experiment_id", "condition_id",
        "raw_condition_text", "ph_values", "temperature_c_values", "buffer_context",
        "ionic_strength_context", "enzyme_concentration_context", "substrate_concentration_context",
        "modifier_concentration_context", "cofactor_context", "detergent_context",
        "all_concentration_mentions", "incubation_time_mentions", "condition_completeness",
        "ph_missing", "temperature_missing", "raw_source_locator", "retrieval_date",
        "parser_version",
    ]
    reference_fields = [
        "source_publication_id", "source_database", "source_release", "brenda_ec_number",
        "brenda_reference_id", "pmid", "doi", "title", "authors", "journal",
        "publication_year", "volume", "pages", "publication_family_id", "mapping_status",
        "raw_source_locator", "retrieval_date",
    ]
    protein_fields = [
        "protein_mapping_id", "source_database", "source_release", "brenda_ec_number",
        "brenda_protein_id", "uniprot_accessions", "uniprot_isoform_status", "organism",
        "protein_comment", "protein_reference_ids", "mapping_method", "mapping_status",
        "mapping_confidence", "sequence_hash", "sequence_cluster_id",
    ]
    compound_fields = [
        "compound_mapping_id", "original_names", "normalized_name", "canonical_smiles",
        "full_inchikeys", "connectivity_keys", "full_inchikey_count", "connectivity_key_count",
        "mapping_method", "mapping_status", "mapping_reason", "stereochemistry_status",
        "salt_mapping_status", "bemis_murcko_scaffold", "close_analogue_group",
    ]

    handles = {}
    writers = {}
    for key, fields in [
        ("ledger", ledger_fields), ("experiments", experiment_fields),
        ("parameters", parameter_fields), ("participants", participant_fields),
        ("conditions", condition_fields),
    ]:
        handles[key] = paths[key].open("w", encoding="utf-8", newline="")
        writers[key] = csv.DictWriter(handles[key], fieldnames=fields, delimiter="\t", lineterminator="\n")
        writers[key].writeheader()

    counters = defaultdict(Counter)
    output_counts = Counter()
    unique_accessions = set()
    unique_inchikeys = set()
    unique_ecs = set()
    protein_rows = {}
    reference_rows = {}
    compound_rows = {}
    compound_original_names = defaultdict(set)
    duplicate_signatures = set()
    duplicate_parameter_rows = 0

    try:
        for ec, obj in brenda_json_stream(args.brenda_tar):
            if ec == "spontaneous":
                continue
            unique_ecs.add(ec)
            recommended_name = clean_text(obj.get("recommended_name", "")) or "NA_not_reported"
            variant_index = build_variant_index(obj)
            raw_root = "{}#data/{}/".format(args.brenda_tar.resolve(), ec)
            for source_field, (base_type, base_unit, role) in PARAMETER_FIELDS.items():
                for source_index, entry in enumerate(obj.get(source_field) or [], 1):
                    counters["source_field"][source_field] += 1
                    source_record_id = "BRENDA:{}:{}:{}:{:06d}".format(
                        EXPECTED_RELEASE, ec, source_field, source_index
                    )
                    raw_locator = raw_root + "{}/{}".format(source_field, source_index - 1)
                    raw_value = clean_text(entry.get("value", ""))
                    comment = clean_text(entry.get("comment", ""))
                    proteins = [clean_text(x) for x in entry.get("proteins") or [] if clean_text(x)]
                    references = [clean_text(x) for x in entry.get("references") or [] if clean_text(x)]
                    if not proteins:
                        proteins = ["NA_no_protein_id"]
                    participant_names = extract_participant_names(raw_value)
                    base_parsed = parse_reported_value(raw_value)
                    secondary = explicit_secondary_parameters(comment)
                    counters["base_parse"][base_parsed["numeric_parse_status"]] += 1

                    participant_rows = []
                    participant_mapping_statuses = []
                    participant_ids = []
                    participant_full_keys = set()
                    participant_connectivities = set()
                    for ordinal, participant_name in enumerate(participant_names or ["NA_no_participant_term"], 1):
                        mapping = compound_mapping_for_name(participant_name, structure_map)
                        normalized = mapping["normalized_name"]
                        compound_original_names[normalized].add(participant_name)
                        compound_rows[normalized] = mapping
                        participant_id = stable_id(
                            "K03PART", source_record_id, role, normalized, ordinal
                        )
                        participant_ids.append(participant_id)
                        participant_mapping_statuses.append(mapping["mapping_status"])
                        if not mapping["full_inchikeys"].startswith("NA_"):
                            participant_full_keys.update(mapping["full_inchikeys"].split(";"))
                        if not mapping["connectivity_keys"].startswith("NA_"):
                            participant_connectivities.update(mapping["connectivity_keys"].split(";"))
                        participant_row = {
                            "source_database": SOURCE_DATABASE,
                            "source_release": EXPECTED_RELEASE,
                            "source_record_id": source_record_id,
                            "participant_id": participant_id,
                            "compound_mapping_id": mapping["compound_mapping_id"],
                            "participant_role": role,
                            "original_name": participant_name,
                            "source_identifier": "NA_BRENDA_parameter_uses_name_only",
                            "canonical_smiles": mapping["canonical_smiles"],
                            "full_inchikeys": mapping["full_inchikeys"],
                            "connectivity_keys": mapping["connectivity_keys"],
                            "mapping_method": mapping["mapping_method"],
                            "mapping_status": mapping["mapping_status"],
                            "mapping_reason": mapping["mapping_reason"],
                            "stereochemistry_status": mapping["stereochemistry_status"],
                            "salt_mapping_status": mapping["salt_mapping_status"],
                            "raw_source_locator": raw_locator,
                            "retrieval_date": retrieval_date,
                            "parser_version": PARSER_VERSION,
                        }
                        writers["participants"].writerow(participant_row)
                        output_counts["participants"] += 1
                        counters["compound_mapping"][mapping["mapping_status"]] += 1

                    if len(participant_names) != 1:
                        compound_status = (
                            "no_participant_term" if not participant_names else "multiple_participant_terms_ambiguous"
                        )
                    elif len(participant_mapping_statuses) == 1:
                        compound_status = participant_mapping_statuses[0]
                    else:
                        compound_status = "participant_mapping_unresolved"

                    for reference_id in references:
                        pub_id = "BRENDA:{}:{}:REF:{}".format(EXPECTED_RELEASE, ec, reference_id)
                        if pub_id not in reference_rows:
                            reference_rows[pub_id] = reference_row(
                                ec, reference_id, obj, raw_root.rstrip("/"), retrieval_date
                            )
                    source_publication_ids = join_values(
                        "BRENDA:{}:{}:REF:{}".format(EXPECTED_RELEASE, ec, x)
                        for x in references
                    ) or "NA_not_reported"

                    protein_statuses_for_ledger = []
                    for protein_id in proteins:
                        protein_obj = (obj.get("protein") or {}).get(str(protein_id), {})
                        pmap = protein_mapping(ec, protein_id, protein_obj)
                        protein_rows[pmap["protein_mapping_id"]] = pmap
                        protein_status = pmap["mapping_status"]
                        protein_statuses_for_ledger.append(protein_status)
                        counters["protein_mapping"][protein_status] += 1
                        if protein_status == "exact_single_uniprot":
                            unique_accessions.update(pmap["uniprot_accessions"].split(";"))

                        linked_variants = []
                        for ref_id in references or ["NA_no_reference"]:
                            linked_variants.extend(variant_index.get((protein_id, ref_id), []))
                        construct_status, mutation_tokens, construct_text = construct_context(
                            comment,
                            "" if pmap["protein_comment"].startswith("NA_") else pmap["protein_comment"],
                            linked_variants,
                        )
                        experiment_id = stable_id("K03EXP", source_record_id, protein_id)
                        condition_id = stable_id("K03COND", experiment_id)
                        assay_series_id = stable_id(
                            "K03ASSAY", ec, protein_id, source_publication_ids
                        )
                        conditions = parse_conditions(comment)
                        experiment_row = {
                            "source_database": SOURCE_DATABASE,
                            "source_release": EXPECTED_RELEASE,
                            "source_record_id": source_record_id,
                            "source_publication_ids": source_publication_ids,
                            "experiment_id": experiment_id,
                            "series_id": "NA_reported_parameter_not_raw_series",
                            "assay_series_id": assay_series_id,
                            "condition_id": condition_id,
                            "brenda_ec_number": ec,
                            "recommended_enzyme_name": recommended_name,
                            "brenda_protein_id": protein_id,
                            "protein_mapping_id": pmap["protein_mapping_id"],
                            "uniprot_accessions": pmap["uniprot_accessions"],
                            "protein_mapping_status": protein_status,
                            "organism": pmap["organism"],
                            "construct_status": construct_status,
                            "mutation_tokens": mutation_tokens,
                            "construct_context": construct_text,
                            "linked_variant_context": join_values(linked_variants) or "NA_not_linked",
                            "variant_link_status": (
                                "same_protein_and_reference_context_linked"
                                if linked_variants
                                else "no_same_protein_reference_variant_context"
                            ),
                            "biochemical_cellular_binding_class": "biochemical_reported_parameter",
                            "assay_type": "enzyme_kinetic_parameter",
                            "readout_direction": "NA_parameter_not_response_direction",
                            "raw_source_locator": raw_locator,
                            "retrieval_date": retrieval_date,
                            "parser_version": PARSER_VERSION,
                            "row_provenance_status": "source_record_preserved",
                        }
                        writers["experiments"].writerow(experiment_row)
                        output_counts["experiments"] += 1
                        counters["construct_status"][construct_status] += 1
                        counters["condition_completeness"][conditions["condition_completeness"]] += 1

                        condition_row = {
                            "source_database": SOURCE_DATABASE,
                            "source_release": EXPECTED_RELEASE,
                            "source_record_id": source_record_id,
                            "experiment_id": experiment_id,
                            "condition_id": condition_id,
                            "raw_source_locator": raw_locator,
                            "retrieval_date": retrieval_date,
                            "parser_version": PARSER_VERSION,
                        }
                        condition_row.update(conditions)
                        writers["conditions"].writerow(condition_row)
                        output_counts["conditions"] += 1

                        base_norm = dict(base_parsed)
                        base_norm.update(
                            {
                                "parameter_type": base_type,
                                "parameter_origin": "BRENDA_dedicated_parameter_field",
                                "raw_value": raw_value,
                                "original_unit": base_unit,
                                "normalized_value": base_parsed["numeric_value"],
                                "normalized_value_min": base_parsed["numeric_value_min"],
                                "normalized_value_max": base_parsed["numeric_value_max"],
                                "normalized_unit": base_unit,
                            }
                        )
                        parameter_candidates = [base_norm] + secondary
                        for parameter_ordinal, candidate in enumerate(parameter_candidates, 1):
                            tier, reasons = quality_tier(
                                candidate["parameter_origin"],
                                candidate["numeric_parse_status"],
                                protein_status,
                                compound_status,
                                bool(references),
                                conditions["condition_completeness"],
                                candidate["normalized_unit"],
                            )
                            parameter_id = stable_id(
                                "K03PAR",
                                experiment_id,
                                candidate["parameter_type"],
                                candidate["parameter_origin"],
                                parameter_ordinal,
                            )
                            parameter_row = {
                                "source_database": SOURCE_DATABASE,
                                "source_release": EXPECTED_RELEASE,
                                "source_record_id": source_record_id,
                                "source_publication_ids": source_publication_ids,
                                "parameter_id": parameter_id,
                                "experiment_id": experiment_id,
                                "assay_series_id": assay_series_id,
                                "brenda_ec_number": ec,
                                "protein_mapping_id": pmap["protein_mapping_id"],
                                "uniprot_accessions": pmap["uniprot_accessions"],
                                "protein_mapping_status": protein_status,
                                "participant_ids": join_values(participant_ids),
                                "participant_names": join_values(participant_names) or "NA_not_reported",
                                "participant_role": role,
                                "compound_mapping_status": compound_status,
                                "full_inchikeys": join_values(participant_full_keys) or "NA_not_mapped",
                                "connectivity_keys": join_values(participant_connectivities) or "NA_not_mapped",
                                "parameter_type": candidate["parameter_type"],
                                "parameter_origin": candidate["parameter_origin"],
                                "source_value_field": source_field,
                                "raw_value": candidate["raw_value"],
                                "raw_numeric_text": candidate["raw_numeric_text"],
                                "numeric_value": candidate["numeric_value"],
                                "numeric_value_min": candidate["numeric_value_min"],
                                "numeric_value_max": candidate["numeric_value_max"],
                                "numeric_uncertainty": candidate["numeric_uncertainty"],
                                "original_unit": candidate["original_unit"],
                                "relation_operator": candidate["relation_operator"],
                                "normalized_value": candidate["normalized_value"],
                                "normalized_value_min": candidate["normalized_value_min"],
                                "normalized_value_max": candidate["normalized_value_max"],
                                "normalized_unit": candidate["normalized_unit"],
                                "numeric_parse_status": candidate["numeric_parse_status"],
                                "measured_fitted_reported_status": "database_reported_unspecified",
                                "kinetic_law": "NA_not_reported_in_dedicated_BRENDA_parameter_row",
                                "rate_equation": "NA_not_reported_in_dedicated_BRENDA_parameter_row",
                                "raw_condition_text": conditions["raw_condition_text"],
                                "condition_completeness": conditions["condition_completeness"],
                                "quality_tier": tier,
                                "quality_reason_codes": reasons,
                                "mechanism_label": "unresolved",
                                "allostery_claim_limit": (
                                    "Reported kinetic parameter only; it is not proof of a remote binding site or allostery."
                                ),
                                "raw_source_locator": raw_locator,
                                "retrieval_date": retrieval_date,
                                "parser_version": PARSER_VERSION,
                                "row_provenance_status": "normalized_without_source_row_collapse",
                            }
                            writers["parameters"].writerow(parameter_row)
                            output_counts["parameters"] += 1
                            counters["parameter_type"][candidate["parameter_type"]] += 1
                            counters["parameter_quality"][tier] += 1
                            counters["parameter_parse"][candidate["numeric_parse_status"]] += 1
                            counters["parameter_type_quality"][(candidate["parameter_type"], tier)] += 1
                            unique_inchikeys.update(participant_full_keys)
                            signature = (
                                pmap["uniprot_accessions"], candidate["parameter_type"],
                                candidate["normalized_value"], candidate["normalized_value_min"],
                                candidate["normalized_value_max"], candidate["normalized_unit"],
                                join_values(participant_full_keys), source_publication_ids,
                                conditions["raw_condition_text"],
                            )
                            if signature in duplicate_signatures:
                                duplicate_parameter_rows += 1
                            else:
                                duplicate_signatures.add(signature)

                    reason_codes = []
                    if base_parsed["numeric_parse_status"] not in {
                        "parsed_single_value", "parsed_value_with_uncertainty", "parsed_range"
                    }:
                        reason_codes.append("base_numeric:" + base_parsed["numeric_parse_status"])
                    if compound_status != "exact_unique_full_inchikey":
                        reason_codes.append("compound:" + compound_status)
                    if not all(x == "exact_single_uniprot" for x in protein_statuses_for_ledger):
                        reason_codes.append("one_or_more_nonexact_protein_mappings")
                    ledger_row = {
                        "source_database": SOURCE_DATABASE,
                        "source_release": EXPECTED_RELEASE,
                        "source_record_id": source_record_id,
                        "brenda_ec_number": ec,
                        "source_value_field": source_field,
                        "source_field_row_index": source_index,
                        "base_parameter_type": base_type,
                        "raw_value": raw_value or "NA_not_reported",
                        "raw_comment": comment or "NA_not_reported",
                        "brenda_protein_ids": join_values(proteins),
                        "brenda_reference_ids": join_values(references) or "NA_not_reported",
                        "participant_names": join_values(participant_names) or "NA_not_reported",
                        "participant_count": len(participant_names),
                        "base_numeric_parse_status": base_parsed["numeric_parse_status"],
                        "derived_parameter_count": len(secondary),
                        "protein_mapping_summary": join_values(protein_statuses_for_ledger),
                        "compound_mapping_summary": compound_status,
                        "normalization_disposition": "retained_with_quality_tier",
                        "reason_codes": join_values(reason_codes) or "none",
                        "raw_source_locator": raw_locator,
                        "retrieval_date": retrieval_date,
                        "parser_version": PARSER_VERSION,
                    }
                    writers["ledger"].writerow(ledger_row)
                    output_counts["ledger"] += 1
    finally:
        for handle in handles.values():
            handle.close()

    protein_rows_sorted = sorted(
        protein_rows.values(),
        key=lambda r: (r["brenda_ec_number"], r["brenda_protein_id"]),
    )
    write_tsv(paths["proteins"], protein_rows_sorted, protein_fields)
    output_counts["proteins"] = len(protein_rows_sorted)

    reference_rows_sorted = sorted(
        reference_rows.values(),
        key=lambda r: (r["brenda_ec_number"], r["brenda_reference_id"]),
    )
    write_tsv(paths["references"], reference_rows_sorted, reference_fields)
    output_counts["references"] = len(reference_rows_sorted)

    compound_rows_sorted = []
    for normalized, row in compound_rows.items():
        compound_rows_sorted.append(
            {
                "compound_mapping_id": row["compound_mapping_id"],
                "original_names": join_values(compound_original_names[normalized]),
                "normalized_name": row["normalized_name"],
                "canonical_smiles": row["canonical_smiles"],
                "full_inchikeys": row["full_inchikeys"],
                "connectivity_keys": row["connectivity_keys"],
                "full_inchikey_count": row["full_inchikey_count"],
                "connectivity_key_count": row["connectivity_key_count"],
                "mapping_method": row["mapping_method"],
                "mapping_status": row["mapping_status"],
                "mapping_reason": row["mapping_reason"],
                "stereochemistry_status": row["stereochemistry_status"],
                "salt_mapping_status": row["salt_mapping_status"],
                "bemis_murcko_scaffold": row["bemis_murcko_scaffold"],
                "close_analogue_group": row["close_analogue_group"],
            }
        )
    compound_rows_sorted.sort(key=lambda r: r["normalized_name"])
    write_tsv(paths["compounds"], compound_rows_sorted, compound_fields)
    output_counts["compounds"] = len(compound_rows_sorted)

    expected_source_records = sum(counters["source_field"].values())
    invariants = {
        "release_is_2026_1": release == EXPECTED_RELEASE,
        "ledger_equals_archive_parameter_records": output_counts["ledger"] == expected_source_records,
        "experiments_equal_conditions": output_counts["experiments"] == output_counts["conditions"],
        "parameter_rows_cover_all_experiments": output_counts["parameters"] >= output_counts["experiments"],
        "source_records_nonzero": output_counts["ledger"] > 0,
        "parameter_rows_nonzero": output_counts["parameters"] > 0,
        "all_five_requested_parameter_types_present": all(
            counters["parameter_type"][x] > 0 for x in ["Km", "kcat", "Ki", "S0.5", "nH"]
        ),
    }
    if not all(invariants.values()):
        raise AssertionError("K03 invariant failure: {}".format(invariants))

    report_path = report_dir / "03_BRENDA_GENERAL_PARAMETER_EXTRACTION.md"
    runtime = time.time() - started
    report_lines = [
        "# K03 BRENDA general kinetic-parameter extraction",
        "",
        "## Outcome",
        "",
        "The complete local BRENDA 2026.1 JSON archive was streamed without restricting it to the prior 25,333-pair universe.",
        "All dedicated Km, turnover-number/kcat, kcat/Km and Ki source records were retained. Explicit S0.5 and Hill-coefficient values were conservatively extracted from parameter-row comments because BRENDA has no dedicated JSON fields for those two types.",
        "",
        "## Source-record accounting",
        "",
        "| Source field | Source records |",
        "|---|---:|",
    ]
    for field in PARAMETER_FIELDS:
        report_lines.append("| {} | {:,} |".format(field, counters["source_field"][field]))
    report_lines.extend(
        [
            "| Total | {:,} |".format(expected_source_records),
            "",
            "Normalized experiments: **{:,}**; parameter rows including explicit comment-derived values: **{:,}**; participants: **{:,}**.".format(
                output_counts["experiments"], output_counts["parameters"], output_counts["participants"]
            ),
            "",
            "## Parameter and quality counts",
            "",
            "| Parameter | Q1 | Q2 | Q3 | EXCLUDE | Total |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for parameter_type in ["Km", "kcat", "kcat_over_Km", "Ki", "S0.5", "nH"]:
        values = [
            counters["parameter_type_quality"][(parameter_type, tier)]
            for tier in ["Q1", "Q2", "Q3", "EXCLUDE"]
        ]
        report_lines.append(
            "| {} | {:,} | {:,} | {:,} | {:,} | {:,} |".format(
                parameter_type, values[0], values[1], values[2], values[3], sum(values)
            )
        )
    report_lines.extend(
        [
            "",
            "Q1 requires an exact single UniProt accession, a unique full InChIKey, a usable numeric value and unit, at least one reference, and both pH and temperature in the row comment. Q2 permits incomplete conditions. Comment-derived S0.5/nH is capped at Q3 pending text validation. Ambiguous or missing protein/compound mappings and malformed values remain present as EXCLUDE rows with reasons.",
            "",
            "## Mapping and context coverage",
            "",
            "- Parameter-linked protein mapping objects: **{:,}**; unique exact UniProt accessions: **{:,}**.".format(
                output_counts["proteins"], len(unique_accessions)
            ),
            "- Unique parameter compound names: **{:,}**; unique mapped full InChIKeys observed in parameter rows: **{:,}**.".format(
                output_counts["compounds"], len(unique_inchikeys)
            ),
            "- Reference objects retained: **{:,}**; EC entries represented: **{:,}**.".format(
                output_counts["references"], len(unique_ecs)
            ),
            "- Exact duplicate normalized parameter signatures were not deleted: **{:,}** later rows share a protein/participant/value/publication/condition signature with an earlier row.".format(
                duplicate_parameter_rows
            ),
            "",
            "### Protein mapping disposition (experiment rows)",
            "",
            "| Mapping status | Rows |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(counters["protein_mapping"].items()):
        report_lines.append("| {} | {:,} |".format(key, value))
    report_lines.extend(
        [
            "",
            "### Compound mapping disposition (participant rows)",
            "",
            "| Mapping status | Rows |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(counters["compound_mapping"].items()):
        report_lines.append("| {} | {:,} |".format(key, value))
    report_lines.extend(
        [
            "",
            "### Base-value parsing and condition coverage",
            "",
            "| Base numeric parse status | Source records |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(counters["base_parse"].items()):
        report_lines.append("| {} | {:,} |".format(key, value))
    report_lines.extend(
        [
            "",
            "| Condition completeness | Experiment rows |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(counters["condition_completeness"].items()):
        report_lines.append("| {} | {:,} |".format(key, value))
    report_lines.extend(
        [
            "",
            "## Units and interpretation",
            "",
            "BRENDA documents Km and Ki in mM and turnover number in 1/s. The database-displayed kcat/Km unit is preserved verbatim as mM/s rather than silently dimension-corrected. Source definitions: [BRENDA datafields]({}).".format(BRENDA_DATAFIELDS_PAGE),
            "",
            "Km was not renamed Kd. Ki was not converted into a mechanism or allostery label. General pH/temperature-optimum fields were not copied into experiment rows; only the parameter-row comment was used for condition context.",
            "",
            "## Important limits",
            "",
            "- These are reported parameters, not raw concentration-response points.",
            "- S0.5/nH has no dedicated archive field and the explicit-comment extraction is intentionally Q3/EXCLUDE.",
            "- Compound structures use exact normalized-name matches to the official BRENDA SPARQL structure map. Multiple structures or absent mappings are retained but cannot enter Q1/Q2.",
            "- Mutation/variant context is preserved from the parameter comment, protein comment, and only protein-variant records sharing the same BRENDA protein and reference. It is not treated as a fully normalized construct sequence.",
            "- Missing conditions are NA, not negative evidence.",
            "",
            "## Reproducibility",
            "",
            "- Archive: {} (member {}, release {}).".format(args.brenda_tar.resolve(), member_name, release),
            "- Official source page: [BRENDA download]({}).".format(BRENDA_DOWNLOAD_PAGE),
            "- Parser: {} under Python {}.".format(PARSER_VERSION, platform.python_version()),
            "- Runtime: {:.1f} seconds.".format(runtime),
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    status_path = status_dir / "K03__BRENDA_PARAMETERS.json"
    status = {
        "work_package": "K03",
        "task": "BRENDA general kinetic-parameter extraction",
        "status": "complete",
        "source_database": SOURCE_DATABASE,
        "source_release": release,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "parser_version": PARSER_VERSION,
        "input_archive": str(args.brenda_tar.resolve()),
        "counts": dict(output_counts),
        "source_field_counts": dict(counters["source_field"]),
        "parameter_type_counts": dict(counters["parameter_type"]),
        "quality_tier_counts": dict(counters["parameter_quality"]),
        "protein_mapping_status_counts": dict(counters["protein_mapping"]),
        "compound_mapping_status_counts": dict(counters["compound_mapping"]),
        "base_numeric_parse_status_counts": dict(counters["base_parse"]),
        "condition_completeness_counts": dict(counters["condition_completeness"]),
        "unique_exact_uniprot_accessions": len(unique_accessions),
        "unique_mapped_full_inchikeys": len(unique_inchikeys),
        "duplicate_parameter_signatures_retained": duplicate_parameter_rows,
        "compound_structure_mapping_rows": structure_mapping_rows,
        "compound_structure_mapping_available": bool(structure_map),
        "validation": {
            "internal_invariants": invariants,
            "all_internal_invariants_passed": all(invariants.values()),
            "independent_validator": "provided_and_run_as_K03_completion_step",
        },
        "limitations": [
            "No raw concentration-response points are present in this source.",
            "S0.5 and nH are conservative explicit-comment extractions and are capped at Q3.",
            "Unique name-to-structure mapping is not equivalent to experimental stereochemical verification.",
            "Ki alone does not identify inhibition mechanism or allostery.",
        ],
    }
    write_json(status_path, status)

    script_path = Path(__file__).resolve()
    manifest_path = manifest_dir / "K03_MANIFEST.tsv"
    manifest_fields = ["role", "path", "bytes", "sha256", "row_count", "note"]
    manifest_rows = []
    manifest_items = [
        ("input", args.brenda_tar.resolve(), "immutable local BRENDA 2026.1 archive"),
        ("script", script_path, "reproducible K03 extractor"),
        ("cache", compound_cache, "official BRENDA compound-name/InChIKey mapping cache"),
        ("cache_metadata", compound_fetch_metadata, "mapping retrieval status and query provenance"),
    ]
    manifest_items.extend(("output", path, key) for key, path in paths.items())
    manifest_items.extend(
        [
            ("report", report_path, "coverage, duplication and limitation report"),
            ("status", status_path, "K03 completion status"),
        ]
    )
    validator_path = script_path.parent / "validate_k03_brenda_parameters.py"
    if validator_path.exists():
        manifest_items.append(("validator", validator_path, "independent K03 output validator"))
    for role, path, note in manifest_items:
        path = Path(path)
        if not path.exists():
            continue
        try:
            relative = path.resolve().relative_to(ROOT)
            shown_path = str(relative)
        except ValueError:
            shown_path = str(path.resolve())
        if path.suffix in {".tsv"} or str(path).endswith(".tsv.gz"):
            rows = row_count(path)
        else:
            rows = "NA"
        manifest_rows.append(
            {
                "role": role,
                "path": shown_path,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "row_count": rows,
                "note": note,
            }
        )
    write_tsv(manifest_path, manifest_rows, manifest_fields)

    checksum_path = checksum_dir / "K03_CHECKSUMS.sha256"
    checksum_targets = [Path(row["path"]) if str(row["path"]).startswith("/") else ROOT / row["path"] for row in manifest_rows]
    checksum_targets.append(manifest_path)
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in checksum_targets:
            try:
                shown_path = path.resolve().relative_to(ROOT)
            except ValueError:
                shown_path = path.resolve()
            handle.write("{}  {}\n".format(sha256_file(path), shown_path))

    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
