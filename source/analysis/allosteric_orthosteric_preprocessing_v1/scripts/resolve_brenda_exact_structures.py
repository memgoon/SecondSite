#!/usr/bin/env python3
"""Resolve canonical SMILES for exact BRENDA InChIKeys through Tuna ES.

Only exact full-InChIKey and exact CID queries are used.  There is no fuzzy
name search and no label is derived from Elasticsearch.  The retrieved SMILES
must reproduce the queried full InChIKey with the pinned RDKit runtime before
being accepted.  Cached local SMILES are subjected to the same check.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import pickle
import re
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem, rdBase


REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = REPO / "analysis/kinetic_allostery_model_v1/data/k03_brenda/K03_BRENDA_PARAMETERS.tsv"
LOCAL_SMILES = Path("/shared_data/11.HS_allostery/Data/smiles_cache.pickle")
ENDPOINT = "http://tuna.snu.ac.kr:9200"
IK_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
NA = "NA"


def eligible_inchikeys() -> set[str]:
    out: set[str] = set()
    with PARAMETERS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            ik = row["full_inchikeys"].strip().upper()
            if (
                row["protein_mapping_status"] == "exact_single_uniprot"
                and row["participant_role"] == "substrate"
                and row["compound_mapping_status"] == "exact_unique_full_inchikey"
                and IK_RE.fullmatch(ik)
            ):
                out.add(ik)
    return out


def canonical_and_key(smiles: str) -> tuple[str, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", ""
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return canonical, Chem.MolToInchiKey(mol)


def post_json(url: str, payload: dict, timeout: int = 60) -> tuple[int, bytes]:
    data = json.dumps(payload, separators=(",", ":")).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def get_bytes(url: str, timeout: int = 15) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def write_gzip_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def query_terms(index: str, field: str, requested: list[str], source_fields: list[str], events: list[dict], batch_size: int) -> list[dict]:
    hits: list[dict] = []
    for ordinal, batch in enumerate(chunks(requested, batch_size)):
        payload = {
            "size": max(1000, len(batch) * 4),
            "track_total_hits": True,
            "_source": source_fields,
            "query": {"terms": {field: batch}},
        }
        status, body = post_json(f"{ENDPOINT}/{index}/_search", payload)
        event = {
            "request_id": f"{index}_{ordinal:04d}",
            "request_type": "exact_terms",
            "index": index,
            "field": field,
            "requested_values": batch,
            "http_status": status,
            "response_sha256": hashlib.sha256(body).hexdigest(),
            "response_body": body.decode("utf-8", errors="replace"),
            "live_network_request": True,
        }
        events.append(event)
        if status != 200:
            raise RuntimeError(f"Tuna {index} query failed with HTTP {status}")
        parsed = json.loads(body)
        if parsed.get("timed_out") or parsed.get("_shards", {}).get("failed", 0):
            raise RuntimeError(f"Tuna {index} query incomplete: {parsed.get('_shards')}")
        batch_hits = parsed.get("hits", {}).get("hits", [])
        total = parsed.get("hits", {}).get("total", {}).get("value", len(batch_hits))
        if total > len(batch_hits):
            raise RuntimeError(f"Tuna {index} response truncated: total={total}, returned={len(batch_hits)}")
        hits.extend(h.get("_source", {}) for h in batch_hits)
    return hits


def run(batch_size: int) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    keys = sorted(eligible_inchikeys())
    local = pickle.load(LOCAL_SMILES.open("rb"))
    resolved: dict[str, dict] = {}
    missing: list[str] = []
    for ik in keys:
        smiles = local.get(ik)
        if isinstance(smiles, str) and smiles.strip():
            canonical, got = canonical_and_key(smiles)
            if canonical and got == ik:
                resolved[ik] = {
                    "resolution_source": "existing_local_smiles_cache_exact_full_inchikey",
                    "canonical_smiles": canonical,
                    "pubchem_cids": NA,
                    "candidate_smiles_count": 1,
                    "rdkit_recomputed_inchikey": got,
                    "resolution_status": "resolved_exact_full_inchikey",
                }
                continue
        missing.append(ik)

    events: list[dict] = []
    health_status, health_body = get_bytes(ENDPOINT + "/")
    events.append(
        {
            "request_id": "server_health_0000",
            "request_type": "server_health",
            "index": NA,
            "field": NA,
            "requested_values": [],
            "http_status": health_status,
            "response_sha256": hashlib.sha256(health_body).hexdigest(),
            "response_body": health_body.decode("utf-8", errors="replace"),
            "live_network_request": True,
        }
    )
    if health_status != 200:
        raise RuntimeError(f"Tuna health request failed: HTTP {health_status}")

    ik_hits = query_terms("idx_inchikey", "inchi_key", missing, ["cid", "inchi_key"], events, batch_size)
    ik_to_cids: dict[str, set[str]] = defaultdict(set)
    for hit in ik_hits:
        ik = str(hit.get("inchi_key", "")).strip().upper()
        cid = str(hit.get("cid", "")).strip()
        if ik in set(missing) and cid:
            ik_to_cids[ik].add(cid)
    cids = sorted({cid for values in ik_to_cids.values() for cid in values}, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
    smiles_hits = query_terms("idx_smiles", "cid", cids, ["cid", "smiles"], events, batch_size)
    cid_to_smiles: dict[str, set[str]] = defaultdict(set)
    for hit in smiles_hits:
        cid = str(hit.get("cid", "")).strip()
        smiles = str(hit.get("smiles", "")).strip()
        if cid and smiles:
            cid_to_smiles[cid].add(smiles)

    for ik in missing:
        valid: set[str] = set()
        all_candidates: set[str] = set()
        for cid in ik_to_cids.get(ik, set()):
            for smiles in cid_to_smiles.get(cid, set()):
                canonical, got = canonical_and_key(smiles)
                if canonical:
                    all_candidates.add(canonical)
                if canonical and got == ik:
                    valid.add(canonical)
        if valid:
            selected = sorted(valid, key=lambda x: (len(x), x))[0]
            _, got = canonical_and_key(selected)
            resolved[ik] = {
                "resolution_source": "Tuna_ES_exact_InChIKey_to_CID_to_SMILES",
                "canonical_smiles": selected,
                "pubchem_cids": ";".join(sorted(ik_to_cids[ik], key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))),
                "candidate_smiles_count": len(all_candidates),
                "rdkit_recomputed_inchikey": got,
                "resolution_status": "resolved_exact_full_inchikey",
            }
        else:
            resolved[ik] = {
                "resolution_source": "Tuna_ES_exact_lookup",
                "canonical_smiles": NA,
                "pubchem_cids": ";".join(sorted(ik_to_cids.get(ik, set()))) or NA,
                "candidate_smiles_count": len(all_candidates),
                "rdkit_recomputed_inchikey": NA,
                "resolution_status": "unresolved_no_exact_matching_smiles",
            }

    rows = []
    for ik in keys:
        r = resolved[ik]
        rows.append(
            {
                "full_inchikey": ik,
                "connectivity_key": ik[:14],
                "canonical_smiles": r["canonical_smiles"],
                "pubchem_cids": r["pubchem_cids"],
                "resolution_source": r["resolution_source"],
                "resolution_status": r["resolution_status"],
                "candidate_smiles_count": r["candidate_smiles_count"],
                "rdkit_recomputed_inchikey": r["rdkit_recomputed_inchikey"],
                "exact_full_inchikey_validated": str(r["resolution_status"] == "resolved_exact_full_inchikey").lower(),
            }
        )

    write_tsv(
        ROOT / "data/BRENDA_EXACT_STRUCTURE_SUPPLEMENT.tsv",
        rows,
        [
            "full_inchikey", "connectivity_key", "canonical_smiles", "pubchem_cids",
            "resolution_source", "resolution_status", "candidate_smiles_count",
            "rdkit_recomputed_inchikey", "exact_full_inchikey_validated",
        ],
    )
    write_gzip_jsonl(ROOT / "cache/TUNA_EXACT_INCHIKEY_QUERY_EVENTS.jsonl.gz", events)
    summary = {
        "resolver_version": "brenda_exact_structure_resolver_v1.0",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": ENDPOINT,
        "query_policy": "exact full InChIKey -> CID -> SMILES; RDKit full-InChIKey reproduction required; no fuzzy or name query",
        "rdkit_version": rdBase.rdkitVersion,
        "eligible_unique_inchikeys": len(keys),
        "existing_cache_resolved": sum(r["resolution_source"].startswith("existing") for r in resolved.values()),
        "tuna_resolved": sum(r["resolution_source"].startswith("Tuna_ES_exact_InChIKey") for r in resolved.values()),
        "unresolved": sum(r["resolution_status"] != "resolved_exact_full_inchikey" for r in resolved.values()),
        "live_network_requests": len(events),
        "tuna_ik_hits": len(ik_hits),
        "unique_tuna_cids": len(cids),
        "tuna_smiles_hits": len(smiles_hits),
    }
    (ROOT / "data/BRENDA_EXACT_STRUCTURE_RESOLUTION_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def verify_existing() -> dict:
    summary = json.loads((ROOT / "data/BRENDA_EXACT_STRUCTURE_RESOLUTION_SUMMARY.json").read_text())
    with (ROOT / "data/BRENDA_EXACT_STRUCTURE_SUPPLEMENT.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    keys = eligible_inchikeys()
    checks = {
        "one_row_per_eligible_inchikey": len(rows) == len(keys) == len({r["full_inchikey"] for r in rows}),
        "exact_key_set": {r["full_inchikey"] for r in rows} == keys,
        "resolved_smiles_reproduce_full_inchikey": True,
        "no_fuzzy_resolution_source": all("fuzzy" not in r["resolution_source"].lower() for r in rows),
    }
    for row in rows:
        if row["resolution_status"] == "resolved_exact_full_inchikey":
            canonical, got = canonical_and_key(row["canonical_smiles"])
            if not canonical or got != row["full_inchikey"] or row["exact_full_inchikey_validated"] != "true":
                checks["resolved_smiles_reproduce_full_inchikey"] = False
                break
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"status": "PASS", "checks": checks, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="Execute exact Tuna queries for missing structures")
    group.add_argument("--verify-existing", action="store_true", help="Read-only verification; performs no network request")
    parser.add_argument("--batch-size", type=int, default=400)
    args = parser.parse_args()
    result = run(args.batch_size) if args.run else verify_existing()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
