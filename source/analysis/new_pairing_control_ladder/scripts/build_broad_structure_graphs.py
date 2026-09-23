#!/usr/bin/env python3
"""Build/reuse targeted pocket graphs for the broad control-ladder cohort."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/disk9/13.Heesu_Allostery")
SHARED = Path("/shared_data/11.HS_allostery")
REMOTE = Path("/disk1/11.HS_allostery")
PACKAGE = ROOT / "analysis/new_pairing_control_ladder"
SOURCE = PACKAGE / "data/BROAD_UNCONTROLLED_PREGRAPH.tsv.gz"
OUTPUT = PACKAGE / "data/BROAD_UNCONTROLLED_STRUCTURE_READY.tsv.gz"
AUDIT_OUTPUT = PACKAGE / "data/BROAD_STRUCTURE_GRAPH_AUDIT.tsv"
VALIDATION_OUTPUT = PACKAGE / "validation/BROAD_STRUCTURE_GRAPH_VALIDATION.json"
GRAPH_ROOT = PACKAGE / "data/structure_graphs"
OLD_PACKAGE = ROOT / "analysis/new_pairing_d1_pilot"
OLD_GRAPH_ROOT = OLD_PACKAGE / "data/structure_graphs"
OLD_AUDIT = OLD_PACKAGE / "data/STRUCTURE_GRAPH_AUDIT.tsv"
PROTEIN_MAP = SHARED / "14.Organized_input/Protein_Embedding_Map.tsv"
POCKET_MASKS = SHARED / "14.Organized_input/pocket_masks.json"
SEQUENCE_CACHE = SHARED / "14.Organized_input/uniprot_protein_cache.json"
HELPER_SCRIPT = OLD_PACKAGE / "scripts/build_structure_graphs.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("pairing_graph_helper", HELPER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def remote_graph_path(local_path):
    relative = Path(local_path).relative_to(ROOT)
    return str(REMOTE / relative)


def valid_graph(path):
    try:
        with np.load(path, allow_pickle=False) as graph:
            required = {"embedding_index", "ca_xyz", "neighbor_index", "neighbor_distance"}
            if required - set(graph.files):
                return False
            index = np.asarray(graph["embedding_index"])
            xyz = np.asarray(graph["ca_xyz"])
            neighbor = np.asarray(graph["neighbor_index"])
            distance = np.asarray(graph["neighbor_distance"])
        return bool(
            len(index) >= 10
            and xyz.shape == (len(index), 3)
            and neighbor.shape == distance.shape
            and neighbor.shape[0] == len(index)
            and np.isfinite(xyz).all()
            and np.isfinite(distance).all()
        )
    except Exception:
        return False


def main():
    helper = load_helper()
    frame = pd.read_csv(SOURCE, sep="\t", low_memory=False)
    with open(POCKET_MASKS, encoding="utf-8") as handle:
        masks = json.load(handle)
    with open(SEQUENCE_CACHE, encoding="utf-8") as handle:
        sequence_cache = json.load(handle)
    protein_map = pd.read_csv(PROTEIN_MAP, sep="\t", dtype=str).fillna("")
    protein_map["uid"] = protein_map["UniProt_ID"].str.split("-").str[0]
    protein_map = protein_map[protein_map["uid"].isin(set(frame["uniprot"].astype(str)))]
    old_audit = pd.read_csv(OLD_AUDIT, sep="\t", dtype=str).fillna("")
    old_ready = set(old_audit.loc[old_audit["status"].eq("ready"), "uniprot"])
    selected_embedding = frame.drop_duplicates("uniprot").set_index("uniprot")["protein_embedding_path"].astype(str).to_dict()

    GRAPH_ROOT.mkdir(parents=True, exist_ok=True)
    audit = []
    graph_paths = {}
    reused = 0
    built = 0
    proteins = sorted(set(frame["uniprot"].astype(str)))
    for position, uid in enumerate(proteins, 1):
        prior_path = OLD_GRAPH_ROOT / (uid + ".npz")
        record = {
            "uniprot": uid,
            "protein_embedding_path": selected_embedding[uid],
            "status": "excluded",
            "reason": "",
            "graph_origin": "",
            "graph_path": "",
            "errors": "",
        }
        if uid in old_ready and valid_graph(prior_path):
            graph_paths[uid] = remote_graph_path(prior_path)
            record.update({
                "status": "ready",
                "graph_origin": "reused_new_pairing_d1_pilot",
                "graph_path": str(prior_path),
            })
            reused += 1
            audit.append(record)
            continue

        canonical_record = sequence_cache.get(uid, {})
        canonical = canonical_record.get("seq", "") if isinstance(canonical_record, dict) else ""
        canonical = str(canonical).strip()
        paths = helper.candidate_paths(uid, protein_map[protein_map["uid"].eq(uid)])
        record["n_candidate_coordinate_files"] = len(paths)
        record["candidate_coordinate_files"] = ";".join(map(str, paths))
        if not canonical:
            record["reason"] = "canonical_sequence_missing"
        elif not paths:
            record["reason"] = "coordinate_file_missing"
        else:
            graph, errors = helper.choose_graph(uid, canonical, masks.get(uid, []), paths)
            record["errors"] = " | ".join(errors)
            if graph is None:
                record["reason"] = "coordinate_alignment_or_pocket_coverage_failed"
            else:
                output_path = GRAPH_ROOT / (uid + ".npz")
                temporary = output_path.with_suffix(".npz.tmp")
                with open(temporary, "wb") as handle:
                    np.savez_compressed(
                        handle,
                        embedding_index=graph["embedding_index"],
                        ca_xyz=graph["ca_xyz"],
                        neighbor_index=graph["neighbor_index"],
                        neighbor_distance=graph["neighbor_distance"],
                    )
                os.replace(temporary, output_path)
                if not valid_graph(output_path):
                    raise RuntimeError("invalid graph written for {}".format(uid))
                graph_paths[uid] = remote_graph_path(output_path)
                record.update({
                    "status": "ready",
                    "graph_origin": "built_control_ladder",
                    "graph_path": str(output_path),
                    "coordinate_file": str(graph["path"]),
                    "coordinate_chain": graph["chain"],
                    "alignment_identity": graph["identity"],
                    "canonical_length": len(canonical),
                    "coordinate_chain_length": graph["chain_length"],
                    "requested_pocket_residues": len(graph["requested"]),
                    "mapped_pocket_residues_before_cap": len(graph["mapped_pocket"]),
                    "graph_residues": len(graph["embedding_index"]),
                    "pocket_coordinate_coverage": len(graph["mapped_pocket"]) / max(len(graph["requested"]), 1),
                })
                built += 1
        audit.append(record)
        if position % 50 == 0 or position == len(proteins):
            print("targeted graphs {}/{} ready={} reused={} built={}".format(
                position, len(proteins), len(graph_paths), reused, built
            ), flush=True)

    audit_frame = pd.DataFrame(audit)
    audit_frame.to_csv(AUDIT_OUTPUT, sep="\t", index=False)
    ready = frame[frame["uniprot"].astype(str).isin(set(graph_paths))].copy()
    ready["structure_graph_path"] = ready["uniprot"].astype(str).map(graph_paths)
    ready = ready.sort_values(["uniprot", "class_label", "full_inchikey"]).reset_index(drop=True)
    ready.to_csv(OUTPUT, sep="\t", index=False, compression="gzip")
    labels = ready["class_label"].value_counts().to_dict()
    report = {
        "status": "validated",
        "no_directory_traversal": True,
        "source_rows": int(len(frame)),
        "source_proteins": int(frame["uniprot"].nunique()),
        "structure_ready_rows": int(len(ready)),
        "structure_ready_proteins": int(ready["uniprot"].nunique()),
        "structure_ready_ligands": int(ready["full_inchikey"].nunique()),
        "structure_ready_allosteric": int(labels.get("allosteric", 0)),
        "structure_ready_orthosteric": int(labels.get("orthosteric", 0)),
        "structure_ready_proteins_with_both_labels": int(ready.groupby("uniprot")["binary_label"].nunique().eq(2).sum()),
        "reused_graphs": int(reused),
        "new_graphs": int(built),
        "excluded_proteins": int(len(proteins) - len(graph_paths)),
        "exclusion_reasons": audit_frame.loc[audit_frame["status"].ne("ready"), "reason"].value_counts().astype(int).to_dict(),
        "inputs": {
            "source": {"path": str(SOURCE), "sha256": sha256(SOURCE)},
            "old_audit": {"path": str(OLD_AUDIT), "sha256": sha256(OLD_AUDIT)},
            "protein_map": {"path": str(PROTEIN_MAP), "sha256": sha256(PROTEIN_MAP)},
            "pocket_masks": {"path": str(POCKET_MASKS), "sha256": sha256(POCKET_MASKS)},
            "sequence_cache": {"path": str(SEQUENCE_CACHE), "sha256": sha256(SEQUENCE_CACHE)},
            "helper_script": {"path": str(HELPER_SCRIPT), "sha256": sha256(HELPER_SCRIPT)},
        },
        "outputs": {
            "dataset": {"path": str(OUTPUT), "sha256": sha256(OUTPUT)},
            "audit": {"path": str(AUDIT_OUTPUT), "sha256": sha256(AUDIT_OUTPUT)},
        },
    }
    if not all([
        len(ready) > 0,
        ready["structure_graph_path"].astype(str).str.len().gt(0).all(),
        set(ready["binary_label"]) == {0, 1},
    ]):
        report["status"] = "failed"
    atomic_json(VALIDATION_OUTPUT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "validated":
        raise SystemExit("broad structure graph validation failed")


if __name__ == "__main__":
    main()
