#!/usr/bin/env python3
"""Fail-closed GPU and data preflight before any expensive fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_definitions import VALID_MODELS, make_model  # noqa: E402


EXPECTED_PARAMETERS = {
    "ligand": 132097,
    "protein": 591106,
    "c1": 722178,
    "c2": 1052161,
    "c3": 1315329,
    "d1": 722178,
    "d2": 1052161,
    "d3": 1315329,
}
EXPECTED_ACTIVE_REGIMES = {
    "every_pair": ["row_random", "unseen_family", "unseen_ligand"],
    "protein_anchored": ["row_random", "unseen_family", "unseen_ligand"],
    "protein_ligand_role_complete": [
        "row_random",
        "unseen_family",
        "unseen_ligand",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--scope", choices=["benchmark", "chembl"], required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(root, path):
    try:
        display = str(path.resolve().relative_to(root))
    except ValueError:
        display = str(path.resolve())
    return {
        "path": display,
        "bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }


def canonical_fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def main():
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    root = args.project_root.resolve()
    package = root / "analysis/role_complete_pair_matrix"
    contract = json.loads((package / "validation/CPU_CONTRACT.json").read_text(encoding="utf-8"))
    if (
        contract.get("status") != "validated"
        or contract.get("contract_id") != "role_complete_pair_matrix_v2"
    ):
        raise RuntimeError("CPU contract failed")
    if contract.get("active_training_regimes_by_cohort") != EXPECTED_ACTIVE_REGIMES:
        raise RuntimeError("active cohort-regime contract changed")
    if int(contract.get("expected_benchmark_fits", -1)) != 1080:
        raise RuntimeError("benchmark fit-count contract changed")
    if (
        int(contract.get("contracted_data_and_validation_files", -1)) != 12
        or int(
            contract.get("contracted_implementation_and_document_files", -1)
        )
        != 22
        or int(contract.get("contracted_internal_files_total", -1)) != 34
        or int(contract.get("contracted_external_resources", -1)) != 7
    ):
        raise RuntimeError("CPU checksum inventory contract changed")
    for relative, expected in contract.get("files", {}).items():
        path = package / relative
        if not path.is_file() or sha256(path) != expected["sha256"]:
            raise RuntimeError("CPU-contracted file changed: {}".format(path))
    for relative, expected in contract.get("implementation_files", {}).items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected["sha256"]:
            raise RuntimeError("implementation file changed: {}".format(path))
    for name, expected in contract.get("external_resources", {}).items():
        path = root / expected["path"]
        if not path.is_file() or sha256(path) != expected["sha256"]:
            raise RuntimeError("external resource changed ({}): {}".format(name, path))
    paths_missing = []
    cohort_rows = {}
    pocket_path = root / "analysis/allosteric_pair_benchmark_broad_superset/data/POCKET_INDICES.json"
    with pocket_path.open(encoding="utf-8") as handle:
        pocket_indices = json.load(handle)
    for arm, filename in [
        ("every_pair", "EVERY_PAIR.tsv.gz"),
        ("protein_anchored", "PROTEIN_ANCHORED.tsv.gz"),
        ("protein_ligand_role_complete", "PROTEIN_LIGAND_ROLE_COMPLETE.tsv.gz"),
    ]:
        frame = pd.read_csv(package / "data" / filename, sep="\t", low_memory=False)
        cohort_rows[arm] = int(len(frame))
        if len(frame) != contract["cohorts"][arm]["rows"]:
            raise RuntimeError("{} row contract changed".format(arm))
        expected_evaluation = int(contract["evaluation_rows"][arm])
        if int(frame["matrix_evaluation_eligible"].sum()) != expected_evaluation:
            raise RuntimeError("{} evaluation-row contract changed".format(arm))
        missing_pockets = sorted(set(frame["uniprot"].astype(str)) - set(pocket_indices))
        if missing_pockets:
            raise RuntimeError("{} proteins lack pocket indices: {}".format(arm, missing_pockets[:10]))
        for column in ["protein_embedding_path", "ligand_embedding_path"]:
            for path in frame[column].dropna().astype(str).unique():
                if not Path(path).is_file():
                    paths_missing.append(path)
    if paths_missing:
        raise FileNotFoundError("{} model input paths missing; first examples: {}".format(len(paths_missing), paths_missing[:10]))

    device = torch.device(args.device)
    batch = {
        "ligand": torch.randn(2, 7, 512, device=device),
        "ligand_mask": torch.ones(2, 7, dtype=torch.bool, device=device),
        "protein": torch.randn(2, 11, 1536, device=device),
        "protein_mask": torch.ones(2, 11, dtype=torch.bool, device=device),
        "pocket": torch.randn(2, 5, 1536, device=device),
        "pocket_mask": torch.ones(2, 5, dtype=torch.bool, device=device),
    }
    observed_parameters = {}
    with torch.no_grad():
        for name in VALID_MODELS:
            model = make_model(name, hidden=256, dropout=0.30, heads=4).to(device).eval()
            observed_parameters[name] = int(sum(x.numel() for x in model.parameters() if x.requires_grad))
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(batch)
            probability = torch.sigmoid(logits.float())
            if tuple(probability.shape) != (2,) or not torch.isfinite(probability).all():
                raise RuntimeError("invalid forward pass for {}".format(name))
            del model
    if observed_parameters != EXPECTED_PARAMETERS:
        raise RuntimeError("parameter contract mismatch: {}".format(observed_parameters))

    chembl_cache = None
    if args.scope == "chembl":
        cache = root / "19.Structure_Unknown/6.Build_Fully_Controlled_Model_Dataset_OODTrain/9.Train_Fully_Controlled_Embedding_Model_OODTrain/ood_tensor_cache"
        expected = []
        for shard in range(4):
            tag = "chembl_ood_validation_set.limit0.shard{:02d}of04".format(shard)
            expected.extend(
                [
                    cache / ("df_ood_" + tag + ".pkl"),
                    cache / ("ood_cache_mapping_" + tag + ".json"),
                    cache / ("ood_lig_" + tag + ".pt"),
                    cache / ("ood_lig_mask_" + tag + ".pt"),
                    cache / ("ood_prot_" + tag + ".pt"),
                    cache / ("ood_prot_len_" + tag + ".pt"),
                ]
            )
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing ChEMBL cache artifacts: {}".format(missing[:10]))
        reference = root / contract["external_resources"][
            "chembl_reference_model_ready"
        ]["path"]
        if not reference.is_file():
            raise FileNotFoundError(reference)
        embedding_validation_path = root / contract["external_resources"]["pocket_embedding_validation"]["path"]
        embedding_validation = json.loads(embedding_validation_path.read_text(encoding="utf-8"))
        if (
            embedding_validation.get("status") != "validated"
            or int(embedding_validation.get("validated_proteins", -1)) != 808
            or int(embedding_validation.get("failed_protein_count", -1)) != 0
        ):
            raise RuntimeError("selected-chain ChEMBL pocket embeddings are not validated")
        embedding_root = root / "analysis/property_balanced_chembl/gpu_cache/chembl_target_chain_embeddings"
        if not embedding_root.is_dir():
            raise FileNotFoundError(embedding_root)
        target_table = pd.read_csv(
            root / contract["external_resources"]["pocket_target_chains"]["path"],
            sep="\t",
            usecols=["uniprot"],
        )
        selected_chain_paths = [
            embedding_root / (str(uid) + ".pt")
            for uid in sorted(target_table["uniprot"].astype(str).unique())
        ]
        missing_selected = [str(path) for path in selected_chain_paths if not path.is_file()]
        if missing_selected:
            raise FileNotFoundError(
                "selected-chain tensors missing: {}".format(missing_selected[:10])
            )
        source_manifest = {
            "contract_id": "role_complete_pair_matrix_v2",
            "cpu_contract_sha256": sha256(
                package / "validation/CPU_CONTRACT.json"
            ),
            "reference": file_record(root, reference),
            "ood_cache_artifacts": [file_record(root, path) for path in expected],
            "selected_chain_tensors": [
                file_record(root, path) for path in selected_chain_paths
            ],
            "inference_script_sha256": sha256(
                package / "scripts/infer_chembl.py"
            ),
            "model_implementation_sha256": sha256(
                package / "scripts/model_definitions.py"
            ),
        }
        source_manifest["source_fingerprint"] = canonical_fingerprint(
            source_manifest
        )
        source_manifest["status"] = "validated"
        atomic_json(
            package / "validation/CHEMBL_SOURCE_FINGERPRINT.json",
            source_manifest,
        )
        chembl_cache = {
            "artifacts": len(expected),
            "reference_ready": True,
            "validated_selected_chain_pocket_embeddings": 808,
            "pfam_cache_validated": True,
            "source_fingerprint": source_manifest["source_fingerprint"],
            "source_files_hashed": int(
                1 + len(expected) + len(selected_chain_paths)
            ),
        }

    report = {
        "status": "validated",
        "scope": args.scope,
        "cohort_rows": cohort_rows,
        "missing_model_input_paths": 0,
        "parameter_counts": observed_parameters,
        "all_model_forward_passes": True,
        "probability_conversion": "sigmoid applied after FP32 logit cast",
        "chembl_cache": chembl_cache,
    }
    atomic_json(package / "validation" / "GPU_PREFLIGHT_{}.json".format(args.scope.upper()), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
