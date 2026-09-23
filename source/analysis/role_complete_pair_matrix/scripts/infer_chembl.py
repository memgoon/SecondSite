#!/usr/bin/env python3
"""Score ChEMBL with whole-chain and validated pocket-subset ensembles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_definitions import LIGAND_DIM, MODEL_VERSION, PROTEIN_DIM, make_model  # noqa: E402


PROBABILITY_CONVERSION = "sigmoid applied after FP32 logit cast"
SEEDS = (20260817, 20260818, 20260819)
GENERAL_BASE_MODELS = ("ligand", "protein", "c1", "c2")
POCKET_MODELS = ("d1", "d2", "d3")
GENERAL_MODELS = GENERAL_BASE_MODELS + POCKET_MODELS
BIOCHEMICAL_MODELS = GENERAL_BASE_MODELS + ("c3",) + POCKET_MODELS
GENERAL_DEPLOY_ARMS = {
    "every_pair": "general_every_pair",
    "general": "general_protein_anchored",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--scope", choices=["reference", "full", "biochemical"], required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--biochemical-batch-size", type=int, default=2)
    return parser.parse_args()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def atomic_tsv_gzip(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False, compression="gzip")
    os.replace(str(temporary), str(path))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def validate_epoch_source(package, report):
    source_arm = {
        "general_every_pair": "every_pair",
        "general_protein_anchored": "protein_anchored",
        "biochemical_role_complete": "protein_ligand_role_complete",
    }.get(report.get("deploy_arm"))
    records = report.get("source_report_records")
    if source_arm is None or not isinstance(records, list) or len(records) != 15:
        raise RuntimeError("invalid deploy epoch-source record count")
    identities = set()
    for record in records:
        identity = (
            record.get("cohort_arm"),
            record.get("regime"),
            record.get("model"),
            int(record.get("seed", -1)),
            int(record.get("outer_fold", -1)),
        )
        identities.add(identity)
        source_path = package / str(record.get("report_path", ""))
        source_report = (
            json.loads(source_path.read_text(encoding="utf-8"))
            if source_path.is_file()
            else {}
        )
        source_checkpoint = source_path.parent / "best.pt"
        source_prediction = source_path.parent / "predictions.tsv.gz"
        if (
            identity[0] != source_arm
            or identity[1] != "unseen_family"
            or identity[2] != report.get("model")
            or identity[3] not in SEEDS
            or identity[4] not in range(5)
            or int(record.get("validation_fold", -1)) != (identity[4] + 1) % 5
            or not source_path.is_file()
            or record.get("report_sha256") != sha256(source_path)
            or source_report.get("checkpoint_sha256")
            != record.get("checkpoint_sha256")
            or source_report.get("prediction_sha256")
            != record.get("prediction_sha256")
            or int(source_report.get("best_epoch", -1))
            != int(record.get("best_epoch", -2))
            or not source_checkpoint.is_file()
            or not source_prediction.is_file()
            or sha256(source_checkpoint) != record.get("checkpoint_sha256")
            or sha256(source_prediction) != record.get("prediction_sha256")
        ):
            raise RuntimeError("invalid deploy epoch-source record")
    expected_identities = {
        (source_arm, "unseen_family", report.get("model"), seed, fold)
        for seed in SEEDS
        for fold in range(5)
    }
    payload = {
        "epoch_rule": report.get("epoch_rule"),
        "source_arm": source_arm,
        "model": report.get("model"),
        "records": records,
    }
    fingerprint = canonical_fingerprint(payload)
    if (
        identities != expected_identities
        or report.get("epoch_source_fingerprint") != fingerprint
    ):
        raise RuntimeError("deploy epoch-source fingerprint mismatch")
    return fingerprint


def load_source_fingerprint(package):
    path = package / "validation/CHEMBL_SOURCE_FINGERPRINT.json"
    if not path.is_file():
        raise FileNotFoundError(
            "run gpu_preflight.py --scope chembl before inference: {}".format(path)
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    stored = value.pop("source_fingerprint", None)
    status = value.pop("status", None)
    if (
        status != "validated"
        or value.get("contract_id") != "role_complete_pair_matrix_v2"
        or stored != canonical_fingerprint(value)
        or value.get("cpu_contract_sha256")
        != sha256(package / "validation/CPU_CONTRACT.json")
        or value.get("inference_script_sha256") != sha256(Path(__file__).resolve())
        or value.get("model_implementation_sha256")
        != sha256(package / "scripts/model_definitions.py")
    ):
        raise RuntimeError("invalid ChEMBL source fingerprint")
    value["source_fingerprint"] = stored
    value["status"] = status
    return value


def deploy_checkpoint_records(package, arm, names):
    contract = json.loads(
        (package / "validation/CPU_CONTRACT.json").read_text(encoding="utf-8")
    )
    filename = {
        "general_every_pair": "EVERY_PAIR.tsv.gz",
        "general_protein_anchored": "PROTEIN_ANCHORED.tsv.gz",
        "biochemical_role_complete": "PROTEIN_LIGAND_ROLE_COMPLETE.tsv.gz",
    }[arm]
    expected_data = contract["files"]["data/" + filename]["sha256"]
    expected_model = contract["implementation_files"][
        "analysis/role_complete_pair_matrix/scripts/model_definitions.py"
    ]["sha256"]
    expected_deploy_trainer = contract["implementation_files"][
        "analysis/role_complete_pair_matrix/scripts/train_deploy_models.py"
    ]["sha256"]
    expected_base_trainer = contract["implementation_files"][
        "analysis/allosteric_pair_benchmark_main/scripts/train_main_benchmark.py"
    ]["sha256"]
    rows = []
    for name in names:
        for seed in SEEDS:
            directory = (
                package
                / "gpu_output/deploy"
                / arm
                / name
                / "seed_{}".format(seed)
            )
            checkpoint = directory / "deploy.pt"
            report_path = directory / "DEPLOY_REPORT.json"
            if not checkpoint.is_file() or not report_path.is_file():
                raise FileNotFoundError(directory)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            actual = sha256(checkpoint)
            if (
                report.get("status") != "validated"
                or report.get("contract_id") != "role_complete_pair_matrix_v2"
                or report.get("deploy_arm") != arm
                or report.get("model") != name
                or int(report.get("seed", -1)) != seed
                or report.get("checkpoint_sha256") != actual
                or report.get("model_version") != MODEL_VERSION
                or report.get("training_data_sha256") != expected_data
                or report.get("model_implementation_sha256") != expected_model
                or report.get("deploy_trainer_sha256") != expected_deploy_trainer
                or report.get("base_trainer_sha256") != expected_base_trainer
                or report.get("epoch_rule")
                != (
                    "median best epoch across the 15 unseen-family CV fits "
                    "for the same cohort and model"
                )
                or report.get("epoch_source_regimes") != ["unseen_family"]
            ):
                raise RuntimeError("invalid deploy report: {}".format(report_path))
            epoch_source_fingerprint = validate_epoch_source(package, report)
            rows.append(
                {
                    "deploy_arm": arm,
                    "model": name,
                    "seed": int(seed),
                    "checkpoint_sha256": actual,
                    "training_data_sha256": report.get("training_data_sha256"),
                    "model_implementation_sha256": report.get(
                        "model_implementation_sha256"
                    ),
                    "deploy_trainer_sha256": report.get(
                        "deploy_trainer_sha256"
                    ),
                    "base_trainer_sha256": report.get("base_trainer_sha256"),
                    "epochs": int(report.get("epochs", -1)),
                    "epoch_source_fingerprint": epoch_source_fingerprint,
                }
            )
    return rows


def inference_run_contract(package, scope, shard, num_shards, deploy_map):
    source = load_source_fingerprint(package)
    checkpoints = []
    for _, (deploy_arm, names) in sorted(deploy_map.items()):
        checkpoints.extend(deploy_checkpoint_records(package, deploy_arm, names))
    value = {
        "contract_id": "role_complete_pair_matrix_v2",
        "scope": scope,
        "shard": int(shard),
        "num_shards": int(num_shards),
        "source_fingerprint": source["source_fingerprint"],
        "inference_script_sha256": sha256(Path(__file__).resolve()),
        "checkpoint_records": checkpoints,
        "probability_conversion": PROBABILITY_CONVERSION,
        "selected_chain_required_for_non_ligand_models": True,
    }
    return {
        "run_fingerprint": canonical_fingerprint(value),
        "run_contract": value,
    }


def load_tensor(path, dimension):
    value = torch.load(str(path), map_location="cpu")
    if isinstance(value, dict):
        values = [x for x in value.values() if torch.is_tensor(x) and x.ndim == 2 and int(x.shape[-1]) == dimension]
        if not values:
            raise TypeError("no compatible tensor in {}".format(path))
        value = values[0]
    value = value.detach().cpu().float()
    if value.ndim != 2 or int(value.shape[0]) < 1 or int(value.shape[-1]) != dimension:
        raise ValueError("invalid tensor at {}".format(path))
    return value.contiguous()


def uniform_truncate(tensor, maximum=4096):
    """Mirror the frozen internal PairDataset whole-chain truncation exactly."""
    if maximum <= 0 or int(tensor.shape[0]) <= maximum:
        return tensor
    index = (
        torch.linspace(0, int(tensor.shape[0]) - 1, steps=maximum)
        .round()
        .long()
        .unique(sorted=True)
    )
    return tensor.index_select(0, index)


def load_mmap(path):
    try:
        return torch.load(str(path), map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def cache_paths(root, shard, num_shards):
    tag = "chembl_ood_validation_set.limit0.shard{:02d}of{:02d}".format(shard, num_shards)
    cache = root / "19.Structure_Unknown/6.Build_Fully_Controlled_Model_Dataset_OODTrain/9.Train_Fully_Controlled_Embedding_Model_OODTrain/ood_tensor_cache"
    return {
        "frame": cache / ("df_ood_" + tag + ".pkl"),
        "mapping": cache / ("ood_cache_mapping_" + tag + ".json"),
        "ligand": cache / ("ood_lig_" + tag + ".pt"),
        "ligand_mask": cache / ("ood_lig_mask_" + tag + ".pt"),
        "protein": cache / ("ood_prot_" + tag + ".pt"),
        "protein_len": cache / ("ood_prot_len_" + tag + ".pt"),
    }


def load_models(package, arm, names, device):
    contract = json.loads(
        (package / "validation/CPU_CONTRACT.json").read_text(encoding="utf-8")
    )
    filename = {
        "general_every_pair": "EVERY_PAIR.tsv.gz",
        "general_protein_anchored": "PROTEIN_ANCHORED.tsv.gz",
        "biochemical_role_complete": "PROTEIN_LIGAND_ROLE_COMPLETE.tsv.gz",
    }[arm]
    expected_data_sha256 = contract["files"]["data/" + filename]["sha256"]
    expected_implementation_sha256 = contract["implementation_files"][
        "analysis/role_complete_pair_matrix/scripts/model_definitions.py"
    ]["sha256"]
    expected_deploy_trainer_sha256 = contract["implementation_files"][
        "analysis/role_complete_pair_matrix/scripts/train_deploy_models.py"
    ]["sha256"]
    expected_base_trainer_sha256 = contract["implementation_files"][
        "analysis/allosteric_pair_benchmark_main/scripts/train_main_benchmark.py"
    ]["sha256"]
    result = {}
    for name in names:
        ensemble = []
        for seed in SEEDS:
            directory = package / "gpu_output/deploy" / arm / name / "seed_{}".format(seed)
            path = directory / "deploy.pt"
            report_path = directory / "DEPLOY_REPORT.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            epoch_source_fingerprint = validate_epoch_source(package, report)
            if (
                report.get("status") != "validated"
                or report.get("deploy_arm") != arm
                or report.get("model") != name
                or int(report.get("seed", -1)) != seed
                or report.get("checkpoint_sha256") != sha256(path)
                or report.get("training_data_sha256") != expected_data_sha256
                or report.get("model_implementation_sha256")
                != expected_implementation_sha256
                or report.get("deploy_trainer_sha256")
                != expected_deploy_trainer_sha256
                or report.get("base_trainer_sha256")
                != expected_base_trainer_sha256
                or report.get("epoch_source_fingerprint")
                != epoch_source_fingerprint
            ):
                raise RuntimeError("deploy report contract mismatch: {}".format(report_path))
            saved = torch.load(path, map_location=device)
            if (
                saved.get("contract_id") != "role_complete_pair_matrix_v2"
                or saved.get("model_version") != MODEL_VERSION
                or saved.get("model") != name
                or saved.get("deploy_arm") != arm
                or int(saved.get("seed", -1)) != seed
                or int(saved.get("epochs", -1)) != int(report.get("epochs", -2))
                or saved.get("training_data_sha256") != expected_data_sha256
                or saved.get("model_implementation_sha256")
                != expected_implementation_sha256
                or saved.get("deploy_trainer_sha256")
                != expected_deploy_trainer_sha256
                or saved.get("base_trainer_sha256")
                != expected_base_trainer_sha256
                or saved.get("epoch_source_fingerprint")
                != epoch_source_fingerprint
            ):
                raise RuntimeError("deploy checkpoint contract mismatch: {}".format(path))
            model = make_model(name, hidden=256, dropout=0.30, heads=4).to(device)
            model.load_state_dict(saved["model_state_dict"])
            model.eval()
            ensemble.append(model)
        result[name] = ensemble
    return result


def load_family_context(root, package):
    contract = json.loads(
        (package / "validation/CPU_CONTRACT.json").read_text(encoding="utf-8")
    )
    if contract.get("contract_id") != "role_complete_pair_matrix_v2":
        raise RuntimeError("CPU contract version mismatch")
    resource = contract["external_resources"]["pfam_cache"]
    path = root / resource["path"]
    if sha256(path) != resource["sha256"]:
        raise RuntimeError("Pfam cache hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("validation", {}).get("status") != "PASS":
        raise RuntimeError("Pfam cache is not validated")
    records = payload.get("records", {})
    training = {}
    training_complete = {}
    for arm, filename in [
        ("every_pair", "EVERY_PAIR.tsv.gz"),
        ("general", "PROTEIN_ANCHORED.tsv.gz"),
        ("role_complete", "PROTEIN_LIGAND_ROLE_COMPLETE.tsv.gz"),
    ]:
        frame = pd.read_csv(package / "data" / filename, sep="\t", usecols=["uniprot"])
        proteins = set(frame["uniprot"].astype(str).str.split("-").str[0])
        missing = sorted(proteins - set(records))
        if missing and arm != "every_pair":
            raise RuntimeError("Pfam cache lacks {} proteins: {}".format(arm, missing[:10]))
        training_complete[arm] = not missing
        training[arm] = {
            pfam
            for accession in proteins & set(records)
            for pfam in records[accession].get("pfam_ids", [])
        }
        if not training[arm]:
            raise RuntimeError("{} training Pfam union is empty".format(arm))
    return {
        "records": records,
        "training_pfams": training,
        "training_annotation_complete": training_complete,
        "sha256": resource["sha256"],
    }


def family_annotations(uniprot, context):
    output = {
        "pfam_ids": [],
        "pfam_annotation_status": [],
        "pfam_family_status_every_pair": [],
        "pfam_family_status_general": [],
        "pfam_family_status_role_complete": [],
    }
    records = context["records"]
    for raw in uniprot.astype(str):
        accession = raw.split("-")[0]
        record = records.get(accession)
        if record is None:
            ids = []
            annotation = "not_in_frozen_cache"
        else:
            ids = sorted(set(record.get("pfam_ids", [])))
            annotation = record.get("annotation_status", "unresolved_accession")
        available = annotation == "annotated" and bool(ids)
        output["pfam_ids"].append(";".join(ids) if ids else "NA")
        output["pfam_annotation_status"].append(annotation)
        for arm in ["every_pair", "general", "role_complete"]:
            column = "pfam_family_status_" + arm
            if not available:
                value = "annotation_unavailable"
            elif set(ids) & context["training_pfams"][arm]:
                value = "seen"
            elif not context["training_annotation_complete"][arm]:
                value = "training_reference_incomplete"
            else:
                value = "unseen"
            output[column].append(value)
    return pd.DataFrame(output, index=uniprot.index)


def load_pocket_context(root, package, requested_uids):
    contract = json.loads(
        (package / "validation/CPU_CONTRACT.json").read_text(encoding="utf-8")
    )
    if contract.get("contract_id") != "role_complete_pair_matrix_v2":
        raise RuntimeError("CPU contract version mismatch")
    resources = contract["external_resources"]
    for name in [
        "pocket_target_chains",
        "pocket_indices",
        "pocket_alignment_audit",
        "pocket_coverage",
        "pocket_embedding_validation",
    ]:
        value = resources[name]
        path = root / value["path"]
        if not path.is_file() or sha256(path) != value["sha256"]:
            raise RuntimeError("pocket resource mismatch: {}".format(path))
    validation = json.loads(
        (root / resources["pocket_embedding_validation"]["path"]).read_text(encoding="utf-8")
    )
    if (
        validation.get("status") != "validated"
        or int(validation.get("validated_proteins", -1)) != 808
        or int(validation.get("failed_protein_count", -1)) != 0
    ):
        raise RuntimeError("selected-chain pocket tensors are not validated")
    table = pd.read_csv(
        root / resources["pocket_target_chains"]["path"], sep="\t", low_memory=False
    )
    table["uniprot"] = table["uniprot"].astype(str)
    table = table.set_index("uniprot")
    masks = json.loads(
        (root / resources["pocket_indices"]["path"]).read_text(encoding="utf-8")
    )
    audit = pd.read_csv(
        root / resources["pocket_alignment_audit"]["path"], sep="\t", low_memory=False
    ).fillna("")
    reasons = {
        str(row.uniprot): ("available" if str(row.status) == "ready" else str(row.reason))
        for row in audit.itertuples(index=False)
    }
    requested = {str(value).strip().split("-")[0] for value in requested_uids}
    embedding_root = root / "analysis/property_balanced_chembl/gpu_cache/chembl_target_chain_embeddings"
    protein_cache = {}
    pocket_cache = {}
    for uid in sorted(requested & set(table.index.astype(str))):
        path = embedding_root / (uid + ".pt")
        tensor = load_tensor(path, PROTEIN_DIM)
        if int(tensor.shape[0]) != int(table.loc[uid, "target_chain_length"]):
            raise RuntimeError("selected-chain tensor length mismatch: {}".format(uid))
        protein_cache[uid] = tensor.contiguous()
        if uid in masks:
            index = torch.as_tensor([int(value) for value in masks[uid]], dtype=torch.long)
            if not len(index) or int(index.min()) < 0 or int(index.max()) >= int(tensor.shape[0]):
                raise RuntimeError("invalid pocket indices: {}".format(uid))
            pocket_cache[uid] = tensor.index_select(0, index).contiguous()
    return {
        "protein_cache": protein_cache,
        "pocket_cache": pocket_cache,
        "reasons": reasons,
        "resource_hashes": {
            k: resources[k]["sha256"]
            for k in resources
            if k.startswith("pocket_")
        },
    }


def attach_selected_chains(batch, uids, context, require_pocket=False):
    canonical = [str(value).strip().split("-")[0] for value in uids]
    available = np.asarray(
        [
            uid in context["protein_cache"]
            and (not require_pocket or uid in context["pocket_cache"])
            for uid in canonical
        ],
        dtype=bool,
    )
    selected = np.flatnonzero(available)
    if not len(selected):
        return None, selected
    result = subset_batch(batch, selected)
    selected_uids = [canonical[position] for position in selected]
    maximum_protein = max(
        int(uniform_truncate(context["protein_cache"][uid]).shape[0])
        for uid in selected_uids
    )
    maximum_pocket = max(
        int(context["pocket_cache"][uid].shape[0])
        if uid in context["pocket_cache"]
        else 1
        for uid in selected_uids
    )
    protein = torch.zeros(
        len(selected), maximum_protein, PROTEIN_DIM, dtype=torch.float32
    )
    protein_mask = torch.zeros(len(selected), maximum_protein, dtype=torch.bool)
    pocket = torch.zeros(
        len(selected), maximum_pocket, PROTEIN_DIM, dtype=torch.float32
    )
    pocket_mask = torch.zeros(len(selected), maximum_pocket, dtype=torch.bool)
    for destination, uid in enumerate(selected_uids):
        chain = uniform_truncate(context["protein_cache"][uid])
        pocket_value = context["pocket_cache"].get(
            uid, torch.zeros(1, PROTEIN_DIM, dtype=torch.float32)
        )
        protein[destination, : len(chain)] = chain
        protein_mask[destination, : len(chain)] = True
        pocket[destination, : len(pocket_value)] = pocket_value
        pocket_mask[destination, : len(pocket_value)] = True
    result["protein"] = protein
    result["protein_mask"] = protein_mask
    result["pocket"] = pocket
    result["pocket_mask"] = pocket_mask
    return result, selected


def subset_batch(batch, positions):
    index = torch.as_tensor(positions, dtype=torch.long)
    return {key: value.index_select(0, index) for key, value in batch.items()}


def make_batch(ligand, ligand_mask, protein, lengths, ligand_indices, protein_indices):
    lig_index = torch.as_tensor(ligand_indices, dtype=torch.long)
    prot_index = torch.as_tensor(protein_indices, dtype=torch.long)
    lig = ligand.index_select(0, lig_index)
    lig_mask = ligand_mask.index_select(0, lig_index).bool()
    selected_lengths = lengths.index_select(0, prot_index).long().clamp(min=1, max=4096)
    maximum = int(selected_lengths.max().item())
    prot = protein.index_select(0, prot_index)[:, :maximum, :]
    positions = torch.arange(maximum).unsqueeze(0)
    prot_mask = positions < selected_lengths.unsqueeze(1)
    return {
        "ligand": lig,
        "ligand_mask": lig_mask,
        "protein": prot,
        "protein_mask": prot_mask,
        # Replaced by selected-chain pockets before D-model scoring.  The
        # placeholder keeps base-model batches structurally complete.
        "pocket": prot[:, :1, :],
        "pocket_mask": torch.ones(len(prot), 1, dtype=torch.bool),
    }


@torch.inference_mode()
def score(model_sets, batch, device):
    gpu = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    result = {}
    for arm, models in model_sets.items():
        for name, ensemble in models.items():
            values = []
            for model in ensemble:
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    logits = model(gpu)
                values.append(torch.sigmoid(logits.float()))
            stacked = torch.stack(values, dim=0)
            key = arm + "_" + name
            result[key + "_mean"] = stacked.mean(dim=0).cpu().numpy()
            result[key + "_sd"] = stacked.std(dim=0, unbiased=False).cpu().numpy()
    return result


def score_external_models(model_sets, batch, uids, context, device):
    ligand_sets = {
        arm: {name: value for name, value in models.items() if name == "ligand"}
        for arm, models in model_sets.items()
    }
    chain_sets = {
        arm: {
            name: value
            for name, value in models.items()
            if name != "ligand" and not name.startswith("d")
        }
        for arm, models in model_sets.items()
    }
    pocket_sets = {
        arm: {name: value for name, value in models.items() if name.startswith("d")}
        for arm, models in model_sets.items()
    }
    ligand_sets = {arm: models for arm, models in ligand_sets.items() if models}
    chain_sets = {arm: models for arm, models in chain_sets.items() if models}
    pocket_sets = {arm: models for arm, models in pocket_sets.items() if models}
    values = score(ligand_sets, batch, device) if ligand_sets else {}
    destinations = {
        key: np.arange(len(batch["ligand"]), dtype=int) for key in values
    }
    selected_batch, selected = attach_selected_chains(
        batch, uids, context, require_pocket=False
    )
    if len(selected) and chain_sets:
        selected_values = score(chain_sets, selected_batch, device)
        values.update(selected_values)
        destinations.update({key: selected for key in selected_values})
    pocket_batch, pocket_selected = attach_selected_chains(
        batch, uids, context, require_pocket=True
    )
    if len(pocket_selected) and pocket_sets:
        pocket_values = score(pocket_sets, pocket_batch, device)
        values.update(pocket_values)
        destinations.update({key: pocket_selected for key in pocket_values})
    return values, destinations


def valid_probability_output(frame, score_columns):
    selected_chain_available = frame["selected_chain_available"].astype(int).eq(1)
    pocket_available = frame["pocket_available"].astype(int).eq(1)
    for column in score_columns:
        values = frame[column]
        is_ligand_only = "_ligand_" in column
        is_pocket_model = any("_{}_".format(name) in column for name in POCKET_MODELS)
        if is_pocket_model:
            expected = pocket_available
        elif not is_ligand_only:
            expected = selected_chain_available
        else:
            expected = pd.Series(True, index=frame.index)
        if not is_ligand_only:
            if (
                not values.loc[expected].notna().all()
                or not values.loc[~expected].isna().all()
            ):
                return False
            values = values.loc[expected]
        elif not values.notna().all():
            return False
        if len(values) and not ((values >= 0) & (values <= 1)).all():
            return False
    return True


def completed_output(output_path, report_path, expected):
    if not output_path.is_file() or not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return (
            report.get("status") == "validated"
            and report.get("contract_id") == "role_complete_pair_matrix_v2"
            and report.get("model_version") == MODEL_VERSION
            and report.get("probability_conversion") == PROBABILITY_CONVERSION
            and report.get("models_per_arm") == expected["models_per_arm"]
            and int(report.get("rows", -1)) == int(expected["rows"])
            and report.get("run_fingerprint") == expected["run_fingerprint"]
            and report.get("output_sha256") == sha256(output_path)
        )
    except Exception:
        return False


def benchmark_sets(root, package):
    broad = pd.read_csv(package / "data/EVERY_PAIR.tsv.gz", sep="\t", usecols=["uniprot", "connectivity_key"])
    anchored = pd.read_csv(
        package / "data/PROTEIN_ANCHORED.tsv.gz",
        sep="\t",
        usecols=["uniprot", "connectivity_key"],
    )
    strict = pd.read_csv(
        package / "data/PROTEIN_LIGAND_ROLE_COMPLETE.tsv.gz",
        sep="\t",
        usecols=["uniprot", "connectivity_key"],
    )
    def canonical_protein(frame):
        return frame["uniprot"].astype(str).str.strip().str.split("-").str[0]

    def canonical_ligand(frame):
        return frame["connectivity_key"].astype(str).str.strip().str.upper()

    broad_protein = canonical_protein(broad)
    broad_ligand = canonical_ligand(broad)
    anchored_protein = canonical_protein(anchored)
    anchored_ligand = canonical_ligand(anchored)
    strict_protein = canonical_protein(strict)
    strict_ligand = canonical_ligand(strict)
    broad_pairs = set(broad_protein + "|" + broad_ligand)
    return {
        "broad_pairs": broad_pairs,
        "broad_proteins": set(broad_protein),
        "broad_ligands": set(broad_ligand),
        "anchored_proteins": set(anchored_protein),
        "anchored_ligands": set(anchored_ligand),
        "strict_proteins": set(strict_protein),
        "strict_ligands": set(strict_ligand),
    }


def load_reference_arrays(frame):
    ligand_paths = sorted(frame["ligand_embedding_path"].astype(str).unique())
    ligand_index = {path: i for i, path in enumerate(ligand_paths)}
    ligand_values = [load_tensor(path, LIGAND_DIM)[:120] for path in ligand_paths]
    max_atoms = max(int(x.shape[0]) for x in ligand_values)
    ligand = torch.zeros(len(ligand_values), max_atoms, LIGAND_DIM)
    ligand_mask = torch.zeros(len(ligand_values), max_atoms, dtype=torch.bool)
    for index, value in enumerate(ligand_values):
        ligand[index, : len(value)] = value
        ligand_mask[index, : len(value)] = True
    # Ligand-only scoring requires a structurally complete batch but never
    # reads the protein tensor.  Every non-ligand model replaces this neutral
    # placeholder with the frozen selected-chain tensor in
    # attach_selected_chains().  The legacy mixed UniProt/PDB tensors are not
    # model inputs.
    protein = torch.zeros(1, 1, PROTEIN_DIM)
    lengths = torch.ones(1, dtype=torch.long)
    row_ligand = np.asarray([ligand_index[x] for x in frame["ligand_embedding_path"].astype(str)], dtype=int)
    row_protein = np.zeros(len(frame), dtype=int)
    return ligand, ligand_mask, protein, lengths, row_ligand, row_protein


def infer_reference(args, root, package, device):
    source_package = root / "analysis/chembl_external_prioritization"
    input_path = source_package / "gpu_cache/REFERENCE_MODEL_READY.tsv.gz"
    if not input_path.is_file():
        raise FileNotFoundError("run the existing ChEMBL reference preparation first: {}".format(input_path))
    frame = pd.read_csv(input_path, sep="\t", low_memory=False)
    sets = benchmark_sets(root, package)
    pair = frame["uniprot"].astype(str).str.split("-").str[0] + "|" + frame[
        "connectivity_key"
    ].astype(str).str.upper()
    excluded = pair.isin(sets["broad_pairs"])
    frame = frame.loc[~excluded].copy().reset_index(drop=True)
    # The 2020 reference is a drug-like class-A GPCR set. The role-complete
    # biochemical arm is intentionally not evaluated out of domain here.
    output = package / "gpu_output/chembl/reference_predictions.tsv.gz"
    report_path = package / "gpu_output/chembl/REFERENCE_INFERENCE.json"
    expected_models = {
        arm: list(GENERAL_MODELS) for arm in GENERAL_DEPLOY_ARMS
    }
    deploy_map = {
        arm: (deploy_arm, list(GENERAL_MODELS))
        for arm, deploy_arm in GENERAL_DEPLOY_ARMS.items()
    }
    fingerprint = inference_run_contract(
        package, "reference", 0, 1, deploy_map
    )
    expected = {
        "rows": len(frame),
        "models_per_arm": expected_models,
        "run_fingerprint": fingerprint["run_fingerprint"],
    }
    if completed_output(output, report_path, expected):
        print("SKIP completed reference inference", flush=True)
        return
    models = {
        arm: load_models(package, deploy_arm, GENERAL_MODELS, device)
        for arm, deploy_arm in GENERAL_DEPLOY_ARMS.items()
    }
    family_context = load_family_context(root, package)
    pocket_context = load_pocket_context(root, package, frame["uniprot"])
    arrays = load_reference_arrays(frame)
    canonical_uids = frame["uniprot"].astype(str).str.split("-").str[0]
    selected_chain_lengths = np.asarray(
        [
            int(pocket_context["protein_cache"][uid].shape[0])
            if uid in pocket_context["protein_cache"]
            else 1
            for uid in canonical_uids
        ],
        dtype=int,
    )
    order = np.argsort(selected_chain_lengths, kind="stable")
    result = {
        arm + "_" + model + suffix: np.full(len(frame), np.nan, dtype=np.float32)
        for arm in models
        for model in models[arm]
        for suffix in ["_mean", "_sd"]
    }
    for start in range(0, len(order), args.batch_size):
        selected = order[start : start + args.batch_size]
        batch = make_batch(*arrays[:4], arrays[4][selected], arrays[5][selected])
        uids = frame["uniprot"].iloc[selected].astype(str).tolist()
        values, destinations = score_external_models(
            models, batch, uids, pocket_context, device
        )
        for key, value in values.items():
            destination = selected[destinations[key]]
            result[key][destination] = value
    for key, value in result.items():
        frame["p_" + key] = value
    canonical = frame["uniprot"].astype(str).str.split("-").str[0]
    frame["selected_chain_available"] = canonical.isin(
        pocket_context["protein_cache"]
    ).astype(int)
    frame["pocket_available"] = canonical.isin(
        pocket_context["pocket_cache"]
    ).astype(int)
    frame["pocket_unavailable_reason"] = [
        "available"
        if uid in pocket_context["pocket_cache"]
        else pocket_context["reasons"].get(uid, "target_not_in_frozen_pocket_universe")
        for uid in canonical
    ]
    frame["selected_chain_unavailable_reason"] = [
        "available"
        if uid in pocket_context["protein_cache"]
        else "target_not_in_frozen_selected_chain_universe"
        for uid in canonical
    ]
    family = family_annotations(canonical, family_context)
    for column in family:
        frame[column] = family[column].to_numpy()
    frame["protein_seen_general"] = canonical.isin(sets["anchored_proteins"]).astype(int)
    frame["ligand_seen_general"] = frame["connectivity_key"].astype(str).isin(
        sets["anchored_ligands"]
    ).astype(int)
    frame["protein_seen_every_pair"] = canonical.isin(sets["broad_proteins"]).astype(int)
    frame["ligand_seen_every_pair"] = frame["connectivity_key"].astype(str).str.upper().isin(
        sets["broad_ligands"]
    ).astype(int)
    score_columns = ["p_" + key for key in result]
    valid = valid_probability_output(frame, score_columns)
    if not valid:
        raise SystemExit("invalid reference predictions")
    atomic_tsv_gzip(output, frame)
    atomic_json(
        report_path,
        {
            "status": "validated",
            "contract_id": "role_complete_pair_matrix_v2",
            "model_version": MODEL_VERSION,
            "rows": int(len(frame)),
            "models_per_arm": expected_models,
            "probability_conversion": PROBABILITY_CONVERSION,
            "broad_training_pairs_excluded": int(excluded.sum()),
            "pocket_available_rows": int(frame["pocket_available"].sum()),
            "selected_chain_available_rows": int(
                frame["selected_chain_available"].sum()
            ),
            "non_ligand_models_require_selected_chain": True,
            "legacy_mixed_protein_tensors_used": False,
            "run_fingerprint": fingerprint["run_fingerprint"],
            "run_contract": fingerprint["run_contract"],
            "pfam_cache_sha256": family_context["sha256"],
            "every_pair_pfam_training_annotation_complete": bool(
                family_context["training_annotation_complete"]["every_pair"]
            ),
            "output_sha256": sha256(output),
            "output": str(output),
        },
    )


def infer_shard(args, root, package, device):
    paths = cache_paths(root, args.shard, args.num_shards)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    source = pd.read_pickle(paths["frame"])
    source_rows = int(len(source))
    sets = benchmark_sets(root, package)
    uid = source["UniProt_ID"].fillna("").astype(str).str.split("-").str[0]
    connectivity = source["InChIKey14"].fillna("").astype(str).str.upper()
    pair = uid + "|" + connectivity
    excluded = pair.isin(sets["broad_pairs"])
    frame = source.loc[~excluded].copy().reset_index(drop=True)
    uid = frame["UniProt_ID"].fillna("").astype(str).str.split("-").str[0]
    connectivity = frame["InChIKey14"].fillna("").astype(str).str.upper()
    if args.scope == "biochemical":
        reference = pd.read_csv(
            package / "data/ORTHOSTERIC_SOURCE_LINKED_BIOCHEMICAL_KEYS.tsv", sep="\t"
        )
        keys = set(reference["connectivity_key"].astype(str))
        frame = frame[connectivity.isin(keys)].copy().reset_index(drop=True)
        uid = frame["UniProt_ID"].fillna("").astype(str).str.split("-").str[0]
        connectivity = frame["InChIKey14"].fillna("").astype(str).str.upper()
        model_names = {
            "every_pair": list(BIOCHEMICAL_MODELS),
            "general": list(BIOCHEMICAL_MODELS),
            "role_complete": list(BIOCHEMICAL_MODELS),
        }
        deploy_map = {
            "every_pair": ("general_every_pair", list(BIOCHEMICAL_MODELS)),
            "general": (
                "general_protein_anchored",
                list(BIOCHEMICAL_MODELS),
            ),
            "role_complete": (
                "biochemical_role_complete",
                list(BIOCHEMICAL_MODELS),
            ),
        }
        batch_size = args.biochemical_batch_size
    else:
        model_names = {
            arm: list(GENERAL_MODELS) for arm in GENERAL_DEPLOY_ARMS
        }
        deploy_map = {
            arm: (deploy_arm, list(GENERAL_MODELS))
            for arm, deploy_arm in GENERAL_DEPLOY_ARMS.items()
        }
        batch_size = args.batch_size
    directory = package / "gpu_output/chembl" / args.scope
    output_path = directory / "{}_predictions_shard{:02d}of{:02d}.tsv.gz".format(
        args.scope, args.shard, args.num_shards
    )
    report_path = directory / "{}_SHARD{:02d}.json".format(args.scope.upper(), args.shard)
    fingerprint = inference_run_contract(
        package, args.scope, args.shard, args.num_shards, deploy_map
    )
    expected = {
        "rows": len(frame),
        "models_per_arm": model_names,
        "run_fingerprint": fingerprint["run_fingerprint"],
    }
    if completed_output(output_path, report_path, expected):
        print("SKIP completed {} shard {}".format(args.scope, args.shard), flush=True)
        return
    if args.scope == "biochemical":
        model_sets = {
            "every_pair": load_models(
                package, "general_every_pair", BIOCHEMICAL_MODELS, device
            ),
            "general": load_models(
                package, "general_protein_anchored", BIOCHEMICAL_MODELS, device
            ),
            "role_complete": load_models(
                package, "biochemical_role_complete", BIOCHEMICAL_MODELS, device
            ),
        }
    else:
        model_sets = {
            arm: load_models(package, deploy_arm, GENERAL_MODELS, device)
            for arm, deploy_arm in GENERAL_DEPLOY_ARMS.items()
        }
    family_context = load_family_context(root, package)
    pocket_context = load_pocket_context(root, package, uid)
    with paths["mapping"].open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    row_ligand = np.asarray(
        [mapping["lig_path_to_idx"][str(x)] for x in frame["Ligand_Embedding_Path"].astype(str)], dtype=int
    )
    ligand = load_mmap(paths["ligand"])
    ligand_mask = load_mmap(paths["ligand_mask"])
    protein = torch.zeros(1, 1, PROTEIN_DIM)
    lengths = torch.ones(1, dtype=torch.long)
    row_protein = np.zeros(len(frame), dtype=int)
    selected_chain_lengths = np.asarray(
        [
            int(pocket_context["protein_cache"][value].shape[0])
            if value in pocket_context["protein_cache"]
            else 1
            for value in uid
        ],
        dtype=int,
    )
    order = np.argsort(selected_chain_lengths, kind="stable")
    result = {
        arm + "_" + model + suffix: np.full(len(frame), np.nan, dtype=np.float32)
        for arm in model_sets
        for model in model_sets[arm]
        for suffix in ["_mean", "_sd"]
    }
    for start in range(0, len(order), batch_size):
        selected = order[start : start + batch_size]
        batch = make_batch(ligand, ligand_mask, protein, lengths, row_ligand[selected], row_protein[selected])
        selected_uids = uid.iloc[selected].astype(str).tolist()
        values, destinations = score_external_models(
            model_sets, batch, selected_uids, pocket_context, device
        )
        for key, value in values.items():
            destination = selected[destinations[key]]
            result[key][destination] = value
        if start % max(batch_size * 100, 1) == 0:
            print("{} shard {} {}/{}".format(args.scope, args.shard, min(start + len(selected), len(order)), len(order)), flush=True)
    output = pd.DataFrame(
        {
            "OOD_Row_ID": frame["OOD_Row_ID"].astype(str),
            "stable_pair_key": frame["target_chembl_id"].fillna("").astype(str) + "|" + frame["ligand_chembl_id"].fillna("").astype(str),
            "target_chembl_id": frame["target_chembl_id"].fillna("").astype(str),
            "ligand_chembl_id": frame["ligand_chembl_id"].fillna("").astype(str),
            "uniprot": uid,
            "connectivity_key": connectivity,
            "target_name": frame["target_name"].fillna("").astype(str),
            "pchembl_like": pd.to_numeric(frame["pchembl_like"], errors="coerce"),
            "legacy_ood_label": pd.to_numeric(frame["Label"], errors="coerce").fillna(-999).astype(int),
            "has_allosteric_text": frame["Has_Allosteric_Text"].fillna(False).astype(int),
            "has_orthosteric_text": frame["Has_Orthosteric_Text"].fillna(False).astype(int),
            "protein_seen_role_complete": uid.isin(sets["strict_proteins"]).astype(int),
            "ligand_seen_role_complete": connectivity.isin(sets["strict_ligands"]).astype(int),
            "protein_seen_protein_anchored": uid.isin(sets["anchored_proteins"]).astype(int),
            "protein_seen_every_pair": uid.isin(sets["broad_proteins"]).astype(int),
            "ligand_seen_protein_anchored": connectivity.isin(
                sets["anchored_ligands"]
            ).astype(int),
            "ligand_seen_every_pair": connectivity.isin(sets["broad_ligands"]).astype(int),
            "selected_chain_available": uid.isin(
                pocket_context["protein_cache"]
            ).astype(int),
            "pocket_available": uid.isin(
                pocket_context["pocket_cache"]
            ).astype(int),
            "pocket_unavailable_reason": [
                "available"
                if value in pocket_context["pocket_cache"]
                else pocket_context["reasons"].get(
                    value, "target_not_in_frozen_pocket_universe"
                )
                for value in uid
            ],
            "selected_chain_unavailable_reason": [
                "available"
                if value in pocket_context["protein_cache"]
                else "target_not_in_frozen_selected_chain_universe"
                for value in uid
            ],
            "MW": pd.to_numeric(frame["MW"], errors="coerce"),
            "LogP": pd.to_numeric(frame["LogP"], errors="coerce"),
            "TPSA": pd.to_numeric(frame["TPSA"], errors="coerce"),
        }
    )
    for key, value in result.items():
        output["p_" + key] = value
    family = family_annotations(uid, family_context)
    for column in family:
        output[column] = family[column].to_numpy()
    probability_columns = [x for x in output if x.startswith("p_") and x.endswith("_mean")]
    valid = valid_probability_output(output, probability_columns)
    if not valid:
        raise SystemExit("invalid external predictions")
    atomic_tsv_gzip(output_path, output)
    atomic_json(
        report_path,
        {
            "status": "validated",
            "contract_id": "role_complete_pair_matrix_v2",
            "model_version": MODEL_VERSION,
            "scope": args.scope,
            "shard": int(args.shard),
            "rows": int(len(output)),
            "source_cache_rows": source_rows,
            "broad_training_pairs_excluded": int(excluded.sum()),
            "allosteric_text_rows": int(output["has_allosteric_text"].sum()),
            "orthosteric_text_rows": int(output["has_orthosteric_text"].sum()),
            "models": probability_columns,
            "models_per_arm": model_names,
            "pocket_available_rows": int(output["pocket_available"].sum()),
            "selected_chain_available_rows": int(
                output["selected_chain_available"].sum()
            ),
            "non_ligand_models_require_selected_chain": True,
            "legacy_mixed_protein_tensors_used": False,
            "run_fingerprint": fingerprint["run_fingerprint"],
            "run_contract": fingerprint["run_contract"],
            "pfam_cache_sha256": family_context["sha256"],
            "every_pair_pfam_training_annotation_complete": bool(
                family_context["training_annotation_complete"]["every_pair"]
            ),
            "probability_conversion": PROBABILITY_CONVERSION,
            "output_sha256": sha256(output_path),
            "output": str(output_path),
        },
    )


def main():
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    if not 0 <= args.shard < args.num_shards:
        raise ValueError("invalid shard")
    root = args.project_root.resolve()
    package = root / "analysis/role_complete_pair_matrix"
    device = torch.device(args.device)
    if args.scope == "reference":
        infer_reference(args, root, package, device)
    else:
        infer_shard(args, root, package, device)


if __name__ == "__main__":
    main()
