#!/usr/bin/env python3
"""Score source and property-balanced deploy ensembles in one FP32 pass."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROBABILITY_CONVERSION = "sigmoid applied after FP32 logit cast"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--scope", choices=["reference", "full"], required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size-reference", type=int, default=64)
    parser.add_argument("--batch-size-full", type=int, default=256)
    return parser.parse_args()


def load_family_context(root, config, property_proteins):
    contract = config["family_novelty"]
    cache_path = root / contract["cache_path"]
    observed_hash = sha256_file(cache_path)
    if observed_hash != contract["cache_sha256"]:
        raise RuntimeError("Pfam cache hash mismatch")
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if payload.get("validation", {}).get("status") != "PASS":
        raise RuntimeError("Pfam cache is not validated")
    records = payload.get("records", {})
    missing_training = sorted(set(property_proteins) - set(records))
    if missing_training:
        raise RuntimeError(
            "Pfam cache lacks property proteins: {}".format(missing_training[:10])
        )
    training_pfams = {
        pfam
        for accession in property_proteins
        for pfam in records[accession].get("pfam_ids", [])
    }
    if not training_pfams:
        raise RuntimeError("property-training Pfam union is empty")
    return {
        "cache_sha256": observed_hash,
        "records": records,
        "training_pfams": training_pfams,
    }


def property_sets(root, config):
    frame = pd.read_csv(root / config["property_cohort"]["path"], sep="\t", low_memory=False)
    canonical_protein = frame["uniprot"].astype(str).str.split("-").str[0]
    proteins = set(canonical_protein)
    compounds = set(frame["connectivity_key"].astype(str).str.upper())
    pairs = set(
        canonical_protein
        + "|"
        + frame["connectivity_key"].astype(str).str.upper()
    )
    family_context = load_family_context(root, config, proteins)
    return pairs, compounds, proteins, family_context


def load_models(root, package, existing_package, trainer, device, config):
    epoch_contract = config["c3_deploy_epoch"]
    epoch_path = root / epoch_contract["audit_path"]
    if sha256_file(epoch_path) != epoch_contract["audit_sha256"]:
        raise RuntimeError("deploy-epoch audit hash mismatch")
    epoch_audit = json.loads(epoch_path.read_text(encoding="utf-8"))
    if (
        epoch_audit.get("status") != "PASS"
        or epoch_audit.get("selection_independent_of_chembl") is not True
        or epoch_audit.get("deploy_epochs") != config["deploy_epochs"]
    ):
        raise RuntimeError("deploy-epoch audit is invalid")
    source_index = json.loads(
        (existing_package / "gpu_output/deploy_models/DEPLOY_INDEX.json").read_text(encoding="utf-8")
    )
    if (
        source_index.get("status") != "validated"
        or set(source_index.get("models", [])) != {"ligand", "protein", "c2"}
        or source_index.get("training_rows") != 4637
    ):
        raise RuntimeError("source deploy index is invalid")
    property_index = json.loads(
        (package / "gpu_output/deploy_models/DEPLOY_INDEX.json").read_text(encoding="utf-8")
    )
    if (
        property_index.get("status") != "validated"
        or property_index.get("contract_id") != config["contract_id"]
        or property_index.get("models") != config["deploy_models"]
        or property_index.get("n_completed")
        != len(config["deploy_models"]) * len(config["seeds"])
        or property_index.get("deploy_epochs") != config["deploy_epochs"]
        or property_index.get("cohort_sha256") != config["property_cohort"]["sha256"]
    ):
        raise RuntimeError("property deploy index is invalid")

    specs = {
        "source_ligand": (existing_package / "gpu_output/deploy_models", "ligand"),
        "source_protein": (existing_package / "gpu_output/deploy_models", "protein"),
        "source_c2": (existing_package / "gpu_output/deploy_models", "c2"),
        "property_ligand": (package / "gpu_output/deploy_models", "ligand"),
        "property_c2": (package / "gpu_output/deploy_models", "c2"),
        "property_c3": (package / "gpu_output/deploy_models", "c3"),
    }
    if list(specs) != config["inference_models"]:
        raise RuntimeError("inference model order changed")
    models = {}
    for output_name, (deploy_root, architecture) in specs.items():
        ensemble = []
        for seed in config["seeds"]:
            path = deploy_root / architecture / "seed_{}".format(seed) / "deploy.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            saved = torch.load(path, map_location=device)
            if output_name.startswith("property_"):
                if (
                    saved.get("contract_id") != config["contract_id"]
                    or saved.get("cohort_sha256") != config["property_cohort"]["sha256"]
                    or saved.get("model") != architecture
                    or int(saved.get("epochs", -1))
                    != int(config["deploy_epochs"][architecture])
                ):
                    raise RuntimeError("property checkpoint contract mismatch: {}".format(path))
            model = trainer.make_model(architecture, hidden=256, dropout=0.30, heads=4).to(device)
            model.load_state_dict(saved["model_state_dict"])
            model.eval()
            ensemble.append(model)
        models[output_name] = ensemble
    return models


@torch.inference_mode()
def score(models, batch, device):
    gpu = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    outputs = {}
    for name, ensemble in models.items():
        values = []
        for model in ensemble:
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(gpu)
            values.append(torch.sigmoid(logits.float()))
        stacked = torch.stack(values, dim=0)
        outputs[name + "_mean"] = stacked.mean(dim=0).cpu().numpy()
        outputs[name + "_sd"] = stacked.std(dim=0, unbiased=False).cpu().numpy()
    return outputs


def add_novelty(
    frame,
    uniprot,
    connectivity,
    property_pairs,
    property_compounds,
    property_proteins,
    family_context,
):
    uniprot = uniprot.astype(str).str.split("-").str[0]
    pair = uniprot + "|" + connectivity.astype(str).str.upper()
    result = pd.DataFrame(index=frame.index)
    result["target_seen_property"] = uniprot.isin(property_proteins).astype(int)
    result["compound_seen_property"] = connectivity.astype(str).str.upper().isin(property_compounds).astype(int)
    result["pair_seen_property"] = pair.isin(property_pairs).astype(int)
    result["double_novel_target_identity_and_connectivity"] = (
        result["target_seen_property"].eq(0) & result["compound_seen_property"].eq(0)
    ).astype(int)
    records = family_context["records"]
    training_pfams = family_context["training_pfams"]
    pfam_ids = []
    annotation_status = []
    overlap_status = []
    seen_values = []
    unseen_values = []
    available_values = []
    for accession in uniprot:
        record = records.get(accession)
        if record is None:
            ids = []
            status = "not_in_frozen_cache"
        else:
            ids = sorted(set(record.get("pfam_ids", [])))
            status = record.get("annotation_status", "unresolved_accession")
        available = status == "annotated" and bool(ids)
        seen = available and bool(set(ids) & training_pfams)
        pfam_ids.append(";".join(ids) if ids else "NA")
        annotation_status.append(status)
        available_values.append(int(available))
        if available:
            overlap_status.append("seen" if seen else "unseen")
            seen_values.append(int(seen))
            unseen_values.append(int(not seen))
        else:
            overlap_status.append("annotation_unavailable")
            seen_values.append(pd.NA)
            unseen_values.append(pd.NA)
    result["pfam_ids"] = pfam_ids
    result["pfam_annotation_status"] = annotation_status
    result["pfam_family_annotation_available"] = available_values
    result["family_seen_property"] = pd.array(seen_values, dtype="Int64")
    result["family_unseen_property"] = pd.array(unseen_values, dtype="Int64")
    result["pfam_family_overlap_status"] = overlap_status
    return result


def family_counts(frame):
    return {
        "pfam_annotated_rows": int(frame["pfam_family_annotation_available"].eq(1).sum()),
        "pfam_family_seen_rows": int(frame["family_seen_property"].eq(1).sum()),
        "pfam_family_unseen_rows": int(frame["family_unseen_property"].eq(1).sum()),
        "pfam_annotation_unavailable_rows": int(
            frame["pfam_family_annotation_available"].eq(0).sum()
        ),
    }


def load_c3_pocket_context(root, package, config, requested_uids, trainer):
    """Load only selected-chain pocket tensors required by this inference scope."""
    contract = config["c3_pocket_extension"]
    pinned = {
        "target_chain_path": "target_chain_sha256",
        "pocket_indices_path": "pocket_indices_sha256",
        "alignment_audit_path": "alignment_audit_sha256",
        "coverage_path": "coverage_sha256",
    }
    paths = {}
    for path_key, hash_key in pinned.items():
        path = root / contract[path_key]
        if not path.is_file() or sha256_file(path) != contract[hash_key]:
            raise RuntimeError("C3 pocket-contract hash mismatch: {}".format(path))
        paths[path_key] = path
    coverage = json.loads(paths["coverage_path"].read_text(encoding="utf-8"))
    if coverage.get("status") != "PASS":
        raise RuntimeError("C3 pocket coverage is not validated")
    embedding_validation_path = root / contract["embedding_validation_path"]
    embedding_validation = json.loads(
        embedding_validation_path.read_text(encoding="utf-8")
    )
    expected_embeddings = int(coverage["union"]["ready_proteins"])
    if (
        embedding_validation.get("status") != "validated"
        or int(embedding_validation.get("expected_proteins", -1))
        != expected_embeddings
        or int(embedding_validation.get("validated_proteins", -1))
        != expected_embeddings
        or int(embedding_validation.get("failed_protein_count", -1)) != 0
        or embedding_validation.get("stale_embedding_files")
    ):
        raise RuntimeError("C3 target-chain embeddings are not fully validated")

    table = pd.read_csv(paths["target_chain_path"], sep="\t", low_memory=False)
    if table["uniprot"].duplicated().any():
        raise RuntimeError("duplicate C3 target-chain UniProt accession")
    table = table.set_index("uniprot")
    masks = json.loads(paths["pocket_indices_path"].read_text(encoding="utf-8"))
    audit = pd.read_csv(
        paths["alignment_audit_path"], sep="\t", low_memory=False
    ).fillna("")
    if audit["uniprot"].duplicated().any():
        raise RuntimeError("duplicate C3 pocket-audit UniProt accession")
    reasons = {
        str(row.uniprot): (
            "available" if str(row.status) == "ready" else str(row.reason)
        )
        for row in audit.itertuples(index=False)
    }
    requested = {
        str(value).strip().split("-")[0] for value in requested_uids
    }
    ready = sorted(requested & set(masks))
    embedding_root = root / contract["embedding_directory"]
    cache = {}
    for uid in ready:
        row = table.loc[uid]
        path = embedding_root / (uid + ".pt")
        tensor = trainer.tensor_from_object(
            torch.load(path, map_location="cpu"), trainer.PROTEIN_DIM
        )
        if int(tensor.shape[0]) != int(row["target_chain_length"]):
            raise RuntimeError("C3 selected-chain tensor length mismatch: {}".format(uid))
        index = torch.tensor([int(value) for value in masks[uid]], dtype=torch.long)
        if (
            not len(index)
            or int(index.min()) < 0
            or int(index.max()) >= int(tensor.shape[0])
        ):
            raise RuntimeError("invalid C3 pocket indices: {}".format(uid))
        cache[uid] = tensor.index_select(0, index).contiguous()
    return {
        "cache": cache,
        "reasons": reasons,
        "requested_proteins": len(requested),
        "available_proteins": len(cache),
        "coverage_sha256": contract["coverage_sha256"],
        "amendment_id": contract["amendment_id"],
    }


def attach_c3_pockets(batch, uids, pocket_cache):
    canonical = [str(value).strip().split("-")[0] for value in uids]
    available = np.asarray([uid in pocket_cache for uid in canonical], dtype=bool)
    maximum = max(
        [int(pocket_cache[uid].shape[0]) for uid in canonical if uid in pocket_cache]
        or [1]
    )
    pocket = torch.zeros(len(canonical), maximum, 1536, dtype=torch.float32)
    pocket_mask = torch.zeros(len(canonical), maximum, dtype=torch.bool)
    for position, uid in enumerate(canonical):
        if uid not in pocket_cache:
            continue
        value = pocket_cache[uid]
        pocket[position, : len(value)] = value
        pocket_mask[position, : len(value)] = True
    result = dict(batch)
    result["pocket"] = pocket
    result["pocket_mask"] = pocket_mask
    return result, available


def subset_batch(batch, selected):
    index = torch.as_tensor(selected, dtype=torch.long)
    return {key: value.index_select(0, index) for key, value in batch.items()}


def score_base_and_c3(models, batch, available, device):
    base_models = {key: value for key, value in models.items() if key != "property_c3"}
    c3_models = {key: value for key, value in models.items() if key == "property_c3"}
    if set(c3_models) != {"property_c3"}:
        raise RuntimeError("exactly one property C3 ensemble is required")
    values = score(base_models, batch, device)
    selected = np.flatnonzero(available)
    if len(selected):
        values.update(score(c3_models, subset_batch(batch, selected), device))
    return values, selected


def validate_probability_output(frame, models):
    base = [
        "p_{}_mean".format(model)
        for model in models if model != "property_c3"
    ]
    c3 = ["p_property_c3_mean", "p_property_c3_sd"]
    base_ok = bool(
        frame[base].notna().all().all()
        and ((frame[base] >= 0) & (frame[base] <= 1)).all().all()
    )
    available = frame["c3_pocket_available"].astype(int).eq(1)
    c3_available = frame.loc[available, c3]
    c3_ok = bool(
        len(c3_available)
        and c3_available.notna().all().all()
        and ((c3_available >= 0) & (c3_available <= 1)).all().all()
        and frame.loc[~available, c3].isna().all().all()
    )
    return base_ok and c3_ok


def annotate_existing_output(
    output_path,
    report_path,
    report,
    root,
    config,
):
    frame = pd.read_csv(output_path, sep="\t", low_memory=False)
    required = {"uniprot", "connectivity_key"}
    if not required <= set(frame):
        return False
    property_pairs, property_compounds, property_proteins, family_context = property_sets(
        root, config
    )
    novelty = add_novelty(
        frame,
        frame["uniprot"],
        frame["connectivity_key"],
        property_pairs,
        property_compounds,
        property_proteins,
        family_context,
    )
    for column in novelty:
        frame[column] = novelty[column].to_numpy()
    if frame["pair_seen_property"].astype(int).any():
        raise RuntimeError("cached predictions contain a property training pair")
    atomic_tsv_gzip(output_path, frame)
    report.update({
        "family_novelty_amendment_id": config["family_novelty"]["amendment_id"],
        "family_novelty_cache_sha256": family_context["cache_sha256"],
        "family_novelty_counts": family_counts(frame),
        "family_annotation_added_without_model_inference": True,
    })
    atomic_json(report_path, report)
    return True


def infer_reference(
    args, root, package, existing_package, helper, trainer, models, device, config
):
    input_path = existing_package / "gpu_cache/REFERENCE_MODEL_READY.tsv.gz"
    output_dir = package / "gpu_output/reference"
    output_path = output_dir / "reference_predictions.tsv.gz"
    report_path = output_dir / "REFERENCE_INFERENCE.json"
    frame = pd.read_csv(input_path, sep="\t", low_memory=False)
    if output_path.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        base_valid = (
            report.get("status") == "validated"
            and report.get("rows") == len(frame)
            and report.get("contract_id") == config["contract_id"]
            and report.get("probability_conversion") == PROBABILITY_CONVERSION
            and report.get("models") == config["inference_models"]
            and report.get("c3_pocket_amendment_id")
            == config["c3_pocket_extension"]["amendment_id"]
            and report.get("c3_pocket_coverage_sha256")
            == config["c3_pocket_extension"]["coverage_sha256"]
        )
        if base_valid:
            if (
                report.get("family_novelty_amendment_id")
                != config["family_novelty"]["amendment_id"]
                or report.get("family_novelty_cache_sha256")
                != config["family_novelty"]["cache_sha256"]
            ):
                if annotate_existing_output(
                    output_path, report_path, report, root, config
                ):
                    print(
                        "ANNOTATED completed reference predictions without model inference",
                        flush=True,
                    )
                    return
            else:
                print("SKIP completed property reference inference", flush=True)
                return

    pocket_context = load_c3_pocket_context(
        root, package, config, frame["uniprot"], trainer
    )

    ligand_paths = sorted(frame["ligand_embedding_path"].astype(str).unique())
    protein_paths = sorted(frame["protein_embedding_path"].astype(str).unique())
    ligand_index = {path: index for index, path in enumerate(ligand_paths)}
    protein_index = {path: index for index, path in enumerate(protein_paths)}
    ligand_values = [helper.load_tensor(path, helper.LIGAND_DIM)[:120] for path in ligand_paths]
    max_atoms = max(int(x.shape[0]) for x in ligand_values)
    ligand = torch.zeros(len(ligand_values), max_atoms, helper.LIGAND_DIM)
    ligand_mask = torch.zeros(len(ligand_values), max_atoms, dtype=torch.bool)
    for index, value in enumerate(ligand_values):
        ligand[index, :len(value)] = value
        ligand_mask[index, :len(value)] = True
    del ligand_values
    protein_values = [helper.load_tensor(path, helper.PROTEIN_DIM)[:4096] for path in protein_paths]
    max_length = max(int(x.shape[0]) for x in protein_values)
    protein = torch.zeros(len(protein_values), max_length, helper.PROTEIN_DIM)
    lengths = torch.zeros(len(protein_values), dtype=torch.long)
    for index, value in enumerate(protein_values):
        protein[index, :len(value)] = value
        lengths[index] = len(value)
    del protein_values
    row_ligand = np.asarray([ligand_index[x] for x in frame["ligand_embedding_path"].astype(str)], dtype=int)
    row_protein = np.asarray([protein_index[x] for x in frame["protein_embedding_path"].astype(str)], dtype=int)
    row_lengths = lengths[torch.as_tensor(row_protein)].numpy()
    order = np.argsort(row_lengths, kind="stable")
    result = {
        name + suffix: np.full(len(frame), np.nan, dtype=np.float32)
        for name in models for suffix in ["_mean", "_sd"]
    }
    for start in range(0, len(order), args.batch_size_reference):
        selected = order[start:start + args.batch_size_reference]
        batch = helper.make_batch(
            ligand, ligand_mask, protein, lengths,
            row_ligand[selected], row_protein[selected],
        )
        selected_uids = frame["uniprot"].iloc[selected].astype(str).tolist()
        batch, available = attach_c3_pockets(
            batch, selected_uids, pocket_context["cache"]
        )
        values, c3_positions = score_base_and_c3(models, batch, available, device)
        for key, value in values.items():
            destination = selected[c3_positions] if key.startswith("property_c3") else selected
            result[key][destination] = value
        if start % (args.batch_size_reference * 50) == 0:
            print("reference {}/{}".format(min(start + len(selected), len(order)), len(order)), flush=True)
    for key, value in result.items():
        frame["p_" + key] = value
    canonical_uids = frame["uniprot"].astype(str).str.split("-").str[0]
    frame["c3_pocket_available"] = canonical_uids.isin(
        pocket_context["cache"]
    ).astype(int)
    frame["c3_pocket_unavailable_reason"] = [
        "available" if uid in pocket_context["cache"] else
        pocket_context["reasons"].get(uid, "target_not_in_frozen_pocket_universe")
        for uid in canonical_uids
    ]

    property_pairs, property_compounds, property_proteins, family_context = property_sets(
        root, config
    )
    uniprot = frame["uniprot"].astype(str)
    connectivity = frame["connectivity_key"].astype(str).str.upper()
    novelty = add_novelty(
        frame, uniprot, connectivity, property_pairs, property_compounds,
        property_proteins, family_context,
    )
    for column in novelty:
        frame[column] = novelty[column].to_numpy()
    if frame["pair_seen_property"].any():
        raise RuntimeError("reference cache contains a property training pair")
    keep = [
        "reference_row_id", "stable_pair_key", "target_chembl_id", "ligand_chembl_id",
        "target_name", "uniprot", "full_inchikey", "connectivity_key",
        "weak2020_label", "is_5a_positive", "reference_source", "pchembl_numeric",
        "target_seen_property", "compound_seen_property", "pair_seen_property",
        "double_novel_target_identity_and_connectivity",
        "pfam_ids", "pfam_annotation_status", "pfam_family_annotation_available",
        "family_seen_property", "family_unseen_property",
        "pfam_family_overlap_status",
        "c3_pocket_available", "c3_pocket_unavailable_reason",
    ] + ["p_" + key for key in sorted(result)]
    output_dir.mkdir(parents=True, exist_ok=True)
    frame[keep].to_csv(output_path, sep="\t", index=False, compression="gzip")
    valid = validate_probability_output(frame, config["inference_models"])
    report = {
        "status": "validated" if valid else "failed",
        "contract_id": config["contract_id"],
        "scope": "reference",
        "rows": int(len(frame)),
        "proteins": int(frame["uniprot"].nunique()),
        "weak2020_allosteric": int(frame["weak2020_label"].eq(1).sum()),
        "weak2020_orthosteric": int(frame["weak2020_label"].eq(0).sum()),
        "five_a_positive": int(frame["is_5a_positive"].eq(1).sum()),
        "models": config["inference_models"],
        "probability_conversion": PROBABILITY_CONVERSION,
        "family_novelty_amendment_id": config["family_novelty"]["amendment_id"],
        "family_novelty_cache_sha256": family_context["cache_sha256"],
        "family_novelty_counts": family_counts(frame),
        "family_annotation_added_without_model_inference": False,
        "c3_pocket_amendment_id": pocket_context["amendment_id"],
        "c3_pocket_coverage_sha256": pocket_context["coverage_sha256"],
        "c3_pocket_available_proteins": int(
            frame.loc[frame["c3_pocket_available"].eq(1), "uniprot"].nunique()
        ),
        "c3_pocket_available_rows": int(frame["c3_pocket_available"].sum()),
        "c3_pocket_unavailable_rows": int(frame["c3_pocket_available"].eq(0).sum()),
        "c3_metric_scope": "pocket_available_rows_only",
        "output": str(output_path),
    }
    atomic_json(report_path, report)
    if not valid:
        raise SystemExit("reference inference validation failed")


def infer_full(
    args, root, package, existing_package, helper, trainer, models, device, config
):
    if args.shard < 0 or args.shard >= args.num_shards:
        raise ValueError("invalid shard")
    tag = "chembl_ood_validation_set.limit0.shard{:02d}of{:02d}".format(args.shard, args.num_shards)
    cache = root / "19.Structure_Unknown/6.Build_Fully_Controlled_Model_Dataset_OODTrain/9.Train_Fully_Controlled_Embedding_Model_OODTrain/ood_tensor_cache"
    paths = {
        "frame": cache / ("df_ood_" + tag + ".pkl"),
        "mapping": cache / ("ood_cache_mapping_" + tag + ".json"),
        "ligand": cache / ("ood_lig_" + tag + ".pt"),
        "ligand_mask": cache / ("ood_lig_mask_" + tag + ".pt"),
        "protein": cache / ("ood_prot_" + tag + ".pt"),
        "protein_len": cache / ("ood_prot_len_" + tag + ".pt"),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = package / "gpu_output/full"
    output_path = output_dir / "full_predictions_shard{:02d}of{:02d}.tsv.gz".format(args.shard, args.num_shards)
    report_path = output_dir / "FULL_INFERENCE_SHARD{:02d}.json".format(args.shard)
    if output_path.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        base_valid = (
            report.get("status") == "validated"
            and report.get("contract_id") == config["contract_id"]
            and report.get("probability_conversion") == PROBABILITY_CONVERSION
            and report.get("models") == config["inference_models"]
            and report.get("c3_pocket_amendment_id")
            == config["c3_pocket_extension"]["amendment_id"]
            and report.get("c3_pocket_coverage_sha256")
            == config["c3_pocket_extension"]["coverage_sha256"]
        )
        if base_valid:
            if (
                report.get("family_novelty_amendment_id")
                != config["family_novelty"]["amendment_id"]
                or report.get("family_novelty_cache_sha256")
                != config["family_novelty"]["cache_sha256"]
            ):
                if annotate_existing_output(
                    output_path, report_path, report, root, config
                ):
                    print(
                        "ANNOTATED completed full shard {} without model inference".format(
                            args.shard
                        ),
                        flush=True,
                    )
                    return
            else:
                print("SKIP completed full shard {}".format(args.shard), flush=True)
                return

    frame = pd.read_pickle(paths["frame"])
    source_rows = int(len(frame))
    with paths["mapping"].open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    pair_blacklist, _, _ = helper.benchmark_sets(existing_package)
    uniprot = frame["UniProt_ID"].fillna("").astype(str).str.split("-").str[0]
    connectivity = frame["InChIKey14"].fillna("").astype(str).str.upper()
    pair = uniprot + "|" + connectivity
    excluded = pair.isin(pair_blacklist)
    frame = frame.loc[~excluded].copy().reset_index(drop=True)
    uniprot = frame["UniProt_ID"].fillna("").astype(str).str.split("-").str[0]
    connectivity = frame["InChIKey14"].fillna("").astype(str).str.upper()
    pocket_context = load_c3_pocket_context(
        root, package, config, uniprot, trainer
    )

    ligand_map = mapping["lig_path_to_idx"]
    protein_map = mapping["uid_to_idx"]
    row_ligand = np.asarray(
        [ligand_map[str(x)] for x in frame["Ligand_Embedding_Path"].astype(str)], dtype=int
    )
    row_protein = np.asarray([protein_map[str(x)] for x in uniprot], dtype=int)
    ligand = helper.load_mmap(paths["ligand"])
    ligand_mask = helper.load_mmap(paths["ligand_mask"])
    protein = helper.load_mmap(paths["protein"])
    lengths = helper.load_mmap(paths["protein_len"]).long()
    row_lengths = lengths[torch.as_tensor(row_protein)].numpy()
    order = np.argsort(row_lengths, kind="stable")
    result = {
        name + suffix: np.full(len(frame), np.nan, dtype=np.float32)
        for name in models for suffix in ["_mean", "_sd"]
    }
    for start in range(0, len(order), args.batch_size_full):
        selected = order[start:start + args.batch_size_full]
        batch = helper.make_batch(
            ligand, ligand_mask, protein, lengths,
            row_ligand[selected], row_protein[selected],
        )
        selected_uids = uniprot.iloc[selected].astype(str).tolist()
        batch, available = attach_c3_pockets(
            batch, selected_uids, pocket_context["cache"]
        )
        values, c3_positions = score_base_and_c3(models, batch, available, device)
        for key, value in values.items():
            destination = selected[c3_positions] if key.startswith("property_c3") else selected
            result[key][destination] = value
        if start % (args.batch_size_full * 100) == 0:
            print(
                "full shard {} {}/{}".format(
                    args.shard, min(start + len(selected), len(order)), len(order)
                ),
                flush=True,
            )

    property_pairs, property_compounds, property_proteins, family_context = property_sets(
        root, config
    )
    novelty = add_novelty(
        frame, uniprot, connectivity, property_pairs, property_compounds,
        property_proteins, family_context,
    )
    output = pd.DataFrame({
        "OOD_Row_ID": frame["OOD_Row_ID"].astype(str),
        "stable_pair_key": frame["target_chembl_id"].fillna("").astype(str) + "|" + frame["ligand_chembl_id"].fillna("").astype(str),
        "target_chembl_id": frame["target_chembl_id"].fillna("").astype(str),
        "ligand_chembl_id": frame["ligand_chembl_id"].fillna("").astype(str),
        "uniprot": uniprot,
        "connectivity_key": connectivity,
        "legacy_ood_label": pd.to_numeric(frame["Label"], errors="coerce").fillna(-999).astype(int),
        "target_name": frame["target_name"].fillna("").astype(str),
        "pchembl_like": pd.to_numeric(frame["pchembl_like"], errors="coerce"),
        "c3_pocket_available": uniprot.isin(pocket_context["cache"]).astype(int),
        "c3_pocket_unavailable_reason": [
            "available" if uid in pocket_context["cache"] else
            pocket_context["reasons"].get(
                uid, "target_not_in_frozen_pocket_universe"
            )
            for uid in uniprot
        ],
    })
    for column in novelty:
        output[column] = novelty[column].to_numpy()
    if output["pair_seen_property"].any():
        raise RuntimeError("full screen retained a property training pair")
    for key, value in result.items():
        output["p_" + key] = value
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, sep="\t", index=False, compression="gzip")
    valid = validate_probability_output(output, config["inference_models"])
    report = {
        "status": "validated" if valid else "failed",
        "contract_id": config["contract_id"],
        "scope": "full",
        "shard": int(args.shard),
        "num_shards": int(args.num_shards),
        "source_cache_rows": source_rows,
        "current_benchmark_pair_rows_excluded": int(excluded.sum()),
        "scored_rows": int(len(output)),
        "models": config["inference_models"],
        "probability_conversion": PROBABILITY_CONVERSION,
        "family_novelty_amendment_id": config["family_novelty"]["amendment_id"],
        "family_novelty_cache_sha256": family_context["cache_sha256"],
        "family_novelty_counts": family_counts(output),
        "family_annotation_added_without_model_inference": False,
        "c3_pocket_amendment_id": pocket_context["amendment_id"],
        "c3_pocket_coverage_sha256": pocket_context["coverage_sha256"],
        "c3_pocket_available_proteins": int(
            output.loc[output["c3_pocket_available"].eq(1), "uniprot"].nunique()
        ),
        "c3_pocket_available_rows": int(output["c3_pocket_available"].sum()),
        "c3_pocket_unavailable_rows": int(output["c3_pocket_available"].eq(0).sum()),
        "c3_metric_scope": "pocket_available_rows_only",
        "output": str(output_path),
    }
    atomic_json(report_path, report)
    if not valid:
        raise SystemExit("full inference validation failed")


def main():
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    root = args.project_root
    package = root / "analysis/property_balanced_chembl"
    existing_package = root / "analysis/chembl_external_prioritization"
    config = json.loads((package / "config.json").read_text(encoding="utf-8"))
    helper = load_module("existing_chembl_helpers", existing_package / "scripts/infer_chembl.py")
    trainer = load_module(
        "property_chembl_trainer",
        root / "analysis/allosteric_pair_benchmark_main/scripts/train_main_benchmark.py",
    )
    device = torch.device(args.device)
    models = load_models(root, package, existing_package, trainer, device, config)
    if args.scope == "reference":
        infer_reference(
            args, root, package, existing_package, helper, trainer, models,
            device, config,
        )
    else:
        infer_full(
            args, root, package, existing_package, helper, trainer, models,
            device, config,
        )


if __name__ == "__main__":
    main()
