#!/usr/bin/env python3
"""Run the three-seed deploy ensemble on direct references or one OOD shard."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


LIGAND_DIM = 512
PROTEIN_DIM = 1536
PROBABILITY_CONVERSION = "sigmoid applied after FP32 logit cast"


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


def load_module(path):
    spec = importlib.util.spec_from_file_location("frozen_main_trainer", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def load_tensor(path, dimension):
    value = torch.load(str(path), map_location="cpu")
    if isinstance(value, dict):
        candidates = [x for x in value.values() if torch.is_tensor(x) and x.ndim == 2 and int(x.shape[-1]) == dimension]
        if not candidates:
            raise TypeError("no compatible tensor in {}".format(path))
        value = candidates[0]
    value = value.detach().cpu().float()
    if value.ndim != 2 or int(value.shape[0]) < 1 or int(value.shape[-1]) != dimension:
        raise ValueError("invalid tensor shape at {}".format(path))
    return value.contiguous()


def load_mmap(path):
    try:
        return torch.load(str(path), map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def load_models(root, package, trainer, device):
    config = json.loads((package / "config.json").read_text(encoding="utf-8"))
    models = {}
    for name in config["models"]:
        values = []
        for seed in config["seeds"]:
            path = package / "gpu_output/deploy_models" / name / "seed_{}".format(seed) / "deploy.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            saved = torch.load(path, map_location=device)
            model = trainer.make_model(name, hidden=256, dropout=0.30, heads=4).to(device)
            model.load_state_dict(saved["model_state_dict"])
            model.eval()
            values.append(model)
        models[name] = values
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


def make_batch(ligand, ligand_mask, protein, protein_lengths, lig_indices, prot_indices):
    lig_index = torch.as_tensor(lig_indices, dtype=torch.long)
    prot_index = torch.as_tensor(prot_indices, dtype=torch.long)
    lig = ligand.index_select(0, lig_index)
    lig_mask = ligand_mask.index_select(0, lig_index).bool()
    lengths = protein_lengths.index_select(0, prot_index).long().clamp(min=1, max=4096)
    maximum = int(lengths.max().item())
    prot = protein.index_select(0, prot_index)[:, :maximum, :]
    positions = torch.arange(maximum).unsqueeze(0)
    prot_mask = positions < lengths.unsqueeze(1)
    # Pocket fields are unused by the three deploy architectures, but the
    # frozen model interface expects the keys to exist.
    return {
        "ligand": lig,
        "ligand_mask": lig_mask,
        "protein": prot,
        "protein_mask": prot_mask,
        "pocket": prot[:, :1, :],
        "pocket_mask": torch.ones(len(prot), 1, dtype=torch.bool),
    }


def benchmark_sets(package):
    table = pd.read_csv(package / "data/CURRENT_BENCHMARK_PAIR_BLACKLIST.tsv.gz", sep="\t")
    return (
        set(table["pair_connectivity_key"].astype(str)),
        set(table.loc[table["seen_in_arm_a"].eq(1), "connectivity_key"].astype(str)),
        set(table.loc[table["seen_in_arm_a"].eq(1), "uniprot"].astype(str)),
    )


def infer_reference(args, root, package, models, device):
    input_path = package / "gpu_cache/REFERENCE_MODEL_READY.tsv.gz"
    output_path = package / "gpu_output/reference/reference_predictions.tsv.gz"
    report_path = package / "gpu_output/reference/REFERENCE_INFERENCE.json"
    frame = pd.read_csv(input_path, sep="\t", low_memory=False)
    if output_path.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") == "validated"
            and report.get("rows") == len(frame)
            and report.get("probability_conversion") == PROBABILITY_CONVERSION
        ):
            print("SKIP completed reference inference", flush=True)
            return

    ligand_paths = sorted(frame["ligand_embedding_path"].astype(str).unique())
    protein_paths = sorted(frame["protein_embedding_path"].astype(str).unique())
    ligand_index = {path: i for i, path in enumerate(ligand_paths)}
    protein_index = {path: i for i, path in enumerate(protein_paths)}
    ligand_values = [load_tensor(path, LIGAND_DIM)[:120] for path in ligand_paths]
    max_atoms = max(int(x.shape[0]) for x in ligand_values)
    ligand = torch.zeros(len(ligand_values), max_atoms, LIGAND_DIM)
    ligand_mask = torch.zeros(len(ligand_values), max_atoms, dtype=torch.bool)
    for i, value in enumerate(ligand_values):
        ligand[i, :len(value)] = value
        ligand_mask[i, :len(value)] = True
    del ligand_values
    protein_values = [load_tensor(path, PROTEIN_DIM)[:4096] for path in protein_paths]
    max_length = max(int(x.shape[0]) for x in protein_values)
    protein = torch.zeros(len(protein_values), max_length, PROTEIN_DIM)
    lengths = torch.zeros(len(protein_values), dtype=torch.long)
    for i, value in enumerate(protein_values):
        protein[i, :len(value)] = value
        lengths[i] = len(value)
    del protein_values
    row_lig = np.asarray([ligand_index[x] for x in frame["ligand_embedding_path"].astype(str)], dtype=int)
    row_prot = np.asarray([protein_index[x] for x in frame["protein_embedding_path"].astype(str)], dtype=int)
    row_lengths = lengths[torch.as_tensor(row_prot)].numpy()
    order = np.argsort(row_lengths, kind="stable")
    result = {name + suffix: np.empty(len(frame), dtype=np.float32) for name in models for suffix in ["_mean", "_sd"]}
    for start in range(0, len(order), args.batch_size_reference):
        selected = order[start:start + args.batch_size_reference]
        batch = make_batch(ligand, ligand_mask, protein, lengths, row_lig[selected], row_prot[selected])
        values = score(models, batch, device)
        for key, value in values.items():
            result[key][selected] = value
        if start % (args.batch_size_reference * 50) == 0:
            print("reference {}/{}".format(min(start + len(selected), len(order)), len(order)), flush=True)
    for key, value in result.items():
        frame["p_" + key] = value
    keep = [
        "reference_row_id", "stable_pair_key", "target_chembl_id", "ligand_chembl_id",
        "target_name", "uniprot", "full_inchikey", "connectivity_key",
        "weak2020_label", "is_5a_positive", "reference_source", "pchembl_numeric",
    ] + ["p_" + key for key in sorted(result)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame[keep].to_csv(output_path, sep="\t", index=False, compression="gzip")
    probability_cols = [x for x in keep if x.startswith("p_") and x.endswith("_mean")]
    valid = bool(frame[probability_cols].notna().all().all() and ((frame[probability_cols] >= 0) & (frame[probability_cols] <= 1)).all().all())
    report = {
        "status": "validated" if valid else "failed", "scope": "reference",
        "rows": int(len(frame)), "proteins": int(frame["uniprot"].nunique()),
        "weak2020_allosteric": int(frame["weak2020_label"].eq(1).sum()),
        "weak2020_orthosteric": int(frame["weak2020_label"].eq(0).sum()),
        "five_a_positive": int(frame["is_5a_positive"].eq(1).sum()),
        "probability_conversion": PROBABILITY_CONVERSION,
        "output": str(output_path),
    }
    atomic_json(report_path, report)
    if not valid:
        raise SystemExit("reference inference validation failed")


def infer_full_shard(args, root, package, models, device):
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
    output_path = package / "gpu_output/full" / ("full_predictions_shard{:02d}of{:02d}.tsv.gz".format(args.shard, args.num_shards))
    report_path = package / "gpu_output/full" / ("FULL_INFERENCE_SHARD{:02d}.json".format(args.shard))
    if output_path.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") == "validated"
            and report.get("probability_conversion") == PROBABILITY_CONVERSION
        ):
            print("SKIP completed full shard {}".format(args.shard), flush=True)
            return
    frame = pd.read_pickle(paths["frame"])
    with paths["mapping"].open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    pair_blacklist, arm_a_compounds, arm_a_proteins = benchmark_sets(package)
    uid = frame["UniProt_ID"].fillna("").astype(str).str.split("-").str[0]
    connectivity = frame["InChIKey14"].fillna("").astype(str).str.upper()
    pair = uid + "|" + connectivity
    excluded = pair.isin(pair_blacklist)
    frame = frame.loc[~excluded].copy().reset_index(drop=True)
    uid = frame["UniProt_ID"].fillna("").astype(str).str.split("-").str[0]
    connectivity = frame["InChIKey14"].fillna("").astype(str).str.upper()
    ligand_map = mapping["lig_path_to_idx"]
    protein_map = mapping["uid_to_idx"]
    row_lig = np.asarray([ligand_map[str(x)] for x in frame["Ligand_Embedding_Path"].astype(str)], dtype=int)
    row_prot = np.asarray([protein_map[str(x).split("-")[0]] for x in uid], dtype=int)
    ligand = load_mmap(paths["ligand"])
    ligand_mask = load_mmap(paths["ligand_mask"])
    protein = load_mmap(paths["protein"])
    lengths = load_mmap(paths["protein_len"]).long()
    row_lengths = lengths[torch.as_tensor(row_prot)].numpy()
    order = np.argsort(row_lengths, kind="stable")
    result = {name + suffix: np.empty(len(frame), dtype=np.float32) for name in models for suffix in ["_mean", "_sd"]}
    for start in range(0, len(order), args.batch_size_full):
        selected = order[start:start + args.batch_size_full]
        batch = make_batch(ligand, ligand_mask, protein, lengths, row_lig[selected], row_prot[selected])
        values = score(models, batch, device)
        for key, value in values.items():
            result[key][selected] = value
        if start % (args.batch_size_full * 100) == 0:
            print("full shard {} {}/{}".format(args.shard, min(start + len(selected), len(order)), len(order)), flush=True)
    output = pd.DataFrame({
        "OOD_Row_ID": frame["OOD_Row_ID"].astype(str),
        "stable_pair_key": frame["target_chembl_id"].fillna("").astype(str) + "|" + frame["ligand_chembl_id"].fillna("").astype(str),
        "target_chembl_id": frame["target_chembl_id"].fillna("").astype(str),
        "ligand_chembl_id": frame["ligand_chembl_id"].fillna("").astype(str),
        "uniprot": uid,
        "connectivity_key": connectivity,
        "legacy_ood_label": pd.to_numeric(frame["Label"], errors="coerce").fillna(-999).astype(int),
        "target_name": frame["target_name"].fillna("").astype(str),
        "pchembl_like": pd.to_numeric(frame["pchembl_like"], errors="coerce"),
        "protein_seen_arm_a": uid.isin(arm_a_proteins).astype(int),
        "compound_seen_arm_a": connectivity.isin(arm_a_compounds).astype(int),
    })
    for key, value in result.items():
        output["p_" + key] = value
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, sep="\t", index=False, compression="gzip")
    probability_cols = [x for x in output.columns if x.startswith("p_") and x.endswith("_mean")]
    valid = bool(output[probability_cols].notna().all().all() and ((output[probability_cols] >= 0) & (output[probability_cols] <= 1)).all().all())
    report = {
        "status": "validated" if valid else "failed", "scope": "full",
        "shard": int(args.shard), "num_shards": int(args.num_shards),
        "source_cache_rows": int(len(frame) + excluded.sum()),
        "current_benchmark_pair_rows_excluded": int(excluded.sum()),
        "scored_rows": int(len(output)),
        "probability_conversion": PROBABILITY_CONVERSION,
        "output": str(output_path),
    }
    atomic_json(report_path, report)
    if not valid:
        raise SystemExit("full inference validation failed")


def main():
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    root = args.project_root
    package = root / "analysis/chembl_external_prioritization"
    trainer = load_module(root / "analysis/allosteric_pair_benchmark_main/scripts/train_main_benchmark.py")
    device = torch.device(args.device)
    models = load_models(root, package, trainer, device)
    if args.scope == "reference":
        infer_reference(args, root, package, models, device)
    else:
        infer_full_shard(args, root, package, models, device)


if __name__ == "__main__":
    main()
