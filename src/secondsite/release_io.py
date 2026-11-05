"""Read the published SecondSite data and checkpoint layouts."""

import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

DATASETS = (
    "every_pair",
    "both_class_proteins",
    "both_class_ligands",
    "both_class_proteins_ligands",
)
MODELS = ("ligand", "protein", "c1", "c2", "c3", "d1", "d2", "d3")
COHORT_IDS = {
    "both_classed_protein": "both_class_proteins",
    "both_classed_ligand": "both_class_ligands",
    "both_classed_proteins_and_ligands": "both_class_proteins_ligands",
}
DATA_REVISION = "d47562ca2832a7ef32853dc73f6409773184745f"
MODEL_REVISION = "a7b5412e5007cd616793c3d678f82d65f1c54c48"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def release_path(root, relative):
    p = PurePosixPath(str(relative))
    if p.is_absolute() or ".." in p.parts or not str(relative):
        raise ValueError("Expected a path relative to the release root")
    return Path(root) / str(p)


def rename_checked(frame, old, new):
    if old not in frame:
        return frame
    if new in frame:
        a, b = frame[old], frame[new]
        if not ((a == b) | (a.isna() & b.isna())).all():
            raise ValueError(f"Conflicting columns: {old}, {new}")
        return frame.drop(columns=old)
    return frame.rename(columns={old: new})


def prediction_frame(frame):
    out = frame.copy()
    for old, new in [("cohort_arm", "dataset"), ("p_allosteric", "score")]:
        out = rename_checked(out, old, new)
    if "dataset" in out:
        out["dataset"] = out.dataset.replace(COHORT_IDS)
    if "score" not in out:
        raise ValueError("Prediction input needs score or p_allosteric")
    out["score"] = pd.to_numeric(out.score, errors="raise")
    return out


def read_predictions(path):
    return prediction_frame(pd.read_csv(path, sep="\t"))


def frozen_partition(frame, membership, dataset, regime, fold):
    """Apply released train/validation/test membership without recalculating splits."""
    from .build_datasets import disjoint

    m = pd.read_csv(membership, sep="\t", dtype={"main_row_id": str})
    m = rename_checked(m, "cohort_arm", "dataset")
    if "dataset" in m:
        m = m[m.dataset.replace(COHORT_IDS).eq(dataset)]
    if "regime" in m:
        m = m[m.regime.eq(regime)]
    m = m[m.outer_fold.astype(str).eq(str(fold))].copy()
    if (
        m.empty
        or m.main_row_id.duplicated().any()
        or not m.partition.isin(["train", "validation", "test", "excluded"]).all()
    ):
        raise ValueError("Invalid frozen partition")
    if not set(m.main_row_id) <= set(frame.main_row_id):
        raise ValueError("Frozen membership contains unknown row IDs")
    parts = [
        frame[frame.main_row_id.isin(m.loc[m.partition.eq(p), "main_row_id"])].copy()
        for p in ("train", "validation", "test")
    ]
    columns = ["main_row_id"]
    if regime in ("pfam_held_out", "double_held_out"):
        columns += ["uniprot", "family_component_id"]
    if regime in ("ligand_held_out", "double_held_out"):
        columns += ["connectivity_key"]
    disjoint(parts, columns)
    return tuple(parts)


class DataRelease:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = pd.read_csv(self.root / "MANIFEST.tsv", sep="\t")
        if self.manifest.destination.duplicated().any():
            raise ValueError("Duplicate manifest path")
        self.files = self.manifest.set_index("destination")

    def path(self, relative, verify=False):
        if relative not in self.files.index:
            raise ValueError("File is not listed in the data manifest: " + relative)
        p = release_path(self.root, relative)
        if not p.is_file():
            raise FileNotFoundError(p)
        if verify and sha256(p) != self.files.loc[relative, "sha256"]:
            raise ValueError("Data checksum mismatch: " + relative)
        return p

    def table(self, relative):
        return pd.read_csv(self.path(relative), sep="\t")

    def cohort(self, dataset):
        if dataset not in DATASETS:
            raise ValueError(dataset)
        table = self.table(f"benchmark/{dataset}.tsv.gz")
        if table.main_row_id.duplicated().any():
            raise ValueError("Duplicate benchmark row ID")
        return table

    def pockets(self, dataset):
        suffix = {
            "every_pair": "main_benchmark",
            "both_class_proteins": "main_benchmark",
            "both_class_ligands": "both_class_ligands",
            "both_class_proteins_ligands": "both_class_proteins_ligands",
        }[dataset]
        return json.loads(self.path(f"inputs/pockets/{suffix}.json").read_text())

    def chembl_scores(self, dataset, population="reference"):
        if dataset not in DATASETS or population not in (
            "reference",
            "full",
            "biochemical",
        ):
            raise ValueError("Unknown ChEMBL collection")
        pairs = self.table(f"chembl/{population}/pairs.tsv.gz")
        scores = self.table(f"chembl/{population}/{dataset}.tsv.gz")
        key = "reference_row_id" if population == "reference" else "OOD_Row_ID"
        result = pairs.merge(scores, on=key, how="inner", validate="one_to_one")
        if len(result) != len(pairs) or len(result) != len(scores):
            raise ValueError("ChEMBL pair and score row IDs differ")
        for model in MODELS:
            found = [
                c
                for c in scores
                if c.startswith("p_") and c.endswith("_" + model + "_mean")
            ]
            if len(found) != 1:
                raise ValueError(f"Ambiguous score field: {dataset}/{model}")
            result[model] = result[found[0]]
        return result

    def biolip_reference(self):
        return self.table("biolip/evaluation/REFERENCE_PAIRS.tsv")

    def biolip_scores(self):
        scores = self.table("biolip/evaluation/OOF_PAIR_SCORES.tsv.gz")
        identity = self.biolip_reference()[["pair_id", "full_inchikey"]]
        out = scores.merge(identity, on="pair_id", how="left", validate="many_to_one")
        if out.full_inchikey.isna().any():
            raise ValueError("Missing BioLiP pair identity")
        return out.rename(columns={"ranking_id": "method", "fold": "outer_fold"})


class ModelRelease:
    def __init__(self, root):
        self.root = Path(root)
        self.index = pd.read_csv(
            self.root / "model_index.tsv", sep="\t", dtype=str
        ).fillna("")
        self.index["dataset"] = self.index.cohort.replace(COHORT_IDS)
        if self.index.path.duplicated().any():
            raise ValueError("Duplicate checkpoint index path")
        self.hashes = {}
        for line in (self.root / "SHA256SUMS").read_text().splitlines():
            h, p = line.split("  ", 1)
            self.hashes[p] = h

    def deployment(self, dataset, model):
        rows = self.index[
            self.index.experiment.eq("deployment")
            & self.index.dataset.eq(dataset)
            & self.index.model.eq(model)
        ]
        if len(rows) != 3 or rows.seed.nunique() != 3:
            raise ValueError("Expected three deployment seeds for the dataset/model")
        return rows.sort_values("seed").to_dict("records")

    def verified(self, relative):
        p = release_path(self.root, relative)
        if relative not in self.hashes or sha256(p) != self.hashes[relative]:
            raise ValueError("Model release checksum mismatch: " + relative)
        return p

    def load(self, record, device="cpu"):
        import torch

        config = json.loads(self.verified(record["config"]).read_text())
        architecture = config.get("architecture", "model_definitions.py")
        if record["architecture"] != architecture or record["model"] != config["model"]:
            raise ValueError("Checkpoint index/config mismatch")
        src = self.verified(architecture)
        spec = importlib.util.spec_from_file_location(
            "secondsite_weights_" + sha256(src)[:12], src
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.make_model(
            config["model"],
            hidden=config["hidden"],
            heads=config["heads"],
            dropout=config["dropout"],
        )
        path = self.verified(record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError("Checkpoint index checksum mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved[config["state_key"]], strict=True)
        return model.to(device).eval()


def require_shared_rows(frame, columns):
    """Use the same finite-score population for every model in a comparison."""
    scores = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    return frame[np.isfinite(scores).all(axis=1)].copy()
