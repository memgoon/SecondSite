"""Score external pairs and evaluate reference labels and assay-keyword recovery."""

from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
import torch

from train_models import make_model, predict, file_hash
from evaluate_models import binary_metrics, conditional_metrics


def input_availability(frame, model, pockets, tensor_root):
    available = pd.Series(True, index=frame.index)
    if model != "protein":
        paths = {
            p: (Path(tensor_root) / str(p)).is_file()
            for p in frame.ligand_embedding_path.unique()
            if pd.notna(p)
        }
        available &= frame.ligand_embedding_path.map(paths).fillna(False).astype(bool)
    if model != "ligand":
        paths = {
            p: (Path(tensor_root) / str(p)).is_file()
            for p in frame.protein_embedding_path.unique()
            if pd.notna(p)
        }
        available &= frame.protein_embedding_path.map(paths).fillna(False).astype(bool)
    if model.startswith("d"):
        available &= frame.uniprot.map(lambda u: bool(pockets.get(str(u), [])))
    return available


def novelty(frame, training):
    out = frame.copy()
    pairs = set(zip(training.uniprot, training.full_inchikey))
    out["training_pair"] = list(zip(out.uniprot, out.full_inchikey))
    out["training_pair"] = out.training_pair.map(pairs.__contains__)
    out["protein_unseen"] = ~out.uniprot.isin(training.uniprot)
    out["ligand_unseen"] = ~out.connectivity_key.isin(training.connectivity_key)
    out["double_unseen"] = out.protein_unseen & out.ligand_unseen
    return out


def keyword_enrichment(frame, fraction=0.001):
    """Deterministic row-ID tie breaking; the within-protein set requires a keyword hit."""
    if not 0 < fraction <= 1:
        raise ValueError("Invalid top fraction")
    required = {"record_id", "uniprot", "score", "keyword_positive"}
    if not required.issubset(frame) or frame.record_id.duplicated().any():
        raise ValueError("Keyword table requires unique record IDs and valid columns")
    if not set(frame.keyword_positive.dropna()).issubset({0, 1}):
        raise ValueError("Keyword flags must be 0/1")
    scored = frame[frame.score.notna() & frame.keyword_positive.notna()].copy()
    rows = []
    for scope in ["pooled", "within_protein"]:
        eligible = (
            scored
            if scope == "pooled"
            else scored[scored.groupby("uniprot").keyword_positive.transform("sum") > 0]
        )
        groups = (
            [eligible]
            if scope == "pooled"
            else [g for _, g in eligible.groupby("uniprot")]
        )
        chosen = []
        for group in groups:
            if len(group):
                k = max(1, math.ceil(len(group) * fraction))
                chosen.append(
                    group.sort_values(
                        ["score", "record_id"], ascending=[False, True], kind="stable"
                    ).head(k)
                )
        top = pd.concat(chosen) if chosen else eligible.iloc[:0]
        positives = int(eligible.keyword_positive.sum())
        hits = int(top.keyword_positive.sum())
        enrichment = (
            (hits / len(top)) / (positives / len(eligible))
            if positives and len(top)
            else np.nan
        )
        rows.append(
            dict(
                scope=scope,
                eligible_rows=len(eligible),
                keyword_positive_rows=positives,
                selected_rows=len(top),
                selected_keyword_positive=hits,
                enrichment=enrichment,
                status="evaluated" if positives else "not_evaluated_no_positive_class",
            )
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--training-pairs", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs=3, type=Path, required=True)
    parser.add_argument("--pockets", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Optional table of record_id, published_label and keyword_positive",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-rows", type=int, default=4096)
    args = parser.parse_args()
    training = pd.read_csv(args.training_pairs, sep="\t")
    frame = novelty(pd.read_csv(args.pairs, sep="\t"), training)
    if frame.record_id.duplicated().any():
        raise ValueError("Duplicate external record")
    frame = frame[~frame.training_pair].copy().reset_index(drop=True)
    pockets = json.loads(args.pockets.read_text())
    device = torch.device(args.device)
    reports = [
        json.loads(p.with_name("fit.json").read_text()) for p in args.checkpoints
    ]
    if (
        len({r["seed"] for r in reports}) != 3
        or len({(r["dataset"], r["model"]) for r in reports}) != 1
    ):
        raise ValueError("Supply three seeds of the same deployment model and dataset")
    for path, report in zip(args.checkpoints, reports):
        if (
            report["regime"] != "deployment"
            or file_hash(path) != report["checkpoint_sha256"]
            or report["input_sha256"] != file_hash(args.training_pairs)
        ):
            raise ValueError("Checkpoint or training-reference mismatch")
    name = reports[0]["model"]
    available = input_availability(frame, name, pockets, args.tensor_root)
    frame["score_status"] = np.where(available, "scored", "input_unavailable")
    for path, report in zip(args.checkpoints, reports):
        saved = torch.load(path, map_location=device, weights_only=True)
        model = make_model(name, saved["hidden"], saved["dropout"], saved["heads"]).to(
            device
        )
        model.load_state_dict(saved["model_state_dict"])
        column = f"seed_{report['seed']}"
        frame[column] = np.nan
        indices = frame.index[available]
        for start in range(0, len(indices), args.chunk_rows):
            selected = indices[start : start + args.chunk_rows]
            frame.loc[selected, column] = predict(
                model,
                frame.loc[selected],
                pockets,
                args.tensor_root,
                device,
                name,
                2 if name == "c3" else 8,
                True,
            )
    seed_columns = [f"seed_{r['seed']}" for r in reports]
    frame["score"] = frame[seed_columns].mean(axis=1)
    frame["seed_sd"] = frame[seed_columns].std(axis=1)
    frame["global_rank"] = frame.score.rank(ascending=False, method="average")
    frame["within_protein_rank"] = frame.groupby("uniprot").score.rank(
        ascending=False, method="average"
    )
    # Annotations enter only after scoring; they never reach the model.
    metrics = []
    if args.annotations:
        annotations = pd.read_csv(args.annotations, sep="\t")
        frame = frame.merge(
            annotations, on="record_id", how="left", validate="one_to_one"
        )
        if "published_label" in frame:
            labelled = frame[
                frame.published_label.isin([0, 1]) & frame.score.notna()
            ].rename(columns={"published_label": "binary_label"})
            for scope, subset in [
                ("all", labelled),
                ("protein_unseen", labelled[labelled.protein_unseen]),
                ("ligand_unseen", labelled[labelled.ligand_unseen]),
                ("double_unseen", labelled[labelled.double_unseen]),
            ]:
                metrics.append(
                    dict(
                        scope=scope,
                        rows=len(subset),
                        **binary_metrics(subset),
                        within_protein=conditional_metrics(subset, "uniprot"),
                    )
                )
    args.output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output / "rankings.tsv.gz", sep="\t", index=False, na_rep="NA")
    (args.output / "reference_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    if "keyword_positive" in frame:
        keyword_enrichment(frame).to_csv(
            args.output / "keyword_enrichment.tsv", sep="\t", index=False, na_rep="NA"
        )


if __name__ == "__main__":
    main()
