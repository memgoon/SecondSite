"""Run SecondSite workflows using the published HF packages."""

import argparse
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from .release_io import (
    DATASETS,
    MODELS,
    DATA_REVISION,
    MODEL_REVISION,
    DataRelease,
    ModelRelease,
    require_shared_rows,
)


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_table(frame, path):
    frame.to_csv(path, sep="\t", index=False, na_rep="NA")


def download(args):
    from huggingface_hub import snapshot_download

    common = [
        "README.md",
        "LICENSE",
        "*MANIFEST*",
        "FILE_LICENSES.tsv",
        "DATA_DICTIONARY.tsv",
        "DATA_GUIDE.md",
        "THIRD_PARTY_NOTICES.txt",
        "SHA256SUMS",
    ]
    profiles = {
        "benchmark": common + ["benchmark/*", "inputs/*"],
        "results": common
        + ["chembl/*", "biolip/*", "attention/*", "docking/*", "time_split/*"],
        "all": None,
    }
    if args.data_profile != "none":
        snapshot_download(
            "memgoon/SecondSite_data",
            repo_type="dataset",
            revision=args.data_revision,
            local_dir=str(args.destination / "data"),
            allow_patterns=profiles[args.data_profile],
        )
    if args.models_profile != "none":
        patterns = (
            None
            if args.models_profile == "all"
            else [
                "deployment/*",
                "model_index.tsv",
                "SHA256SUMS",
                "model_definitions.py",
                "architectures/*",
                "README.md",
                "LICENSE",
                "load_model.py",
                "requirements.txt",
            ]
        )
        snapshot_download(
            "memgoon/SecondSite-models",
            revision=args.model_revision,
            local_dir=str(args.destination / "models"),
            allow_patterns=patterns,
        )
    print(
        json.dumps(
            dict(
                data_revision=args.data_revision,
                model_revision=args.model_revision,
                destination=str(args.destination),
            ),
            indent=2,
        )
    )


def check_release(args):
    data = DataRelease(args.data_root)
    report = {}
    for name, n in zip(DATASETS, (6854, 4637, 1365, 395)):
        data.path(f"benchmark/{name}.tsv.gz", verify=True)
        f = data.cohort(name)
        if len(f) != n or not f.binary_label.isin([0, 1]).all():
            raise ValueError("Unexpected cohort content: " + name)
        for c in ("protein_embedding_path", "ligand_embedding_path"):
            for relative in f[c].unique():
                data.path(relative)
        report[name] = dict(
            pairs=len(f),
            proteins=f.uniprot.nunique(),
            ligands=f.full_inchikey.nunique(),
        )
    if args.models_root:
        import torch
        from .train_models import PairDataset, collate_pairs

        torch.set_num_threads(args.cpu_threads)
        release = ModelRelease(args.models_root)
        checks = []
        for name in DATASETS:
            f = data.cohort(name).head(1)
            pockets = data.pockets(name)
            for kind in MODELS:
                record = release.deployment(name, kind)[0]
                model = release.load(record)
                batch = collate_pairs([PairDataset(f, pockets, data.root, kind)[0]])
                with torch.no_grad():
                    value = model(batch)
                if value.shape != (1,) or not torch.isfinite(value).all():
                    raise ValueError("Invalid model output")
                checks.append(dict(dataset=name, model=kind, seed=record["seed"]))
        report["checkpoint_forward_checks"] = checks
    print(json.dumps(report, indent=2))


def predict_pairs(args):
    import torch
    from .train_models import predict
    from .screen_chembl import novelty, input_availability

    data = DataRelease(args.data_root)
    models = ModelRelease(args.models_root)
    frame = pd.read_csv(args.pairs, sep="\t")
    required = {
        "record_id",
        "uniprot",
        "full_inchikey",
        "connectivity_key",
        "protein_embedding_path",
        "ligand_embedding_path",
    }
    if not required <= set(frame) or frame.record_id.duplicated().any():
        raise ValueError(
            "Pairs need unique record_id, identities and embedding paths; see docs/INPUTS.md"
        )
    frame = novelty(frame, data.cohort(args.dataset))
    frame = frame[~frame.training_pair].copy().reset_index(drop=True)
    pockets = (
        json.loads(args.pockets.read_text())
        if args.pockets
        else data.pockets(args.dataset)
    )
    tensor_root = args.tensor_root or args.data_root
    device = torch.device(args.device)
    available = input_availability(frame, args.model, pockets, tensor_root)
    frame["score_status"] = np.where(available, "scored", "input_unavailable")
    columns = []
    for record in models.deployment(args.dataset, args.model):
        net = models.load(record, device)
        col = "seed_" + record["seed"]
        columns.append(col)
        frame[col] = np.nan
        if available.any():
            frame.loc[available, col] = predict(
                net,
                frame[available],
                pockets,
                tensor_root,
                device,
                args.model,
                args.batch_size,
                constant_single_input=True,
            )
    frame["score"] = frame[columns].mean(axis=1)
    frame["seed_sd"] = frame[columns].std(axis=1, ddof=0)
    frame["global_rank"] = frame.score.rank(ascending=False, method="average")
    frame["within_protein_rank"] = frame.groupby("uniprot").score.rank(
        ascending=False, method="average"
    )
    frame["dataset"] = args.dataset
    frame["model"] = args.model
    out = new_output(args.output)
    write_table(frame, out / "rankings.tsv.gz")


def chembl_metrics(args):
    from .evaluate_models import binary_metrics

    data = DataRelease(args.data_root)
    rows = []
    for cohort in DATASETS:
        f = data.chembl_scores(cohort, "reference")
        f = f[
            f.weak2020_label.isin([0, 1])
            & f.selected_chain_available.eq(1)
            & f.pocket_available.eq(1)
        ]
        f = require_shared_rows(f, MODELS).rename(
            columns={"weak2020_label": "binary_label"}
        )
        for model in MODELS:
            values = []
            eligible = 0
            for _, g in f.groupby("uniprot"):
                if g.binary_label.nunique() == 2:
                    values.append(binary_metrics(g, model)["auroc"])
                    eligible += len(g)
            rows.append(
                dict(
                    dataset=cohort,
                    model=model,
                    rows=len(f),
                    **binary_metrics(f, model),
                    within_protein_auroc=float(np.mean(values)),
                    two_class_proteins=len(values),
                    within_protein_rows=eligible,
                )
            )
    out = new_output(args.output)
    write_table(pd.DataFrame(rows), out / "reference_metrics.tsv")


def biolip_rank(args):
    from .rank_biolip import rank_candidates, held_out_scores

    data = DataRelease(args.data_root)
    ref = data.biolip_reference()
    candidates = data.table("biolip/rankings/PAIR_RECORDS.tsv")
    rankings = rank_candidates(ref, candidates)
    out = new_output(args.output)
    write_table(rankings, out / "candidate_rankings.tsv.gz")
    if args.evaluate:
        scores = pd.concat(
            [held_out_scores(ref, c) for c in ("protein_fold", "family_fold")],
            ignore_index=True,
        )
        write_table(scores, out / "reference_scores.tsv.gz")


def docking_jobs(args):
    """Export an archived job list to the docking runner's tabular schema."""
    # Validate the complete member list before creating or extracting anything.
    with tarfile.open(args.archive, "r:gz") as tar:
        members = tar.getmembers()
        names = set()
        for m in members:
            p = PurePosixPath(m.name)
            if p.is_absolute() or ".." in p.parts or not m.isfile() or m.name in names:
                raise ValueError("Unsafe/duplicate archive member: " + m.name)
            if m.name != "jobs.json" and not m.name.startswith(
                ("inputs/receptor/", "inputs/ligand/", "poses/")
            ):
                raise ValueError("Unexpected archive member: " + m.name)
            names.add(m.name)
        jobs = json.load(tar.extractfile("jobs.json"))
        records = []
        for job in jobs:
            for k in ("receptor", "ligand", "pose"):
                if job[k] not in names:
                    raise ValueError("Missing archived input: " + job[k])
            if len(job["center"]) != 3 or len(job["size"]) != 3:
                raise ValueError("Invalid box")
            row = {
                k: job[k]
                for k in [
                    "job_id",
                    "system_id",
                    "pocket_id",
                    "scoring",
                    "seed",
                    "exhaustiveness",
                    "cpu",
                    "num_modes",
                ]
            }
            row.update(
                ligand_id=job["chembl_id"],
                ligand_role=job["role"],
                uniprot=job["uniprot"],
                match_id=job["match_id"],
            )
            for k in ("receptor", "ligand"):
                row[k] = str((args.output / "archive" / job[k]).resolve())
            for k in ("center", "size"):
                row.update({k + "_" + axis: v for axis, v in zip("xyz", job[k])})
            records.append(row)
        frame = pd.DataFrame(records)
        if frame.job_id.duplicated().any():
            raise ValueError("Duplicate job ID")
        out = new_output(args.output)
        for m in members:
            path = out / "archive" / m.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as source, path.open("wb") as dest:
                shutil.copyfileobj(source, dest)
    write_table(frame, out / "jobs.tsv")
    print(
        json.dumps(
            dict(jobs=len(frame), systems=frame.system_id.nunique(), output=str(out)),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download", help="Download pinned model/data releases")
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument(
        "--data-profile",
        choices=["all", "benchmark", "results", "none"],
        default="benchmark",
    )
    p.add_argument(
        "--models-profile", choices=["deployment", "all", "none"], default="deployment"
    )
    p.add_argument("--data-revision", default=DATA_REVISION)
    p.add_argument("--model-revision", default=MODEL_REVISION)
    p.set_defaults(run=download)
    p = sub.add_parser(
        "check-release",
        help="Check benchmark tensors and released model forward passes",
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--models-root", type=Path)
    p.add_argument("--cpu-threads", type=int, default=1)
    p.set_defaults(run=check_release)
    p = sub.add_parser(
        "predict",
        help="Rank new embedding-backed pairs with the released deployment models",
    )
    for x in ["data-root", "models-root", "pairs", "output"]:
        p.add_argument("--" + x, type=Path, required=True)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--tensor-root", type=Path)
    p.add_argument("--pockets", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.set_defaults(run=predict_pairs)
    p = sub.add_parser(
        "chembl-metrics",
        help="Evaluate all four released ChEMBL reference rankings on matched rows",
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(run=chembl_metrics)
    p = sub.add_parser(
        "biolip-rank",
        help="Fit the structural rankings from released pair-level features",
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--evaluate", action="store_true")
    p.set_defaults(run=biolip_rank)
    p = sub.add_parser(
        "docking-jobs", help="Prepare runnable jobs from a released docking archive"
    )
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(run=docking_jobs)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
