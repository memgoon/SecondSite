#!/usr/bin/env python3
"""Offline tests for C3 masking and matched-subset aggregation."""

from __future__ import annotations

import importlib.util
import argparse
import ast
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inference_call_contract(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    main = functions["main"]
    for name in ("infer_reference", "infer_full"):
        definition = functions[name]
        parameters = [argument.arg for argument in definition.args.args]
        assert "trainer" in parameters
        calls = [
            node for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ]
        assert len(calls) == 1
        assert len(calls[0].args) == len(parameters)
        supplied = [
            node.id if isinstance(node, ast.Name) else None for node in calls[0].args
        ]
        assert supplied[parameters.index("trainer")] == "trainer"


def test_inference_partition(infer):
    import torch

    class DummyModel(torch.nn.Module):
        def __init__(self, use_pocket):
            super().__init__()
            self.use_pocket = use_pocket

        def forward(self, batch):
            source = batch["pocket"] if self.use_pocket else batch["ligand"]
            return source.mean(dim=(1, 2))

    batch = {
        "ligand": torch.ones(3, 2, 4),
        "ligand_mask": torch.ones(3, 2, dtype=torch.bool),
        "protein": torch.ones(3, 2, 1536),
        "protein_mask": torch.ones(3, 2, dtype=torch.bool),
        "pocket": torch.ones(3, 1, 1536),
        "pocket_mask": torch.ones(3, 1, dtype=torch.bool),
    }
    models = {
        "property_ligand": [DummyModel(False)],
        "property_c2": [DummyModel(False)],
        "property_c3": [DummyModel(True)],
    }
    values, positions = infer.score_base_and_c3(
        models, batch, np.asarray([True, False, True]), torch.device("cpu")
    )
    assert positions.tolist() == [0, 2]
    assert len(values["property_ligand_mean"]) == 3
    assert len(values["property_c2_mean"]) == 3
    assert len(values["property_c3_mean"]) == 2

    frame = pd.DataFrame({
        "c3_pocket_available": [1, 0, 1],
        "p_property_ligand_mean": [0.2, 0.3, 0.4],
        "p_property_c2_mean": [0.3, 0.4, 0.5],
        "p_property_c3_mean": [0.4, np.nan, 0.6],
        "p_property_c3_sd": [0.01, np.nan, 0.02],
    })
    assert infer.validate_probability_output(
        frame, ["property_ligand", "property_c2", "property_c3"]
    )
    frame.loc[1, "p_property_c3_mean"] = 0.5
    assert not infer.validate_probability_output(
        frame, ["property_ligand", "property_c2", "property_c3"]
    )


def test_matched_aggregation(aggregate):
    models = [
        "source_ligand", "source_protein", "source_c2",
        "property_ligand", "property_c2", "property_c3",
    ]
    rows = []
    # U1 and U2 have both labels and pockets. U3 has both labels but no pocket.
    for target_index, target in enumerate(["U1", "U2", "U3"]):
        for label in [0, 1]:
            row = {
                "uniprot": target,
                "weak2020_label": label,
                "is_5a_positive": label,
                "target_seen_property": int(target == "U1"),
                "compound_seen_property": 0,
                "double_novel_target_identity_and_connectivity": int(target != "U1"),
                "pfam_family_overlap_status": (
                    "seen" if target == "U1" else
                    "unseen" if target == "U2" else "annotation_unavailable"
                ),
                "c3_pocket_available": int(target != "U3"),
                "c3_pocket_unavailable_reason": (
                    "available" if target != "U3" else "fpocket_mapping_missing"
                ),
            }
            for model_index, model in enumerate(models):
                score = 0.2 + 0.5 * label + 0.01 * model_index + 0.001 * target_index
                row["p_{}_mean".format(model)] = (
                    score if model != "property_c3" or target != "U3" else np.nan
                )
            rows.append(row)
    with tempfile.TemporaryDirectory() as temporary:
        package = Path(temporary) / "package"
        reference = package / "gpu_output/reference/reference_predictions.tsv.gz"
        output = package / "gpu_output/aggregate"
        reference.parent.mkdir(parents=True)
        output.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(reference, sep="\t", index=False)
        _, _, metric_rows, bootstrap_rows, _ = aggregate.reference_analysis(
            package, output, models, 100
        )
        c3 = [row for row in metric_rows if row["model"] == "property_c3"]
        assert len(c3) == 1
        assert c3[0]["evaluation_subset"] == "pocket_available_matched"
        assert c3[0]["n"] == 4
        c2_subsets = {
            row["evaluation_subset"]
            for row in metric_rows if row["model"] == "property_c2"
        }
        assert c2_subsets == {"all_rows", "pocket_available_matched"}
        assert any(
            row["test_model"] == "property_c3"
            and row["evaluation_subset"] == "pocket_available_matched"
            for row in bootstrap_rows
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregation-only", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    test_inference_call_contract(here / "infer_chembl.py")
    aggregate = load_module("c3_aggregate", here / "aggregate_results.py")
    test_matched_aggregation(aggregate)
    if not args.aggregation_only:
        infer = load_module("c3_infer", here / "infer_chembl.py")
        test_inference_partition(infer)
    print(
        "PASS: C3 matched-subset aggregation{}".format(
            "" if args.aggregation_only else " and unavailable-score masking"
        )
    )


if __name__ == "__main__":
    main()
