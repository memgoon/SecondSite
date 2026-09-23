#!/usr/bin/env python3
"""Small dependency-light checks for AP and weighted cluster-bootstrap code."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / "scripts" / "{}.py".format(name)
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    aggregate = load("aggregate_main_results")
    bootstrap = load("bootstrap_oof")
    frame = pd.DataFrame({
        "binary_label": [1, 0, 1, 0],
        "p_allosteric": [0.9, 0.8, 0.7, 0.1],
        "uniprot": ["A", "A", "B", "B"],
        "family_component_id": ["X", "X", "Y", "Y"],
    })
    direct = aggregate.binary_metrics(frame)["allosteric_positive_ap"]
    weighted = bootstrap.observed_ap(
        frame["binary_label"].to_numpy(), frame["p_allosteric"].to_numpy()
    )
    if not np.isclose(direct, 5.0 / 6.0) or not np.isclose(direct, weighted):
        raise AssertionError("AP implementations disagree")
    counts = np.asarray([[1, 1], [2, 0], [0, 2]], dtype=int)
    values = bootstrap.weighted_ap(
        frame["binary_label"].to_numpy(), frame["p_allosteric"].to_numpy(),
        np.asarray([0, 0, 1, 1]), counts,
    )
    if not np.allclose(values, [5.0 / 6.0, 1.0, 1.0]):
        raise AssertionError("cluster weights are not reproduced")
    print("metric consistency checks passed")


if __name__ == "__main__":
    main()
