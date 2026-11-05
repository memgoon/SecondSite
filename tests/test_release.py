import importlib
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from secondsite.release_io import (
    DATASETS,
    MODELS,
    DataRelease,
    ModelRelease,
    frozen_partition,
    prediction_frame,
    release_path,
    require_shared_rows,
)
from secondsite.evaluate_models import ensemble_predictions


class SchemaTests(unittest.TestCase):
    def test_released_scores(self):
        f = pd.DataFrame(
            dict(
                cohort_arm=["every_pair"] * 3,
                p_allosteric=[0.1, 0.2, 0.3],
                model=["c2"] * 3,
                main_row_id=["p"] * 3,
                seed=[1, 2, 3],
            )
        )
        r = ensemble_predictions(f)
        self.assertAlmostEqual(r.score.iloc[0], 0.2)
        self.assertEqual(r.dataset.iloc[0], "every_pair")

    def test_cohorts_stay_separate(self):
        a = pd.DataFrame(
            dict(
                cohort_arm=["every_pair"] * 3,
                p_allosteric=[0.1, 0.2, 0.3],
                model=["c2"] * 3,
                main_row_id=["p"] * 3,
                seed=[1, 2, 3],
            )
        )
        b = a.copy()
        b["cohort_arm"] = "both_class_proteins"
        self.assertEqual(len(ensemble_predictions(pd.concat([a, b]))), 2)

    def test_conflicting_alias_rejected(self):
        with self.assertRaises(ValueError):
            prediction_frame(pd.DataFrame({"score": [0.1], "p_allosteric": [0.2]}))

    def test_duplicate_seed_rejected(self):
        f = pd.DataFrame(
            dict(
                score=[0.1, 0.2, 0.3],
                model=["c2"] * 3,
                main_row_id=["p"] * 3,
                seed=[1, 1, 2],
            )
        )
        with self.assertRaises(ValueError):
            ensemble_predictions(f)

    def test_shared_available_rows(self):
        f = pd.DataFrame({"c2": [0.1, 0.2, np.nan], "c3": [0.2, np.nan, 0.4]})
        self.assertEqual(list(require_shared_rows(f, ["c2", "c3"]).index), [0])

    def test_single_input_numerical_aggregation(self):
        rows = []
        for fold, value in [(0, 0.2), (1, 0.8)]:
            for pair, delta in [("a", 0), ("b", 0.00003)]:
                for seed in [1, 2, 3]:
                    rows.append(
                        dict(
                            model="protein",
                            uniprot="P",
                            outer_fold=fold,
                            main_row_id=f"{fold}{pair}",
                            seed=seed,
                            score=value + delta,
                        )
                    )
        f = pd.DataFrame(rows)
        r = ensemble_predictions(f)
        self.assertEqual(r.groupby("outer_fold").score.nunique().tolist(), [1, 1])
        self.assertAlmostEqual(r[r.outer_fold.eq(0)].score.iloc[0], 0.200015)
        f.loc[f.main_row_id.eq("0b"), "score"] = 0.4
        with self.assertRaises(ValueError):
            ensemble_predictions(f)

    def test_release_path(self):
        with self.assertRaises(ValueError):
            release_path("/tmp", "../outside")
        with self.assertRaises(ValueError):
            release_path("/tmp", "/outside")

    def test_fixed_membership(self):
        f = pd.DataFrame(
            dict(
                main_row_id=["a", "b", "c", "d"],
                uniprot=["A", "B", "C", "D"],
                family_component_id=["FA", "FB", "FC", "FD"],
                connectivity_key=["LA", "LB", "LC", "LD"],
            )
        )
        m = pd.DataFrame(
            dict(
                main_row_id=["a", "b", "c", "d"],
                cohort_arm=["every_pair"] * 4,
                outer_fold=[0] * 4,
                partition=["train", "validation", "test", "excluded"],
            )
        )
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "membership.tsv"
            m.to_csv(p, sep="\t", index=False)
            parts = frozen_partition(f, p, "every_pair", "double_held_out", 0)
            self.assertEqual(
                [list(x.main_row_id) for x in parts], [["a"], ["b"], ["c"]]
            )
            f.loc[1, "connectivity_key"] = "LA"
            with self.assertRaises(ValueError):
                frozen_partition(f, p, "every_pair", "double_held_out", 0)

    def test_modules_import(self):
        for name in [
            "attention_analysis",
            "build_datasets",
            "compare_rankings",
            "docking_sites",
            "evaluate_models",
            "prepare_references",
            "rank_biolip",
            "ranking_recovery",
            "screen_chembl",
            "structural_followup",
            "time_split",
            "train_models",
            "release_cli",
        ]:
            importlib.import_module("secondsite." + name)


@unittest.skipUnless(
    os.environ.get("SECONDSITE_DATA_ROOT"),
    "Set SECONDSITE_DATA_ROOT for release checks",
)
class DataIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = DataRelease(os.environ["SECONDSITE_DATA_ROOT"])

    def test_cohort_counts_and_folds(self):
        for name, n in zip(DATASETS, [6854, 4637, 1365, 395]):
            self.assertEqual(len(self.data.cohort(name)), n)
        self.assertIn("run_family_fold", self.data.cohort("both_class_ligands"))

    def test_frozen_murcko_folds(self):
        f = self.data.cohort("every_pair")
        path = self.data.path(
            "benchmark/splits/murcko_holdout_extension/membership.tsv"
        )
        total = 0
        for fold in range(5):
            a, b, c = frozen_partition(f, path, "every_pair", "double_held_out", fold)
            self.assertGreater(len(a), 0)
            self.assertGreater(len(b), 0)
            total += len(c)
        self.assertEqual(total, 4637)

    def test_chembl_matched_population(self):
        for name in DATASETS:
            f = self.data.chembl_scores(name)
            f = f[
                f.weak2020_label.isin([0, 1])
                & f.selected_chain_available.eq(1)
                & f.pocket_available.eq(1)
            ]
            self.assertEqual(len(require_shared_rows(f, MODELS)), 10628)

    def test_biolip_join(self):
        f = self.data.biolip_scores()
        self.assertEqual(len(f), 121624)
        self.assertEqual(f.method.nunique(), 46)
        self.assertFalse(f.full_inchikey.isna().any())

    def test_tensor_indexing(self):
        from secondsite.train_models import PairDataset

        f = self.data.cohort("every_pair").head(1)
        item = PairDataset(f, self.data.pockets("every_pair"), self.data.root, "d2")[0]
        self.assertEqual(item["ligand"].shape[1], 512)
        self.assertEqual(item["pocket"].shape[1], 1536)
        self.assertGreater(len(item["pocket"]), 0)


@unittest.skipUnless(
    os.environ.get("SECONDSITE_MODELS_ROOT"),
    "Set SECONDSITE_MODELS_ROOT for weight checks",
)
class ModelIntegrationTests(unittest.TestCase):
    def test_deployment_loading(self):
        release = ModelRelease(os.environ["SECONDSITE_MODELS_ROOT"])
        for cohort in DATASETS:
            for model in MODELS:
                rows = release.deployment(cohort, model)
                self.assertEqual(len(rows), 3)
                self.assertIsNotNone(release.load(rows[0]))


if __name__ == "__main__":
    unittest.main()
