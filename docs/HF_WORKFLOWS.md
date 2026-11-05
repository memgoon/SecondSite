# Using the published data and models

Install with `python -m pip install -e .` from the repository root. Commands below
use data downloaded to `artifacts/data` and models downloaded to `artifacts/models`.
`python -m secondsite download --destination artifacts --data-profile all --models-profile all`
downloads both pinned releases. Each download accepts a revision override.

## Benchmark

Use the frozen cohort tables directly:

| Dataset | Data path | Pocket mapping |
|---|---|---|
| Every-pair | `benchmark/every_pair.tsv.gz` | `inputs/pockets/main_benchmark.json` |
| Both-class proteins | `benchmark/both_class_proteins.tsv.gz` | `inputs/pockets/main_benchmark.json` |
| Both-class ligands | `benchmark/both_class_ligands.tsv.gz` | `inputs/pockets/both_class_ligands.json` |
| Both-class proteins & ligands | `benchmark/both_class_proteins_ligands.tsv.gz` | `inputs/pockets/both_class_proteins_ligands.json` |

All paths in this table are relative to `artifacts/data`. Embedding paths in the
pair tables use that same root. Dataset-specific fold columns are retained.

Evaluate the main five-fold predictions:

```bash
python -m secondsite.evaluate_models --predictions artifacts/data/benchmark/predictions/main_five_fold_all_seeds.tsv.gz --output results/main_benchmark
```

To select a dataset and setting, add, for example,
`--dataset both_class_proteins --regime pfam_held_out`. To calculate paired
intervals, add `--bootstrap-reference ligand --replicates 10000`.
For an already averaged prediction table, add `--ensemble`, for example:

```bash
python -m secondsite.evaluate_models --predictions artifacts/data/benchmark/predictions/murcko_ensemble.tsv.gz --ensemble --output results/murcko
python -m secondsite.evaluate_models --predictions artifacts/data/time_split/predictions/curated/every_pair_c2.tsv.gz --ensemble --output results/time_split_c2
```

The reader uses `cohort_arm` as the dataset identifier and `p_allosteric` as the
score when reading the release tables. Dataset, regime, model and row ID remain
separate grouping keys.

The scaffold experiment can be fitted from the fixed tensors and membership
table, without reconstructing molecular scaffolds:

```bash
python -m secondsite.train_models --pairs artifacts/data/benchmark/every_pair.tsv.gz --pockets artifacts/data/inputs/pockets/main_benchmark.json --tensor-root artifacts/data --dataset every_pair --regime double_held_out --membership artifacts/data/benchmark/splits/murcko_holdout_extension/membership.tsv --model c2 --fold 0 --seed 20260817 --device cuda:0 --output results/murcko_fit
```

## Released model weights

`model_index.tsv` selects the checkpoint, seed, configuration and architecture.
The loader verifies the indexed files and loads the recorded architecture with
an exact state-dictionary match. `check-release` runs one input through each of
the 32 deployment model/dataset combinations using the first released seed:

```bash
python -m secondsite check-release --data-root artifacts/data --models-root artifacts/models
```

To load any indexed checkpoint in Python:

```python
from secondsite.release_io import ModelRelease
release = ModelRelease("artifacts/models")
record = release.deployment("both_class_proteins", "c2")[0]
model = release.load(record, device="cpu")
```

For new pairs, use `python -m secondsite predict` with the identifiers, embedding
paths and pocket mapping described in [INPUTS.md](INPUTS.md). This command
selects all three deployment seeds, excludes training pairs and writes scores
and both global and within-protein ranks.

## ChEMBL

The released ChEMBL pair and ranking tables can be evaluated directly:

```bash
bash scripts/evaluate_chembl.sh artifacts/data results/chembl
```

The command joins reference records on `reference_row_id`, selects the common
sequence-and-pocket-available population and reports pooled and protein-macro
metrics for all four training datasets and eight models. Full-screen rankings
join on `OOD_Row_ID`. Existing ranking tables can be loaded in Python with
`DataRelease("artifacts/data").chembl_scores("every_pair", "full")`.

## BioLiP

```bash
bash scripts/rank_biolip.sh artifacts/data results/biolip
```

This command starts from `biolip/evaluation/REFERENCE_PAIRS.tsv` and
`biolip/rankings/PAIR_RECORDS.tsv`, which already contain the pair-level D, MW,
LP and AR features. It fits 46 rankings and evaluates the fixed protein and
family folds. Structural observations are not aggregated a second time.
`DataRelease.biolip_scores()` joins the released held-out scores to their
chemical identities for downstream ranking comparisons.

## Docking

Convert the released archive into the docking runner's job table:

```bash
python -m secondsite docking-jobs --archive artifacts/data/docking/archives/expansion/every_pair_c2.tar.gz --output results/docking_inputs
python -m secondsite.structural_followup --jobs results/docking_inputs/jobs.tsv --vina /path/to/vina --output results/docking_rerun
```

The first command extracts the archive's prepared receptors, ligands and poses,
and converts its `jobs.json` box vectors and identifiers into the runner's TSV
columns. The existing scores and classifications remain available under
`docking/initial/`, `docking/expansion/` and `docking/known_sites/`.

## Molecular dynamics

The preparation, steering and production commands are in
[workflows/md/README.md](../workflows/md/README.md). Install the simulation engines
separately, then use the pinned reference inputs and the H1–H5 scripts there.
