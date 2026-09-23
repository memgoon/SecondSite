# Allosteric versus orthosteric pair benchmark

This package evaluates whether protein-ligand models retain discrimination when
target families and compound identities are not reused from development data.
It contains the graph-free 431-protein target-chain-aligned main benchmark,
exact protein/ligand identity validation, five-fold/three-seed training, OOF
aggregation, and cluster bootstrap analysis.

## Evaluation names used in the manuscript

- **Row-random**: proteins and compounds may recur across folds.
- **Unseen-family**: Pfam-connected protein components are held out.
- **Unseen-family + unseen-compound**: the same unseen-family model is evaluated
  only on compounds whose molecular connectivity was absent from training and
  validation.

The third evaluation is implemented with the first InChIKey block. It is not a
Murcko-scaffold split and does not train another model.

## Main models

`ligand`, `protein`, `c1`, `c2`, `c3`, and `d1` are compared on one common
target-chain cohort. C3 and D1 use the union of the five highest-scoring
fpocket pockets after chain-aware residue mapping; no coordinate graph is
required. The reduced
LABind-style graph model is kept outside the main benchmark because it requires
a much smaller structure-ready intersection and showed no consistent pilot
gain.

## Reproduction

On the CPU server, the transfer command rebuilds the chain-aware pocket masks,
freezes the cohort, validates it, and transfers the compact package:

```bash
bash analysis/allosteric_pair_benchmark_main/scripts/send_gpu_main.sh
```

Run the command printed by the transfer script. The GPU workflow first embeds
the frozen target-chain FASTAs with the local ESM3 Docker image, validates their
lengths against the masks, then resolves exact ligand identities and trains the
models. It is resumable: completed fit reports and prediction universes are
checked before a fit is skipped. After completion, retrieve and postprocess the
results:

For the established project server, the ESM3 wrapper reuses the private token
already configured in `14.Organized_input/10.docker_run.sh` when neither
`ESM3_HF_TOKEN` nor `HF_TOKEN` is present. The token value is not copied into
the package or printed.

```bash
bash analysis/allosteric_pair_benchmark_main/scripts/fetch_gpu_main.sh
```

The fetch script verifies the archive, extracts it, and runs the 10,000-sample
Pfam-component bootstrap on the CPU server. It then renders the draft main
performance figure as PDF and PNG under `figures/`.

## Output

Primary result tables are written under
`gpu_output/main_benchmark/aggregate/`:

- `MODEL_SUMMARY.tsv`
- `OOF_METRICS_BY_SEED.tsv`
- `PAIRED_DELTAS_VS_C1.tsv`
- `CLUSTER_BOOTSTRAP_METRICS.tsv`
- `CLUSTER_BOOTSTRAP_DELTAS_VS_C1.tsv`
- `SPLIT_PENALTY_FAMILY_BOOTSTRAP.tsv`
- `SPLIT_PENALTY_OBSERVED_BY_SEED.tsv`

The Figure 2(c) split-penalty intervals can be regenerated on the CPU server
with 12 worker processes:

```bash
python3 analysis/allosteric_pair_benchmark_main/scripts/bootstrap_split_penalty.py \
  --replicates 10000 --workers 12
```

This uses paired family-component resampling separately for all 4,637 test
pairs and for the 2,466 pairs whose ligand connectivity is absent from training
under both split regimes.

See `methods/MATERIALS_AND_METHODS.md` for manuscript-ready methods and
`EXPERIMENT_CONTRACT.md` for frozen interpretation constraints.

## Publication note

The code is organized for a public repository, but redistribution rights for
database-derived rows must be checked before publishing `data/MAIN_COHORT.tsv.gz`.
The release checklist describes a manifest-only fallback for restricted fields.
