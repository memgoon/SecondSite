# Murcko scaffold exclusion: limited supplementary experiment

Only every-pair training is scheduled: 8 models x 5 folds x 3 seeds = 120 fits.
The test universe is the same 4,637 pairs as the completed explicit double-held-out
experiment. All old files and manuscript results remain unchanged.

## Frozen design

Start from the original explicit double-held-out train/validation/test partitions.
Remove test Murcko scaffolds from training AND validation. Remove remaining
validation scaffolds from training as well. Never reinstate rows previously purged
for connectivity. No scores are consulted. Protein, Pfam component, connectivity,
and nonempty Murcko scaffolds must be disjoint across every pair of partitions.

Scaffolds are calculated from the frozen canonical SMILES using RDKit Bemis-Murcko,
retaining atom/bond types, without stereochemistry. This is not generic-scaffold
grouping, tautomer normalization, or a molecular-similarity threshold. No exemption
for frequent scaffolds is permitted. Empty scaffolds (acyclic molecules) are not
grouped into a single fictitious scaffold; their connectivity exclusion is retained.
Both all-test and ring-scaffold-only performance are reported; do not describe the
acyclic subset as scaffold-controlled. Ring-only comparisons always use the SAME
rows for the previous and new predictions.

When the frozen input contains alternate charge/protonation representations for
one connectivity, every observed scaffold variant is retained and any shared
variant triggers exclusion. Variants are not selected using labels or scores.

The original eight architectures, embeddings, 25 maximum epochs, patience 5,
optimizer, weighting and seeds 20260817/18/19 are preserved. Checkpoint selection
uses validation only: ligand-only within-protein AUROC; all other models
within-ligand AUROC; fewer than 8 valid groups triggers pooled symmetric AP.
Fallback counts and epoch-1 rates are reported. No test-based epoch selection.
No data-size matched control is scheduled, so a performance change cannot be
attributed solely to scaffold novelty. Reduced training and validation support
is an explicit limitation. Existing whole-family held-out results are NOT reused
as the baseline: the baseline is completed explicit double-held-out retraining.

Frozen feasibility (all test rows retained):

| Fold | Old train | New train | New validation | Test |
|---|---:|---:|---:|---:|
| 0 | 3474 | 2856 | 517 | 1205 |
| 1 | 3899 | 3437 | 389 | 879 |
| 2 | 4123 | 3635 | 447 | 871 |
| 3 | 3914 | 3361 | 531 | 797 |
| 4 | 3383 | 2738 | 1006 | 885 |

63/120 fits use the unchanged low-support checkpoint fallback (protein-only and
six pair models in folds 0/1/2; none of the 15 ligand-only fits). Previously this
was 42/120 for the every-pair double-held-out baseline. Thus the selection RULE
is unchanged, but fewer validation groups can change which criterion it invokes.
The test universe contains 3,676 rows with observed ring scaffolds and 961 without.
These counts use the freshly calculated union of input representations, not the
older protein-anchored descriptor lookup. Thirty exact ligand IDs have multiple
observed scaffold variants; all are retained for conservative exclusion.

## Analysis

Primary sensitivity: change in ligand-only pooled AUROC after scaffold exclusion,
on matched test rows. Also report changes for all models, and joint minus
ligand-only within-protein AUROC. Report AUROC, allosteric AP and symmetric AP
pooled / within-protein / within-ligand / family-macro with valid-row/group counts.
Prespecified all-row and ring-only universes; no outcome-based model selection.
Paired 10,000-replicate family-cluster bootstrap for pooled changes and
within-protein changes/lifts, preserving sampled family multiplicity.
CI is conditional on the three-seed ensemble, not total training uncertainty.
Report individual-seed results separately. AP is descriptive, not bootstrapped.
Intervals are pointwise, not multiple-comparison-adjusted; exploratory contrasts
must not be used to choose an unreported post hoc model winner.

## Local send

```bash
cd /disk9/13.Heesu_Allostery
bash analysis/murcko_holdout_extension/scripts/send_gpu_input.sh
```

Data, split manifests, source code and old comparison predictions are frozen and
sent; embeddings already present on the GPU host are reused. RDKit is needed only
for local contract generation, NOT for GPU training or local result aggregation.
Preflight compares all tensor hashes and architecture parameter counts with the
completed baseline run, and stops if the representation has changed.

## GPU run / resume

```bash
cd /disk1/11.HS_allostery
mkdir -p analysis/murcko_holdout_extension/gpu_output/logs
nohup env GPU_IDS=0,1,2,3 TORCH_NUM_THREADS=4 \
  bash analysis/murcko_holdout_extension/scripts/run_gpu.sh \
  >> analysis/murcko_holdout_extension/gpu_output/logs/run.nohup.log 2>&1 < /dev/null &
```

One worker per GPU; OMP/MKL/BLAS thread counts are bounded. Completed compatible
fits resume with their checksums; partial fits restart. A run lock prevents a
second launcher. Worker logs append, not truncate. Do not change code or inputs
mid-run. Preflight hashes only explicit tensor paths, never a recursive disk scan.
No promised ETA: full-chain bidirectional C3 is comparatively expensive.

```bash
python3 analysis/murcko_holdout_extension/scripts/status.py
tail -F analysis/murcko_holdout_extension/gpu_output/logs/train_worker0.log
```

## Local fetch and CPU bootstrap

```bash
cd /disk9/13.Heesu_Allostery
BOOTSTRAP_WORKERS=32 bash analysis/murcko_holdout_extension/scripts/fetch_gpu_return.sh
```

CPU work runs HERE after retrieval, not on the GPU server. To repeat CPU analysis
without downloading or training: prefix the command with `LOCAL_ONLY=1`.
Outputs: `cpu_output/METRICS.tsv`, `PAIRED_BOOTSTRAP.tsv`,
`ENSEMBLE_PREDICTIONS.tsv.gz`, `TRAINING_STABILITY.tsv`, `VALIDATION.json`.
All fit reports, histories, checkpoints and predictions are included in the return
archive. The local validator rechecks identities, hashes, membership and labels.

Vendored base_trainer.py/model_definitions.py are identical to the explicit
double-held-out package; trainer_core.py differs only in the regime name.
Preparation/static checks do not imply that GPU training has completed.
