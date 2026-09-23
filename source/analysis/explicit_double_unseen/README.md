# Explicit double-unseen retraining extension

This is an isolated extension. No old matrix, embedding, checkpoint, ChEMBL result,
or manuscript numerical result is overwritten. GPU execution has not been performed
by the preparation session.

## Scientific question and split

Can models distinguish allosteric versus orthosteric ligands for a held-out protein
family when ALL test ligand connectivities are also excluded from training and validation?

For each existing outer family fold f:

1. Freeze its complete eligible test set, without discarding test rows.
2. Use family fold (f+1) mod 5 as candidate validation data; remove rows whose ligand
   connectivity is in test.
3. Use the other family folds (plus the every-pair broad-only augmentation) as
   candidate training data; remove rows whose connectivity occurs in test OR the
   retained validation set.
4. Save every row as train, validation, test, or excluded for EACH outer fit.
   Check that UniProt, family component and ligand connectivity are pairwise
   disjoint across ALL THREE partitions.

The split is frozen using identities alone, not scores or test labels. Label support
is audited but is not used to change the partition. Test has priority over validation,
and validation over training. This is an explicit purged double-cold split; it does
not claim to partition all rows losslessly into joint connected components.
The partition differs by outer fold, as in cross-validation. All 4,637 evaluation
rows receive exactly one OOF prediction per model/seed. A training pair may be
excluded for one fit and eligible for another; no fit can access its own held-out identities.

## Frozen scope

Two training datasets, every-pair and protein-anchored. Eight original heads
(ligand, protein, C1-C3, D1-D3), five folds, model seeds 20260817/18/19:
**240 newly trained fits**, not another 1,080-fit matrix. Both arms use identical
4,637 test pairs and identical validation pairs. All folds retain both labels.

| Fold | Every-pair train | Protein-anchored train | Validation | Test |
|---|---:|---:|---:|---:|
| 0 | 3474 | 1653 | 688 | 1205 |
| 1 | 3899 | 2032 | 548 | 879 |
| 2 | 4123 | 2195 | 546 | 871 |
| 3 | 3914 | 2064 | 679 | 797 |
| 4 | 3383 | 1581 | 1055 | 885 |

Double-anchored is audited but not scheduled: only 46/19/16/67/91 training pairs
remain. It is mathematically possible after discarding overlaps, but too small
for the requested general-data comparison. No claim of logical impossibility is made.
Role completeness describes the starting dataset; purged training partitions
need not preserve both labels for every individual protein or ligand.

## Training and representation continuity

Original eight-head architectures and cached selected-target-chain/pocket embeddings
are reused. This is NOT the full-canonical-UniProt representation pilot.
No encoder training, new embeddings, docking or ChEMBL screening is scheduled.
25 epochs maximum, patience 5, original optimizer and batching, FP32 sigmoid,
train-only protein/label weighting, and three model seeds remain unchanged.

For a clean comparison to existing family-held-out fits, preserve the original
model-aware **family** checkpoint criterion: ligand-only uses validation
within-protein AUROC, other models validation within-ligand AUROC. Below eight
two-class groups the frozen fallback is pooled symmetric AP. This criterion
need not equal the primary reporting metric. Changing the selection rule here
would introduce another difference from the existing family baseline.
Validation within-ligand support is 8/3/5/12/11 groups; hence 84 of 240 fits use
the prespecified fallback. Validation within-protein support is 46/51/45/57/35.
Report this limitation, seed dispersion and best-epoch distribution; no post hoc
seed, epoch, model, or split selection is allowed.

`trainer_core.py` is a vendored snapshot of the matrix fit loop: only the split
function, contract plumbing, local base-trainer import, the double_unseen
regime alias in checkpoint selection, history-hash resume validation and per-fit
wall-clock timing were changed. Its old main was removed; the unused legacy
argument parser is not the extension entrypoint.
`base_trainer.py` and `model_definitions.py` are unmodified snapshots. Frozen
hashes pin these files; the old package is never edited by this extension.

## Analysis and interpretation

Primary: fold-restricted within-protein AUROC, with joint models versus ligand-only.
Also report pooled, within-ligand and family-macro AUROC/allosteric AP/symmetric AP,
including row/group coverage and individual-seed results.

Use the existing frozen family-held-out predictions on the SAME 4,637 rows for a
paired comparison; original row-random predictions are a descriptive reference.
The old 2,721-row subset is no longer the new experiment's test universe.
This eliminates evaluation-row composition changes in the new versus family-held-out
comparison. It does NOT isolate a pure ligand-novelty causal effect: training size,
training chemistry and validation membership also change. A size-matched retraining
control is not included. Do not describe the prior subset analysis as leaked or
as a separate double-unseen-trained model.

CPU aggregation averages probabilities across the three seeds, then performs
10,000 paired family-cluster bootstrap replicates for each primary model lift
and new-versus-family comparison (28 contrasts). Family draw multiplicities are
preserved by weighting precomputed protein-level AUROCs: A,A,B -> 2/3, not 1/2.
At least 95% of requested replicates must be defined. CIs are conditional on fitted
models and do not integrate all training uncertainty. Per-seed metrics are separate.
CPU bootstrap is run **locally after fetch**, not on the GPU host. No sklearn needed.

## Run and resume

On the local CPU host:

```bash
cd /disk9/13.Heesu_Allostery
bash analysis/explicit_double_unseen/scripts/send_gpu_input.sh
```

On the GPU host:

```bash
cd /disk1/11.HS_allostery
mkdir -p analysis/explicit_double_unseen/gpu_output/logs
nohup env GPU_IDS=0,1 bash analysis/explicit_double_unseen/scripts/run_gpu.sh \
  > analysis/explicit_double_unseen/gpu_output/logs/run.nohup.log 2>&1 < /dev/null &
```

Use a fresh log filename for the top-level nohup log if retaining older console
output. Worker logs are appended, never truncated. One process per listed GPU,
deterministic fit-level cost balancing, and an exclusive run lock prevent duplicate
runs. Progress is written after each completed fit. C3 remains the expensive head.
For a lightweight count without tensor reads or directory traversal:
`python3 analysis/explicit_double_unseen/scripts/status.py`.
Do not promise a short runtime from test-set size; 240 fits still use thousands
of training rows. Estimate ETA after observing completed fits by model and fold.

Preflight reads only explicitly listed cached tensor paths (no recursive search),
hashes them once per invocation, validates real-input forwards for all eight heads,
and fingerprints inputs, code, training settings and numerical library versions.
Valid completed fits resume only with the same fingerprint; incompatible or
incomplete fits are retrained. Do not replace tensors or source files mid-run.
Use Ctrl-C/SIGTERM on the supervising shell for coordinated worker termination.

All 240 reports, histories, checkpoints, predictions and worker logs are returned
in `explicit_double_unseen_gpu_return_v1.tar.gz` with SHA256.

Local retrieval and CPU analysis:

```bash
cd /disk9/13.Heesu_Allostery
BOOTSTRAP_WORKERS=32 bash analysis/explicit_double_unseen/scripts/fetch_gpu_return.sh
```

If CPU analysis was interrupted after extraction, skip download and training:

```bash
LOCAL_ONLY=1 BOOTSTRAP_WORKERS=32 \
  bash analysis/explicit_double_unseen/scripts/fetch_gpu_return.sh
```

Useful local outputs: `data/SPLIT_COUNTS.tsv`, `data/SPLIT_MEMBERSHIP.tsv.gz`,
`data/CHECKPOINT_SUPPORT.tsv`, `validation/CPU_CONTRACT.json`.
After GPU execution: `gpu_output/FIT_SUMMARY.tsv` and `FITS_VALIDATION.json`.
After local analysis: `cpu_output/METRICS.tsv`, `PAIRED_WITHIN_PROTEIN_BOOTSTRAP.tsv`,
`TRAINING_STABILITY.tsv`, `ENSEMBLE_PREDICTIONS.tsv.gz`, `VALIDATION.json`.

Manuscript Results and figures remain based on completed experiments until these
new results have been retrieved and reviewed. The extension is not represented as completed.
