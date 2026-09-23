# Double-anchored training: matched family-only and double-unseen tests

## Question and scope

All head training starts from the original **395 double-anchored pairs**, not from
the every-pair or protein-anchored cohorts. The source contains 98 proteins, 40
full InChIKeys and 47 family components (143 allosteric / 252 orthosteric pairs).
Two conditions share precisely the same test rows:

1. `family_only`: leave one family out; train on the remaining families.
2. `double_unseen`: leave that family out and additionally exclude every training
   row whose ligand connectivity key occurs in the test family.

There are 47 test folds. Each source row is tested exactly once in each condition.
This is **not a five-way joint connected-component split**. Cross-boundary rows
are discarded separately for each test family. Connectivity-level exclusion also
prevents stereoisomer variants with the same connectivity from leaking into train.
Family and ligand overlap in the double-unseen condition must both be zero.

The surviving training set is derived exclusively from the double-anchored source,
but is **not guaranteed to remain two-label balanced for every protein and ligand**.
We do not reapply a fixed-point filter after purging and do not add ligand-ratio
weights. The original train-only protein/label weighting is retained. Double-
anchoring here describes the source cohort, not a claim of zero identity signal
in every purged training set.

## Fixed experiment

- 2 conditions × 47 families × 8 heads × 3 seeds = **2,256 fits**.
- Seeds: 20260817, 20260818, 20260819.
- Heads: ligand-only, protein-only, C1/C2/C3, D1/D2/D3.
- C = whole **selected target chain**, D = pocket residues. No new canonical
  UniProt embeddings are generated. 1 = concatenation, 2 = one-way attention,
  3 = bidirectional attention. Existing cached representations are reused.
- **25 epochs for every fit; final epoch only; no validation or early stopping.**
  25 is the previous maximum training budget, fixed before this run. It is not
  estimated from held-out results. No test predictions are computed during training.
- Existing hidden width 256, 4 heads, dropout 0.30, AdamW lr 1e-4 and decay 1e-4,
  gradient norm clipping 5. Batch 6 (C3: 1), prediction batch 8 (C3: 2).
- No external training pairs and no reused trained head weights.
- FP32 sigmoid after AMP logits. Single-input heads are evaluated once per actual
  identical input (protein accession or ligand embedding file), with batch size 1,
  then broadcast. This removes padding-induced numerical variation, not biological
  signal; it is not score rounding.
- Maximum cached ligand atoms 120 and protein length 4096 match earlier heads;
  this package refuses protein inputs exceeding the latter limit.

This completes a **double-anchored-derived train / explicit double-unseen test**
experiment, but does not make a perfectly protocol-identical matrix with earlier
five-fold validation-selected experiments. For a fair split comparison here, use
the new matched family-only condition, not the older five-fold result.
The difference also includes loss of training rows and optimizer updates from
purging; it does not isolate memorization as a causal effect. No size-matched
training control is included. No result-dependent epoch/seed/head selection.

## Evaluation, fixed before training

- Primary: mean within-protein AUROC, 395 rows / 98 proteins / 47 family clusters.
  Every protein belongs to just one test family, so scores from different fitted
  folds are never compared within a protein.
- Report pooled, family-macro, allosteric AP and symmetric AP as secondary metrics.
- Within-ligand AUROC is fold-restricted to the full InChIKey: only **42 rows / 14
  valid groups**, so it is descriptive, not a headline generalization estimate.
- Report all 3 seeds and their mean-score ensemble. No best-seed selection.
- Primary bootstrap: 10,000 paired family-cluster resamples, preserving family
  occurrence counts (A,A,B counts A twice). 36 prespecified comparisons:
  16 model/condition AUROC-minus-0.5 estimates, 12 joint-minus-ligand contrasts,
  and 8 double-unseen-minus-family-only contrasts. Add 0.5 to the first CI to obtain
  the absolute AUROC CI. Minimum valid fraction 95%; failure does not pass silently.
- CIs describe held-out family sampling conditional on the fitted ensemble; they
  do not fully account for training-set overlap or training-seed uncertainty.
  `bootstrap_fraction_gt_zero` is not a Bayesian posterior probability or a
  multiplicity-adjusted hypothesis test. Report the full model table.
- Protein-only within protein and ligand-only within exact ligand must be 0.500
  as implementation controls. The primary substantive comparator is ligand-only.
- Per-ligand scores and ATP/ADP row counts; exclude ATP, ADP and both as separate
  evaluation-only sensitivities. No claim of new training from these exclusions.
- All metric tables include valid row/group coverage and skipped-group counts.

## Runtime and reproducibility

GPU server performs preflight, training, fit validation and packaging. **It does
not perform bootstrap.** Local fetch performs independent hash/metadata validation,
aggregation and CPU bootstrap. CPU analysis needs numpy and pandas, not sklearn.

Default: four GPUs, six PyTorch/BLAS CPU threads per GPU. In-memory embedding caches
are loaded once per worker. A transactional dynamic queue gives available workers
the next pending fit; it is not a fixed 540-fit log-based estimate. Heavy jobs are
scheduled first. DataLoader workers are zero because samples are already cached
in RAM; this does not restrict PyTorch to one CPU thread.

Each completed fit saves a checkpoint, 25-row training history, predictions and
report with SHA256. Valid completed fits are reused after a restart. **Interrupted
fits restart from epoch 1**, rather than silently resuming without optimizer state.
Run fingerprint includes code/data manifest, all named embedding-cache hashes and
software versions. Changing GPU count or CPU thread count is allowed, but runtime
settings are recorded. Scientific changes require a new output location/version.
Do not rebuild or resend the contract while a run is active.

The return archive includes all 2,256 checkpoints for independent validation and
will be several GB. Enough disk space is required for the archive and extracted
artifacts. No recursive workspace scan is performed: all fit/input paths are explicit.

## Commands

### 1. Send from the local CPU server

```bash
cd /disk9/13.Heesu_Allostery
GPU_LOGIN=hs0517@secondsite.snu.ac.kr \
bash analysis/double_anchored_lofo/scripts/send_gpu_input.sh
```

Override `GPU_LOGIN` with your SSH host/alias if necessary. The default remote
project root is `/disk1/11.HS_allostery`; existing embedding caches must be there.
Transfer only sends this new package, not prior results or large embedding files.

### 2. Launch on the GPU server

```bash
cd /disk1/11.HS_allostery
mkdir -p analysis/double_anchored_lofo/gpu_output/logs
nohup env GPU_IDS=0,1,2,3 CPU_THREADS_PER_GPU=6 \
  bash analysis/double_anchored_lofo/scripts/run_gpu.sh \
  >> analysis/double_anchored_lofo/gpu_output/logs/run.nohup.log 2>&1 < /dev/null &
```

```bash
python3 analysis/double_anchored_lofo/scripts/status.py
tail -F analysis/double_anchored_lofo/gpu_output/logs/run.nohup.log
tail -F analysis/double_anchored_lofo/gpu_output/logs/train_worker0.log
```

Ctrl-C in a `tail` terminal only stops log viewing. To stop training, send TERM to
the PID stored in `gpu_output/supervisor.pid`; the supervisor stops its workers.
Run the same launch command to resume completed-fit reuse. Do not delete output.

### 3. Fetch and run local CPU analysis

```bash
cd /disk9/13.Heesu_Allostery
BOOTSTRAP_WORKERS=32 GPU_LOGIN=hs0517@secondsite.snu.ac.kr \
bash analysis/double_anchored_lofo/scripts/fetch_gpu_return.sh
```

If only local aggregation needs rerunning (no download and **no GPU training**):

```bash
LOCAL_ONLY=1 BOOTSTRAP_WORKERS=32 \
bash analysis/double_anchored_lofo/scripts/fetch_gpu_return.sh
```

Principal outputs: `local_analysis/METRICS.tsv`, `PRIMARY_PAIRED_BOOTSTRAP.tsv`,
`ATP_ADP_EXCLUSION_SENSITIVITY.tsv`, `PER_LIGAND_RESULTS.tsv`, and `VALIDATION.json`.
The last file must say `validated`. GPU `FITS_VALIDATION.json` alone does not mean
local bootstrap has completed.

## Current verification boundary

Local static checks verify the frozen source, 94 train/test partitions, coverage,
leakage assertions, CPU metrics and multiplicity test. CUDA is not available in the
local development environment; actual eight-head forward validation and cache
validation occur in GPU preflight before any training job is queued.
