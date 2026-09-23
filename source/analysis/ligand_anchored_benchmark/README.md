# Ligand-anchored fourth cohort: 4 x 4 benchmark extension

This is the execution contract, superseding the earlier `GPU_TASK_PLAN.md` proposal.
Existing results are preserved. Only the new ligand-anchored cohort is trained.

## Matrix and main question

Datasets: every-pair / protein-anchored / ligand-anchored / double-anchored.
Splits: row-random / protein-family-held-out / ligand-held-out / explicit double-unseen.
Here **protein-unseen retains the prior family-held-out definition**, not a new
exact-UniProt-only grouping. The main question is whether ligand-only pooled
performance is reduced in the ligand-anchored cohort under family-held-out, and
whether pair heads outperform protein-only within the same ligand. Double-unseen
is included to fill the fourth split, not promoted to the main claim.

**480 new fits** = 4 splits x 8 heads x 5 folds x 3 seeds. Three old cohorts are
not retrained. Models and seeds are unchanged: ligand, protein, C1--3, D1--3;
20260817/18/19. The four-by-four summary is descriptive: evaluation cohorts differ,
and the existing double-anchored double-unseen cell uses 47 folds/fixed 25 epochs,
not the five-fold validation-selected protocol. Source/protocol/denominator fields
are preserved in `MATRIX_4X4_ALL_MODELS.tsv`; do not hide these differences.

## Frozen cohort and folds

Source: explicit_double_unseen/data/EVERY_PAIR.tsv.gz, 6854 pairs. Keep full
InChIKeys observed with BOTH roles: 1365 pairs, 543 proteins, 122 exact ligands,
251 family components, 405 allosteric and 960 orthosteric. Only 143 proteins have
both labels; 755 rows belong to 400 one-label proteins. Exact ligand identity does
not determine the label, but ligand prevalence still ranges .076923--.9; pooled
ligand-only performance is not forced to chance. No ligand reweighting/downsampling.

274 rows have legacy random/family folds -1. Preserve all assigned folds, then
assign missing rows by SHA256(20260916|row_id) and minimum row load. Assign 108
missing families atomically, descending size then family ID, to minimum row load
(ties lowest fold). All 1365 rows now have OOF predictions. Ligand folds reuse
existing connectivity-disjoint assignments; full InChIKey remains the anchoring
and primary within-ligand grouping identity.

Test fold k, validation fold (k+1)%5, other three folds for training. Double-unseen
uses the same family test folds, removes test connectivity from validation, then
test plus retained validation connectivity from training. All required train/val/
test overlaps are checked, not just train-test. No test scores choose folds.

| Split | Train rows (range) | Test rows | Within-ligand eligible rows / fold-ligand blocks |
|---|---:|---:|---:|
| random | 819 | 1365 total | 884 / 151 |
| family | 808--827 | 1365 total | 859 / 122 |
| ligand | 721--918 | 1365 total | 1365 / 122 |
| double-unseen | 65--133 | 1365 total | 859 / 122 |

The 122 family-held-out blocks contain 79 DISTINCT ligands. Do not conflate block
counts with independent ligands. Double-unseen remains training-data limited;
training rows derive from ligand-anchored data but need not retain both roles per
ligand after purging. Conditional metric support differs across splits; a naive
recovery ladder does not isolate a split effect on identical comparisons.

## Training and checkpoint contract

Use prior main-benchmark settings: maximum 25 epochs, patience 5, minimum 1 epoch;
validation-selected checkpoint, not the no-validation rule of the 395-row LOFO
pilot. Hidden 256, 4 heads, dropout .30, AdamW lr/weight decay 1e-4, clip 5, batch
6 (C3=1). Weighted minibatch BCE is sum(w*BCE)/sum(w), matching the main benchmark.
Weights are train-only protein/label inverse counts with equal label totals.

Checkpoint selection retains the ORIGINAL split-aware rule (supersedes the
proposal to use within-ligand for every regime):
- ligand-only: validation within-protein AUROC;
- protein-only: validation within-ligand AUROC;
- pair heads: family/double -> within-ligand; ligand -> within-protein;
  random -> mean of both, with both required;
- every requested metric needs >=8 two-label groups, otherwise recorded pooled
  symmetric-AP fallback. This matters especially for the small double-unseen val.

The frozen support predicts 75 fallback fits, all in double-unseen (75/120);
the other three regimes have zero support-based fallbacks. The validator checks
this exact count, not merely whether a report file exists.

The substantive MAIN family-held-out comparison (pair vs protein-only) shares the
same selection metric. In other regimes within-ligand comparisons are descriptive;
do not imply all heads were selected by the same endpoint. Report fallback rates,
seed dispersion and epoch-1 fraction (>=1/3 flagged). Test is evaluated only after
checkpoint selection, and saved best validation predictions are independently
re-scored on fetch to verify the selected epoch.

Frozen architectures/embeddings: C uses whole SELECTED TARGET CHAIN, D pocket
residues; not canonical UniProt sequence or a new encoder. No transferred head
weights, new embeddings, ChEMBL screen or ATP/ADP exclusion in this experiment.
FP32 sigmoid after AMP logits. Single-input heads infer once per actual identical
input with batch 1 and broadcast, preventing padding-dependent rank artifacts.

## Evaluation and uncertainty

Report pooled AUROC/AP, within-protein, within-exact-ligand, and family-macro for
all individual seeds and mean-score ensemble. The main pooled ligand-only table
is mandatory: within-ligand ligand-only=0.5 is a mathematical control and CANNOT
prove that ligand-only signal disappeared from the whole cohort.

Primary pair lift: within-ligand pair minus protein-only, same rows/blocks. Include
all six pair heads, not a selected winner. Protein-only within protein and ligand-
only within exact ligand must be exactly .5 wherever the metric is defined.

Local bootstrap: 10,000 paired EXACT-LIGAND-cluster replicates, all fold blocks
of each ligand travel together and repeated draws retain multiplicity. 56
comparisons (4 x [8 chance contrasts + 6 protein-only contrasts]). CIs are
**conditional on the fixed protein/family panel**, not family-population CIs:
shared families across ligands and overlapping training sets are not fully
accounted for. No multiplicity-adjusted confirmatory claim from an isolated CI.
This scope is explicit rather than silently recycling a nested protein bootstrap
for crossed family/ligand blocks.

Additionally compute exact delete-one-family sensitivity for family/double tests,
recomputing ligand AUROCs and valid groups after each deletion. This flags family
concentration but is NOT called a confidence interval. Family-population inference
would require a separately specified and validated crossed-dependence analysis.
No bootstrap on the GPU server; local fetch performs these CPU tasks.

## Runtime, safety and outputs

Four GPUs by default; CPU_THREADS_PER_GPU=6; dynamic SQLite fit queue. Only named
cache paths are hashed, no recursive workspace scan. Frozen code/data/dependencies,
all embedding hashes and software versions enter the run fingerprint. Completed
fits are reused only with compatible identities and artifact hashes. Interrupted
fits restart, not resume without optimizer state. Do not resend/rebuild an active
package. CUDA preflight checks cache loading and all eight heads before training.

Return includes best checkpoints, validation/test predictions, histories and
reports. GPU validates and archives. Fetch validates hashes, metadata, selection
replay and independent best-validation scores, then aggregates on local CPU.
No sklearn dependency. Expect a several-GB return archive, not just metric tables.

Key outputs under `local_analysis/`:
- `MATRIX_4X4_ALL_MODELS.tsv`, `MATRIX_LIGAND_ONLY_POOLED.tsv` and pooled/within-
  protein matrix views: old result snapshots plus new fourth cohort, not retraining.
- `METRICS.tsv`, `LIGAND_ONLY_ALL_METRICS.tsv`: new cohort, all denominators.
- `WITHIN_LIGAND_CONDITIONAL_BOOTSTRAP.tsv`: conditional ligand-cluster CIs.
- `DELETE_ONE_FAMILY_PAIRED_LIFT.tsv`: family deletion sensitivity, not CIs.
- `TRAINING_STABILITY.tsv`, `VALIDATION.json`: required audit outputs.

## Commands

Local CPU server, send new package:
```bash
cd /disk9/13.Heesu_Allostery
bash analysis/ligand_anchored_benchmark/scripts/send_gpu_input.sh
```
Default SSH: hs0517@secondsite.snu.ac.kr; override GPU_LOGIN if needed. Existing
embedding caches must already exist under REMOTE_ROOT (default /disk1/11.HS_allostery).

GPU server:
```bash
cd /disk1/11.HS_allostery
mkdir -p analysis/ligand_anchored_benchmark/gpu_output/logs
nohup env GPU_IDS=0,1,2,3 CPU_THREADS_PER_GPU=6 \
  bash analysis/ligand_anchored_benchmark/scripts/run_gpu.sh \
  >> analysis/ligand_anchored_benchmark/gpu_output/logs/run.nohup.log 2>&1 < /dev/null &
```
Progress/logs:
```bash
python3 analysis/ligand_anchored_benchmark/scripts/status.py
tail -F analysis/ligand_anchored_benchmark/gpu_output/logs/train_worker0.log
```
Stop training by sending TERM to the PID in gpu_output/supervisor.pid; Ctrl-C in
tail only stops log viewing. Restart with the same launch command; do not delete
outputs. CPU threads can change; scientific/software fingerprints cannot silently
change within one output directory.

After completion, on the local CPU server:
```bash
cd /disk9/13.Heesu_Allostery
BOOTSTRAP_WORKERS=32 bash analysis/ligand_anchored_benchmark/scripts/fetch_gpu_return.sh
```
Retry CPU work only:
```bash
LOCAL_ONLY=1 BOOTSTRAP_WORKERS=32 bash analysis/ligand_anchored_benchmark/scripts/fetch_gpu_return.sh
```

Local static/synthetic tests do not claim CUDA execution. Real eight-head forward
tests run on the GPU server; no remote jobs are launched during code preparation.
