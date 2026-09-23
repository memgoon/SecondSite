# ChEMBL completion: four training cohorts, eight models

## Scope

INFERENCE ONLY: **zero new training fits**. Reuse 33 existing deployment weights
(11 arm/head combinations x 3 seeds): all eight double-anchored heads, plus C3
for every-pair, protein-anchored and ligand-anchored training. Existing 21 arm/head
predictions remain unchanged. Final outputs cover four training cohorts x eight
heads on reference, full screening and the source-linked biochemical subset.

Reference: 29,929 frozen input rows. Primary labelled common structural subset:
10,628 rows (4,941 allosteric / 5,687 orthosteric), 226 targets; target-macro
discrimination uses 38 targets and 3,497 rows with both labels. Do not present
this entire reference as a metabolite-domain evaluation. The double-anchored
model is trained on a small, chemically restricted cohort and applied out of domain.

Full screen: 796,165 frozen rows after the existing broad-pair blacklist and
legacy_ood_label != 0 rule. No thresholds or eligibility rules are chosen from
the new scores. Biochemical: the same 492 row IDs are selected from the new full
screen, NOT inferred again. This is an orthosteric-source-linked subset, not an
independent metabolite catalog. No allosteric-text positives means no enrichment
evaluation; unknown rows are never true negative labels.

All training data, deployment epochs, seeds, weights, and protein/ligand inputs
are unchanged. C models use the existing selected structure-chain inputs and
D models take pocket residues from that chain. This is **not** the full protein
sequence representation pilot. Missing structure is NA, never zero probability.
Only available rows are scored; comparisons use identical available rows.

C3 is now included in reference AND full screening. It uses residue/atom
bidirectional attention and can be much slower than other heads even without
training. No short runtime is promised. Every job holds at most 4,096 source rows
and writes an independent, checksummed three-seed prediction table. Four GPUs
share a file-locked queue; completed chunks are not repeated. There is no DDP.
Only the interrupted chunk restarts. Preflight hashes explicit input paths once
and loads all 33 checkpoints for real-input forward tests. No directory tree scan.

FP32 sigmoid is applied after exiting autocast. New single-input inference uses
batch one, actual-input deduplication and broadcasting; repeated scores are ties,
not an arbitrary preference among pairs. Pair batches are length-sorted and trim
padding. Older imported means/SD are preserved exactly, not recomputed.
Minor floating point differences from old biochemical inference are possible
because those 492 scores now come from the full-screen batching, with the same weights.

## Preparation / transfer (local CPU host)

The CPU contract is already generated. Existing `role_complete_pair_matrix` and
`ligand_anchored_chembl` predictions, deployment checkpoints and embedding caches
must remain at their current paths on BOTH servers. They are read-only dependencies.
Only the small new package is transferred, not the old embeddings or weights.
Some reference ligand tensors are stored only on the GPU host. Their expected
hashes are taken from the validated preceding GPU run manifest, not reconstructed
or assumed locally. GPU preflight verifies their actual bytes before inference.

The prior CPU-only `ligand_anchored_chembl/local_analysis/VALIDATION.json` is
bundled as `data/PRIOR_LOCAL_VALIDATION.json`, including its original path and
SHA256. Its original location is NOT a GPU dependency. The initial package
incorrectly required that local-only path; this packaging correction does not
change model weights, inference settings or tasks. Re-send the package after
this preflight failure; do not create a dummy validation file or disable checks.

```bash
cd /disk9/13.Heesu_Allostery
bash analysis/chembl_four_cohort_completion/scripts/send_gpu_input.sh
```

## GPU execution

Optional quick check of the exact contract paths (all 33 weights and reports,
without reading tensor contents or scanning directories):

```bash
cd /disk1/11.HS_allostery
python3 analysis/chembl_four_cohort_completion/scripts/check_paths.py
```

This does not replace the full GPU preflight hash and input-forward checks.

```bash
cd /disk1/11.HS_allostery
mkdir -p analysis/chembl_four_cohort_completion/gpu_output/logs
nohup env GPU_IDS=0,1,2,3 CPU_THREADS=6 C3_BATCH_SIZE=4 \
  bash analysis/chembl_four_cohort_completion/scripts/run_gpu.sh \
  >> analysis/chembl_four_cohort_completion/gpu_output/logs/run.nohup.log \
  2>&1 < /dev/null &
```

Default ordinary batch 16, C3 batch 4, single-input batch 1. CPU threads are per
worker; choose settings before the first run. Do not change batch/thread settings,
source code or caches during resume. Changed fingerprints fail closed.
`RUN_STAGE=preflight` tests inputs without starting prediction; `RUN_STAGE=pack`
revalidates finished chunks and rebuilds the return archive without inference.

```bash
python3 analysis/chembl_four_cohort_completion/scripts/status.py
tail -F analysis/chembl_four_cohort_completion/gpu_output/logs/infer_worker0.log
```

The launcher owns a run lock and coordinates worker termination on SIGTERM/INT.
The same command resumes finished compatible chunks. Do not start a competing run
or move input files while inference is running. Logs append rather than truncate.

## Retrieval and local CPU analysis

```bash
cd /disk9/13.Heesu_Allostery
BOOTSTRAP_WORKERS=16 \
  bash analysis/chembl_four_cohort_completion/scripts/fetch_gpu_return.sh
```

CPU aggregation and 10,000-replicate bootstrap run on the LOCAL server, not GPU.
Use `LOCAL_ONLY=1 BOOTSTRAP_WORKERS=16 ...` to repeat analysis without downloading.
No local torch, RDKit or sklearn is required. The return archive includes all
new chunk predictions, individual seed scores and their provenance reports;
it does not duplicate the existing checkpoints. Worker logs stay on the GPU host.

Reference: pooled AUROC/AUPRC and target-macro AUROC on the same rows, all 32
combinations. Common novelty strata use every-pair training identities for all
arms; arm-relative novelty flags are also exported but not mixed into matched
comparisons. Pfam annotation missing/incomplete is not called unseen.
Reference bootstrap: 72 paired comparisons (three arms vs every-pair for eight
models, pooled/target-macro; and six pair heads vs ligand-only within each arm).
10,000 target-cluster resamples preserve multiplicities; at least 95% must be
defined. CIs are conditional on fixed ensembles and pointwise, not corrected
for multiple comparisons or related-target family dependence. Do not select
the highest-scoring model afterward as a prospectively validated winner.

Full/biochemical: pooled and within-text-positive-target enrichment at 0.1%,
0.5%, 1%, 5%, common novelty and Pfam strata. Minimum one top-k row per eligible
target; ties use frozen row ID. Text enrichment is annotation recovery, not
precision or AUROC against unknown compounds. No enrichment CI is claimed.

## Files for the web session (no Django/PostgreSQL changes)

`local_analysis/PAIRS_REFERENCE.tsv.gz`, `PAIRS_FULL.tsv.gz`,
`PAIRS_BIOCHEMICAL.tsv.gz`: shared metadata, labelled/reference or weak-text flags,
input availability, and novelty relative to EACH of the four training datasets.
Primary join keys: reference_row_id for reference, OOD_Row_ID for full/biochemical.

`RANKINGS_{REFERENCE,FULL,BIOCHEMICAL}_{arm}.tsv.gz`: one table per scope and
training cohort. Every model has a mean score, seed SD, global rank, and
within-target rank. Rank 1 is highest; unavailable predictions remain NA.
Equal scores share the minimum rank. Cohort names: every_pair, protein_anchored,
ligand_anchored, double_anchored. Model names: ligand, protein, c1-c3, d1-d3.
Historical score prefixes remain general=protein_anchored and
role_complete=double_anchored; this explicit mapping is in `scripts/common.py`.
`WEB_COLUMN_MAP.tsv` gives all 32 cohort/model combinations, corresponding score
and rank columns, training-set sizes, and whether each result is newly inferred
or reused. It is copied into local_analysis for the web-session handoff.
The `p_` prefix is compatibility naming, **not** a calibrated allostery probability.
Individual seed columns exist for newly inferred combinations and previous
ligand-anchored predictions. Old every-pair/protein-anchored scores supplied only
mean/SD; do not invent per-seed values or interpret missing seed columns as NA means.
All available means and SD are retained for all 32 combinations.

`REFERENCE_METRICS.tsv`, `REFERENCE_PAIRED_BOOTSTRAP.tsv`, `TEXT_ENRICHMENT.tsv`
and `VALIDATION.json` are the analysis outputs. `CANDIDATES_{arm}.tsv.gz` lists
top 100 per model among rows with neither allosteric nor orthosteric text, with
a separate common double-novel selection. These are candidates for review, not
experimentally established allosteric binders. No automatic default model or
newly discovered site is asserted. Existing web files and manuscript are untouched.
