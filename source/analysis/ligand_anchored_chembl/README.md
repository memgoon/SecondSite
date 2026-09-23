# Ligand-anchored ChEMBL deployment extension

This independent package adds ligand-anchored rankings for SecondSite. It does
not rerun the 4×4 benchmark or alter existing ChEMBL results. The manuscript's
completed external results remain unchanged until this extension is validated.

## Frozen plan

1. Train on all 1,365 ligand-anchored pairs (543 proteins, 122 exact ligands;
   405 allosteric and 960 orthosteric). No external rows enter training.
2. Eight heads × three fixed seeds = **24 deployment fits**, not new CV fits.
   Epochs are the integer median of best validation epochs from the existing
   **15 five-fold Pfam held-out fits per head**. No ChEMBL score selects epochs,
   seeds, heads or thresholds. Optimizer, loss normalization, input tensors and
   training weights match the ligand-anchored benchmark. The final epoch is saved.
3. Freeze the same external rows as the previous screening. All exact
   UniProt–connectivity pairs in the 6,854-pair broad reference remain excluded.
   Reference: 29,929 rows. Full screen: 796,165 rows. Source-linked biochemical
   subset: 492 rows. As in the existing aggregator, `legacy_ood_label == 0`
   rows are excluded from full/biochemical ranking (raw shards have 799,873/520
   rows respectively). This is a fixed prior eligibility rule, not selection
   using the new scores. The last subset is not an independent metabolite catalogue.
4. Full screen and reference: ligand, protein, C1, C2, D1, D2, D3. Biochemical
   subset: all eight, including C3. Full-screen C3 is omitted for the same
   computational reason as the earlier deployment, not on observed performance.
5. Non-ligand heads use the selected target-chain tensor, not mixed UniProt
   tensors; D heads extract pocket residues from that same chain. Unavailable
   inputs yield NA. Three seed scores, their mean and population SD are retained.
   Sigmoid follows FP32 logit casting outside autocast. Single-input inference
   deduplicates actual ligand input/protein identity, scores at batch one, and
   broadcasts, avoiding padding-dependent rank artifacts.
6. Compare all three general training arms on identical structurally available
   rows. Primary cross-arm strata use the protein-anchored training reference.
   Arm-relative novelty is also output but does not constitute matched-row
   comparison. Missing training Pfam annotations prevent an unseen-family claim.
7. CPU outputs: reference pooled AUROC/AUPRC and target-macro AUROC; weak-text
   enrichment at 0.1%, 0.5%, 1% and 5%; exact-target, ligand and double novelty;
   Pfam strata; all-head candidate lists and whole-universe web ranking exports.
   Within-target enrichment includes only targets with text positives and uses
   max(1, ceil(n*fraction)); zero-positive strata remain not evaluated, not zero.
   Ties use frozen row-ID ordering for deterministic top-k. Single-input ties
   carry no within-identity biological preference. No bootstrap is scheduled.

## Before transfer

Existing `ligand_anchored_benchmark` Pfam fits, `role_complete_pair_matrix`
ChEMBL outputs, and their embedding/cache files must still exist on the GPU
server. They are read-only dependencies, not retransmitted or overwritten.
The preflight hashes explicit paths; it never recursively scans directories.
It reads sizable cache files once per launch, so preflight can take time.
It also verifies 120 CV reports/artifacts and tests all eight GPU forward passes.

The input archive is frozen against local scripts/reports/resources. A mismatch
on the server stops execution rather than silently mixing experiment versions.
Python 3.8 + the existing torch 2.4.1/CUDA environment is sufficient. CPU fetch
requires only numpy and pandas: neither torch nor scikit-learn is needed locally.

## Local → GPU

```bash
cd /disk9/13.Heesu_Allostery
bash analysis/ligand_anchored_chembl/scripts/send_gpu_input.sh
```

Override `GPU_LOGIN` and `REMOTE_ROOT` if the old defaults are not your server.

## GPU run (nohup, up to all four GPUs)

```bash
cd /disk1/11.HS_allostery
mkdir -p analysis/ligand_anchored_chembl/gpu_output/logs
nohup env GPU_IDS=0,1,2,3 CPU_THREADS=6 RUN_STAGE=all \
  bash analysis/ligand_anchored_chembl/scripts/run_gpu.sh \
  > analysis/ligand_anchored_chembl/gpu_output/logs/run.nohup.log 2>&1 < /dev/null &
```

`CPU_THREADS` is per GPU worker (four × six threads by default). GPU tasks are
independent processes, not DDP; GPU IDs may contain 1–4 or more available devices.
Only one active run is allowed by a package lock. Logs append, never truncate.

```bash
python3 analysis/ligand_anchored_chembl/scripts/status.py
tail -n 30 analysis/ligand_anchored_chembl/gpu_output/logs/run.nohup.log
tail -n 10 analysis/ligand_anchored_chembl/gpu_output/logs/infer_worker0.log
```

Same command resumes completed deploy fits and inference model/scope blocks.
An interrupted fit restarts that fit, not the full run. An interrupted inference
block restarts that model/shard, not completed blocks. `RUN_STAGE=deploy`, `infer`
or `pack` can rerun only a phase. GPU inference uses fixed batch 16 (2 for the
biochemical scope; 1 for single-input heads). Do not change code mid-run.
No reliable runtime estimate is possible before the first deployment and shard.

## Fetch → CPU validation and web tables

```bash
cd /disk9/13.Heesu_Allostery
bash analysis/ligand_anchored_chembl/scripts/fetch_gpu_return.sh
```

Downloads the return archive, checks SHA256, safely extracts explicit files, and
independently validates all rows, labels, old scores, new score availability,
seed means, novelty and checkpoint hashes. All CPU analysis runs here, not on GPU.
After a local dependency problem, rerun only:

```bash
LOCAL_ONLY=1 bash analysis/ligand_anchored_chembl/scripts/fetch_gpu_return.sh
```

`local_analysis/WEB_FULL_RANKINGS.tsv.gz`, `WEB_REFERENCE_RANKINGS.tsv.gz`, and
`WEB_BIOCHEMICAL_RANKINGS.tsv.gz` are wide-format tables for the web session.
They include stable IDs, target/ligand metadata, three seed scores, mean/SD,
per-model pooled and within-target ranks, availability, and training novelty.
`p_` column names preserve pipeline interoperability, but these scores are **not
calibrated probabilities**. No model is automatically designated the default.
NaN means unavailable, never a zero prediction. Database import and live web
deployment are intentionally not performed by this package.

The return includes all 24 deploy checkpoints, histories and provenance reports,
all nine full prediction tables and reports, and the run fingerprint. Temporary
per-model inference blocks and verbose worker logs stay on the GPU server.

## Validation limits

CPU unit tests, syntax checks and frozen row/epoch audits are run locally.
The local environment has no torch/CUDA; end-to-end GPU execution is not claimed.
GPU preflight must pass before any deployment training starts. This extension
adds rankings for later evaluation and browsing; it does not assert improved
allostery detection or change the current manuscript's numerical conclusions.
