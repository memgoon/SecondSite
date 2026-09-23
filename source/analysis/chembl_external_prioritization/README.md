# ChEMBL external prioritization

This package asks whether a model trained without ChEMBL text-derived labels
can prioritize independently word-mined allosteric compound--target pairs.

The primary reference is the Burggraaff et al. 2020 ChEMBL22 set because it
contains both allosteric and orthosteric weak labels.  The local `5A` ChEMBL
text set contains only allosteric-like positives and is therefore evaluated by
rank enrichment, never by treating the unlabeled background as a negative
class.

## Fixed design

- Deploy training: the 4,637-pair, 426-protein Arm A cohort in
  `allosteric_pair_benchmark_main`.
- Models: ligand-only, protein-only, and C2 whole-protein cross-attention.
- Training labels: no ChEMBL text-derived label is admitted.
- Exclusion: any ChEMBL target--ligand connectivity pair present in either the
  Arm A or broad-superset benchmark is removed before evaluation.
- Primary endpoint: allosteric-positive AUPRC on mapped 2020 allosteric versus
  orthosteric weak labels.
- Secondary endpoints: AUROC, target-macro metrics, target-cluster bootstrap,
  and 5A enrichment at fixed top-ranked fractions.
- Full-screen outputs are candidate priorities, not validated mechanisms.
- Mixed-precision forward passes cast logits to FP32 before sigmoid. Legacy
  inference reports lacking this numerical contract are automatically rescored.

Only C2 is used as the pair model for the full ChEMBL universe.  C1, C3, and
D1 require pocket masks that do not exist under a uniform contract for the
full external target set.

## Compute design

The existing four-shard ChEMBL tensor cache is reused in place.  It is roughly
196 GB and is never transferred.  Three independently initialized deploy
models are trained on all Arm A rows at fixed epoch counts learned from the
completed family-held-out experiment.

Two GPUs are used as two independent single-GPU workers.  There is no
`DataParallel`, distributed training, or automatic use of all visible GPUs.
Set `GPU_IDS=0` for one GPU or `GPU_IDS=0,1` for two.

## Local preparation and transfer

```bash
python3 analysis/chembl_external_prioritization/scripts/build_cpu_contract.py
bash analysis/chembl_external_prioritization/scripts/send_gpu_input.sh
```

## GPU server

Quick reference-only run:

```bash
GPU_IDS=0 RUN_SCOPE=reference bash /disk1/11.HS_allostery/analysis/chembl_external_prioritization/scripts/run_gpu_chembl.sh
```

Full ChEMBL screen with two explicitly selected GPUs:

```bash
GPU_IDS=0,1 RUN_SCOPE=full bash /disk1/11.HS_allostery/analysis/chembl_external_prioritization/scripts/run_gpu_chembl.sh
```

Both commands are resumable.  The reference run trains the deploy models; the
full run reuses them. Existing deploy weights do not require retraining for the
FP32-sigmoid correction, but legacy reference and full-screen predictions do.

## Fetch

```bash
bash analysis/chembl_external_prioritization/scripts/fetch_gpu_return.sh
```

The default return archive contains validation reports, aggregate metrics,
candidate tables, logs, and deploy checkpoints.  Full row-level prediction
shards remain on the GPU server because they are large.
