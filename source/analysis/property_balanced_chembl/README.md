# Property-balanced ChEMBL deployment

This package runs two locked tasks and two outcome-blind amendments:

1. a new ChEMBL FP32 screen using deploy weights trained on all 3,022
   property-balanced rows;
2. the frozen 30-fit random-draw stability extension as a secondary/SI check;
3. exact-Pfam-overlap novelty strata on the frozen ChEMBL reference universe.
4. property-trained C3 inference on the frozen selected-chain-pocket subset.

The ChEMBL pass scores the existing source ensembles and the new
property-balanced ensembles together. C3 uses separate selected-chain ESM3
tensors because its frozen pocket indices are selected-PDB-chain coordinates,
whereas C2 and ligand-only retain the existing full cache.
The Pfam amendment is joined by UniProt accession and does not require model
retraining or rescoring. Missing Pfam annotation is kept separate from
family-unseen.

The 30-fit count-matched extension is retained as a secondary/SI stability
analysis. It is not the main model-selection gate, and `run_gpu.sh` executes it
after the higher-priority ChEMBL deployment and aggregation.

## Run

From the local server:

```bash
bash /disk9/13.Heesu_Allostery/analysis/property_balanced_chembl/scripts/send_gpu_input.sh
```

On the GPU server:

```bash
GPU_IDS=0,1 RUN_SCOPE=full bash /disk1/11.HS_allostery/analysis/property_balanced_chembl/scripts/run_gpu.sh
```

One GPU is also supported with `GPU_IDS=0`. In two-GPU mode ligand-only and C2
deploy fits run concurrently, C3 then runs on one GPU, and full ChEMBL shards
are distributed across the same two GPUs. Before training, the workflow
generates and validates 808 selected-chain ESM3 tensors. Set `ESM3_HF_TOKEN`
or `HF_TOKEN` if the GPU server's established private ESM3 runner is absent.

After completion, from the local server:

```bash
bash /disk9/13.Heesu_Allostery/analysis/property_balanced_chembl/scripts/fetch_gpu_return.sh
```

`RUN_SCOPE=reference` stops after the two-label reference evaluation.
`RUN_SCOPE=full` additionally runs the complete ChEMBL screen, 5A enrichment,
and candidate export.

The property-trained protein-only comparator is intentionally deferred until
the separately frozen property-balanced matrix-completion run supplies its
matched family-held-out fits and a non-ChEMBL deploy-epoch rule. C3 is reported
only for pocket-available targets (804/837 full-screen proteins and 254/568
reference proteins); matched ligand-only and C2 metrics are emitted alongside
it. Scores outside that subset remain `NA`.
