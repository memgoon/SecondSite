# New-pairing control ladder

This package runs the newly expanded allosteric/orthosteric pairing resource
through three progressively stricter evaluations and five matched downstream
heads: C1, C2, C3, D1, and a reduced LABind-style graph/cross-attention model.

The frozen scientific design is in `EXPERIMENT_CONTRACT.md`.

CPU preparation has completed:

- clean uncontrolled source: 20,640 pairs / 7,384 proteins;
- named structure candidates: 6,030 pairs / 793 proteins;
- common structure-ready table: 4,812 pairs / 633 proteins;
- graphs: 318 reused, 315 newly built, 160 exclusions audited.

The GPU server performs only UniMol resolution/generation, freezes the three
stage tables, and trains the five downstream heads. ESM3 embeddings and all
structure graphs are fixed.

## Local files

- `data/BROAD_UNCONTROLLED_STRUCTURE_READY.tsv.gz`: GPU input table.
- `data/BROAD_STRUCTURE_GRAPH_AUDIT.tsv`: one-row-per-protein graph audit.
- `validation/BROAD_PREGRAPH_VALIDATION.json`: broad candidate validation.
- `validation/BROAD_STRUCTURE_GRAPH_VALIDATION.json`: graph validation.
- `scripts/send_gpu_input.sh`: one-password transfer to the GPU server.
- `scripts/run_gpu_control_ladder.sh`: GPU preparation and 15 single-run fits.
- `scripts/fetch_gpu_return.sh`: retrieve the compact result archive.

No script recursively searches the project tree.

## Run

From the CPU server:

```bash
bash analysis/new_pairing_control_ladder/scripts/send_gpu_input.sh
```

Then run the single printed SSH command. After completion, retrieve the result:

```bash
bash analysis/new_pairing_control_ladder/scripts/fetch_gpu_return.sh
```
