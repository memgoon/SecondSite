# ChEMBL C3 append-on

This package computes only the missing whole-selected-chain C3 predictions for
the accepted `role_complete_pair_matrix_v2` ChEMBL full screen.  It never
rewrites or recomputes the seven existing model outputs.

The append-on emits four score columns on the exact upstream row universe:

- `p_every_pair_c3_mean`
- `p_every_pair_c3_sd`
- `p_general_c3_mean`
- `p_general_c3_sd`

C3 is evaluated only where the frozen selected-chain tensor exists.  Other
rows remain present with all four scores set to `NA` and
`c3_status=selected_chain_unavailable`.

## Execution

From the CPU server:

```bash
bash /disk9/13.Heesu_Allostery/analysis/chembl_c3_appendon/scripts/send_gpu_input.sh
```

On the GPU server, first measure a deterministic 10,000-row timing sample:

```bash
GPU_IDS=2,3 RUN_STAGE=pilot \
  bash /disk1/11.HS_allostery/analysis/chembl_c3_appendon/scripts/run_gpu.sh
```

Then execute the resumable full append-on:

```bash
GPU_IDS=2,3 RUN_STAGE=full \
  bash /disk1/11.HS_allostery/analysis/chembl_c3_appendon/scripts/run_gpu.sh
```

Completed 10,000-row chunks and completed shards are hash-validated and
skipped on an identical rerun.

After completion, on the CPU server:

```bash
bash /disk9/13.Heesu_Allostery/analysis/chembl_c3_appendon/scripts/fetch_gpu_return.sh
```

The existing ChEMBL rankings are joined later by exact `OOD_Row_ID`; this
package does not mutate the accepted upstream rankings or the web handoff.

