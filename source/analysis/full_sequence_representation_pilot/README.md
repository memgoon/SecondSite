# Full-sequence representation sensitivity pilot

This bounded experiment asks whether the frozen protein-anchored,
unseen-family benchmark changes when C-models receive the full canonical
UniProt sequence rather than the selected PDB target chain used originally.
It is not a replacement for the main benchmark and does not reconstruct a
biological assembly, mature processed chain, or pair-specific binding site.

## Frozen scope

- Cohort: protein-anchored, 4,637 pairs, 426 proteins, 165 Pfam components.
- Split: frozen unseen-family folds only.
- Heads: protein-only, C1, C2, and C3.
- Seeds/folds: 3 model seeds × 5 folds.
- New fits: 120 total: 60 full sequence and 60 selected-chain control.

## Three-way comparison

| Representation | Role |
|---|---|
| Legacy selected structure chain | Existing benchmark input and OOF baseline. |
| Rechunked selected structure chain | Same residues as legacy, re-embedded with the same overlap-chunk procedure as full sequence. This is the embedding-method control. |
| Full canonical UniProt sequence | All canonical residues, overlap-chunked and stitched. |

The primary contrast is full canonical sequence minus rechunked selected chain.
The control contrast shows whether chunking alone altered the result.

## Interpretation

Coverage is aligned canonical residues divided by canonical length.  In this
cohort, 215/426 proteins are below 90%, 92/426 below 50%, and 42/426 below
25% coverage.  The main endpoint is fold-restricted within-ligand AUROC;
within-protein and pooled AUROC are secondary.

Full canonical sequence still omits assembly geometry and may contain a
precursor, propeptide, or irrelevant domains.  This therefore tests sequence
representation sensitivity only.  A separate table also excludes the two
proteins whose canonical sequence exceeds 4,096 residues.

## GPU workflow

1. Run `scripts/send_gpu_input.sh` locally.
2. Run `RUN_STAGE=all` through `scripts/run_gpu.sh` on the GPU server.
3. Run `scripts/fetch_gpu_return.sh` locally after completion.  It recomputes
   the aggregate and bootstrap locally from returned OOF predictions.
