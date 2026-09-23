# ChEMBL full-screen C3 append-on contract

## Scope

The only objective is to fill the C3 omission in the accepted full ChEMBL
screen.  The accepted seven-model full-screen files are immutable inputs.
No model is retrained, no ChEMBL row is added or removed, and no existing
probability is recomputed.

## Frozen models

Two deployment arms are scored:

1. `general_every_pair`, C3, seeds 20260817--20260819;
2. `general_protein_anchored`, C3, seeds 20260817--20260819.

All six checkpoints were trained before external ChEMBL evaluation.  Their
deployment epochs come from the median best epoch across the 15 unseen-family
CV fits for the same cohort and model.  ChEMBL outcomes do not select an epoch.

## Frozen inputs and membership

The four accepted upstream full-screen shards define membership and order.
The C3 runner independently reconstructs each shard from the frozen OOD tensor
cache and requires exact equality of `OOD_Row_ID`, target, ligand, UniProt, and
connectivity fields with the accepted shard before scoring.

Expected upstream accounting is 799,873 rows.  C3 is expected on 785,182 rows
with a validated selected-chain tensor; the remaining 14,691 rows are retained
with explicit missing scores.  These counts are validation gates, not values
silently forced into output construction.

## Computation

C3 is the frozen bidirectional atom--residue cross-attention model.  FP16
autocast is used for model forward passes, logits are cast to FP32 before the
sigmoid, and the three seed probabilities are summarized by their population
mean and standard deviation.

Rows are ordered deterministically by selected-chain length for inference.
Batches are constrained by both maximum row count and padded-residue count.
Production defaults are eight rows and 4,096 padded residues.  These values,
the chunk size, all checkpoint hashes, upstream hashes, and implementation
hashes are part of the run fingerprint.

## Resume and promotion

Production output is checkpointed in deterministic 10,000-row source-order
chunks.  Each chunk is written atomically with a hash-bearing report.  A chunk
is reused only when its run fingerprint, exact row-ID digest, row count, and
file hash all validate.  Shard and union files are likewise written atomically
only after complete validation.

## Interpretation

This is an append-only sensitivity output.  It does not replace C2 as the
prespecified primary model and does not create binding-site, mechanism,
competitive/noncompetitive, or allostery ground truth.  C3 is unavailable
where the frozen selected-chain representation is unavailable.

