# Frozen ChEMBL external-evaluation contract

## Scientific question

Does allosteric-versus-orthosteric discrimination learned from the curated,
protein-anchored benchmark rank ChEMBL pairs carrying independently derived
allosteric text support above appropriate comparison pairs?

This is an external prioritization analysis.  Word-mined labels are noisy
evidence and are not described as ground truth.

## Training

The primary deploy pool is Arm A from `allosteric_pair_benchmark_main`:
4,637 unique full-InChIKey pairs, 426 proteins, and both labels on every
protein.  The full pool is used for deployment training.  No ChEMBL 5A,
van Westen 2014, or Burggraaff 2020 word-mined row is supervision.

Fixed deploy epochs are the medians of the 15 completed unseen-family CV fits:

- ligand-only: 9 epochs;
- protein-only: 4 epochs;
- C2: 4 epochs.

Seeds are 20260817, 20260818, and 20260819.  Optimization, weighting, hidden
dimension, dropout, and input truncation equal the main benchmark.

Inference keeps the mixed-precision forward pass but casts logits to FP32
before sigmoid. Legacy prediction files lacking this explicit numerical
contract are invalidated and rescored; deploy weights do not require
retraining.

## External universe and contamination exclusion

The source universe is the existing ChEMBL36 OOD table and its validated
four-shard tensor cache.  Before inference, a row is removed when its exact
UniProt--InChIKey connectivity pair occurs in either Arm A or the completed
broad-superset benchmark.  Full-InChIKey overlap is recorded as an additional
audit.  Rows failing the pre-existing embedding contract are absent from the
cache and counted, not imputed.

The old `Is_Ligand14_SeenInDevelopment` field is not used because it refers to
an earlier training pipeline.  Novelty is recomputed against the current
benchmark.

## References and endpoints

### Primary: Burggraaff 2020

Stable ChEMBL target and compound identifiers link ChEMBL22 weak labels to the
ChEMBL36 OOD universe.  Pure allosteric and pure orthosteric pairs form the
binary evaluation; conflicting pairs are excluded.  Primary metric is pooled
allosteric-positive AUPRC.  AUROC and equal-target macro metrics are secondary.
The C2-minus-ligand difference is bootstrapped over target clusters.

This reference is restricted to Class A GPCRs and therefore tests recovery in
a pharmacologically important but family-narrow domain.  It is not evidence
of generalization across all protein families.

### Secondary: local ChEMBL 5A

5A supplies positive allosteric-like text hits only.  No unlabeled row is
converted to a negative.  Evaluation reports enrichment among fixed top
fractions of the full ranked universe, both pooled and after ranking within
target.  AUPRC against the unlabeled universe is forbidden.

### Candidate output

Top-scoring pairs lacking either weak label are exported for literature
triage.  They are model-ranked candidates, not discoveries or validated
allosteric pairs.

## Representation limitation

The external cache contains Uni-Mol ligand tensors and full-chain ESM3 residue
tensors.  It does not contain uniformly validated fpocket masks under the new
target-chain contract.  Therefore full-universe inference is limited to
ligand-only, protein-only, and C2.  C1/C3/D1 remain internal ablations.

## Compute isolation

Each worker is pinned to one GPU through `CUDA_VISIBLE_DEVICES`.  With two GPU
IDs, shards are assigned round-robin to two independent processes.  The code
never invokes `DataParallel` and never selects unlisted GPUs.
