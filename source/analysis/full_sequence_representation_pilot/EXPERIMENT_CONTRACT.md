# Prespecified experiment contract

## Question

How sensitive is the protein-anchored unseen-family benchmark to using a
single selected PDB target-chain sequence rather than a full canonical UniProt
sequence?

## Fixed data and splits

The source is the frozen `PROTEIN_ANCHORED.tsv.gz`: 4,637 rows, 426 UniProt
IDs, and 165 Pfam family components.  Its five existing `matrix_family_fold`
assignments are reused.  All representations are evaluated on identical OOF
row, fold, seed, label, UniProt, ligand-connectivity, and family metadata.

## Inputs

Canonical sequences come from the frozen local UniProt cache.  Selected-chain
sequences come from `TARGET_CHAIN_SEQUENCES.tsv`; their hashes must equal the
frozen cohort hashes.  Both newly embedded representations use ESM3 windows of
1,024 residues with 256-residue overlap, linear overlap-ramp stitching, and
no downstream protein-residue truncation.

The matched selected-chain control is mandatory: it separates a chunking
effect from the effect of adding the rest of the canonical sequence.

## Models and training

Protein-only, C1, C2, and C3 are rerun.  D1–D3 are excluded because a full
sequence does not supply a valid full-protein or assembly-aware pocket mask.
Ligand-only is unchanged and remains an existing interpretive control.

The training hyperparameters, protein-label weights, seed values, split,
model-aware checkpoint selection, and FP32 post-logit sigmoid match the
controlled benchmark.  The design is 2 representations × 4 heads × 3 seeds ×
5 folds = 120 new fits.

## Endpoints and uncertainty

Primary endpoint: fold-restricted within-ligand AUROC under unseen-family
evaluation.  Primary contrast: full canonical sequence minus rechunked
selected chain.  Secondary endpoints are fold-restricted within-protein AUROC
and pooled AUROC.

Coverage strata are `<0.25`, `0.25–<0.50`, `0.50–<0.90`, and `>=0.90`, based
on aligned canonical residues divided by canonical length.  Sparse
coverage-stratified within-ligand values retain support counts but do not get
confidence-interval claims.

An additional transparent sensitivity table excludes the two proteins whose
canonical sequence exceeds 4,096 residues; this guards against a result being
driven by a very long precursor or repeat-rich canonical sequence.

Confidence intervals use 10,000 paired family-component bootstrap replicates.
Repeated clusters retain draw multiplicity.  The aggregate fails if fewer than
95% of replicates define a requested statistic.

## Interpretation limits

Canonical UniProt sequence is not necessarily the mature assay target, a
biological assembly, or the binding-site representation.  It cannot restore a
missing interface partner or validate docking geometry.  A favorable result
supports representation sensitivity; a null or unfavorable result does not
validate the selected-chain input.
