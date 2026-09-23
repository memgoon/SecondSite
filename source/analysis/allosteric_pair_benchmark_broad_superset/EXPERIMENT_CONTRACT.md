# Frozen experiment contract

## Question and comparison

The experiment isolates the effect of the training pool. Arm A uses proteins
with both labels. Arm B must be a strict superset of Arm A and adds every exact
pair that passes the same graph-free input contract. Both arms are evaluated
on the same frozen Arm A validation and test rows.

This is not an evidence-source ablation. ASD, BioLiP, BRENDA, GtoPdb, KLIFS,
UniProt, and KEGG provenance are not removed one at a time.

## Frozen Arm A

- 4,637 exact protein-full-InChIKey pairs;
- 426 proteins;
- 1,511 allosteric and 3,126 orthosteric rows;
- both labels for every protein;
- original row-random and Pfam-component fold assignments unchanged;
- original Arm A 180 fits and OOF predictions reused, not retrained.

## Arm B construction

Arm B is `A union E`. Before GPU ligand validation it contains 6,975 pairs and
950 proteins: all 4,637 Arm A rows plus 2,338 additional rows. Frozen Arm A
target-chain ESM3 tensors and pocket masks are reused. Additional proteins must
pass the same thresholds:

- target-chain alignment identity at least 0.70;
- target-chain coverage at least 0.50;
- at least 30 aligned residues;
- at least five mapped fpocket residues;
- union of at most five highest druggability-score fpocket pockets.

The final Arm B count may only decrease through a documented failure of an
additional exact ligand tensor. Loss of any Arm A row is a fatal validation
error. Single-label additional proteins are allowed.

## Exact ligand identity

No first-block/connectivity fallback is permitted for model inputs. An
existing ligand tensor must have the complete full InChIKey in its filename.
A generated tensor is accepted only if RDKit reproduces the complete original
InChIKey from the generating SMILES. PubChem is used only to retrieve a
candidate isomeric SMILES for an otherwise unresolved original key, followed
by the same RDKit equality gate.

## Locked splits

For outer fold `f`, `(f + 1) mod 5` is validation.

### Row-random

Arm B training is the frozen Arm A training rows plus all model-ready
additional rows. Protein, family, and compound reuse is allowed. No exact Arm
A validation/test pair may enter training.

### Unseen-family

Arm B training is the frozen Arm A family-training rows plus eligible
additional rows. An additional row is ineligible if its protein is a locked
validation/test protein, if any exact Pfam accession overlaps the locked
validation/test proteins, or if Pfam is unavailable. Protein and Pfam overlap
must both be zero.

### Common unseen-compound subset

No model is trained for this condition. Within each unseen-family fold, a core
test row is retained when its first InChIKey block is absent from Arm B
training and locked validation. These exact row identifiers are used for both
Arm A and Arm B.

## Models, optimization, and endpoints

The six architectures, initialization seeds, optimizer, weighting, early
stopping, and maximum epochs are imported from the frozen main trainer.
Additional compute is 6 models x 2 regimes x 5 folds x 3 seeds = 180 fits.

Primary endpoint: allosteric-positive AP on concatenated locked OOF test rows.
Secondary endpoints: orthosteric-positive AP, symmetric AP, AUROC,
protein-macro and Pfam-component-macro metrics, plus the common
unseen-compound subset. Probabilities are averaged across seeds before a
10,000-replicate paired Pfam-component bootstrap of Arm B minus Arm A.

## Fail-closed gates

Training does not start unless:

1. class and binary labels agree on every row;
2. all 4,637 Arm A row identifiers and metadata are unchanged in Arm B;
3. target-chain tensor lengths and pocket indices agree exactly;
4. exact full-InChIKey ligand identity is validated;
5. locked validation/test row identifiers equal Arm A for every fold;
6. unseen-family protein and Pfam overlaps are zero;
7. common unseen-compound connectivity overlap is zero.
