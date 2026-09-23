# Frozen experiment contract

## Central question

Can a protein-ligand model distinguish allosteric from orthosteric engagement
when neither the target family nor the compound identity can be reused from
development data, and which input interaction mechanism retains useful signal?

## Reader-facing evaluations

The manuscript and figures use three names:

1. **Row-random**: labelled pairs are distributed across five folds while
   preserving class balance. Protein, family, and compound reuse is allowed.
2. **Unseen-family**: all proteins connected through any shared exact Pfam
   accession are assigned to one component, and components are held out.
3. **Unseen-family + unseen-compound**: no additional model is trained. The
   unseen-family OOF predictions are restricted to test compounds whose first
   InChIKey block is absent from both training and validation.

The third name is intentionally reader-facing. In Methods, it is defined as an
InChIKey-connectivity-group exclusion. It must not be called a Murcko-scaffold
split, double-disjoint training, or a new model.

## Common cohort

All main models use the same graph-free, target-chain-aligned input universe:

- 431 proteins;
- 4,693 exact protein-full-InChIKey rows before GPU ligand-tensor validation;
- 1,526 allosteric and 3,167 orthosteric rows;
- 166 connected Pfam components;
- both labels represented for every protein.

The source cohort contained 448 proteins. Seventeen were excluded before
modelling because no fpocket coordinate file was available or no target chain
met the prespecified alignment and pocket-coverage contract. For every retained
protein, ESM3 is generated from one structure-matched target chain and each
pocket index is a zero-based position in that exact chain. The five highest
fpocket druggability-score pockets are unioned on that chain. The final
model-ready count is frozen after exact ligand tensor validation. A protein is
removed if tensor loss leaves only one class.

## Exact ligand identity

Connectivity-key fallback is prohibited for model input. A ligand tensor is
accepted only if either:

- an existing validated tensor filename contains the complete full InChIKey;
  or
- the tensor is generated from a SMILES for which RDKit reproduces the exact
  full InChIKey.

If the local candidate SMILES fail this check, the workflow may query PubChem
by the original full InChIKey for an isomeric SMILES. That string is accepted
only if RDKit recreates the same full InChIKey; the query URL and outcome are
saved in `PUBCHEM_EXACT_SMILES.tsv.gz`.

The first InChIKey block is used only to define the unseen-compound evaluation
group. It is never used to substitute one stereochemical ligand tensor for
another.

## Main model ablation

- `ligand`: mean-pooled Uni-Mol atom embedding followed by an MLP.
- `protein`: attention-pooled ESM3 representation of the selected
  structure-matched target chain followed by an
  MLP.
- `c1`: ligand and pocket are pooled independently, concatenated, and
  classified without cross-attention.
- `c2`: a pooled ligand query attends to the complete selected target chain.
- `c3`: the C2 interaction is restricted to the union of the five highest
  fpocket druggability-score pockets on that chain.
- `d1`: ligand atoms and the same pocket-union residues attend to one another in both
  directions.

The reduced LABind-style geometry model is not a main model because its graph
requirement reduces the common cohort from 431 to at most 318 structure-ready
proteins and it showed no consistent pilot advantage. It is a Supplementary
analysis and is not an official LABind reproduction.

## Repetition and model selection

Both trained regimes use all five outer folds and three optimization seeds.
For outer fold `f`, fold `(f + 1) mod 5` is validation and the remaining three
folds are training. This yields 180 main fits:

`6 models x 2 trained regimes x 5 folds x 3 seeds`.

Training examples receive equal total weight within each `(protein, label)`
group, followed by equal total weight between the two labels. Early stopping
uses validation Pfam-component-macro symmetric AP, with a maximum of 25 epochs
and patience of five epochs.

## Endpoints

The primary endpoint is allosteric-positive average precision (AP; area under
the precision-recall curve) from concatenated five-fold OOF predictions.
Orthosteric-positive AP, their mean (symmetric AP), AUROC, protein-macro and
Pfam-component-macro results are reported. Every evaluation is also repeated
on the subset of proteins retaining both labels.

Three-seed mean, standard deviation, minimum, and maximum are reported.
Ninety-five percent confidence intervals use 10,000 paired bootstrap samples of
Pfam connected components after probabilities are averaged across seeds.

## Interpretation limits

- Row-random versus unseen-family tests evaluation reuse on a fixed cohort; it
  does not estimate the effect of all upstream data-construction choices.
- Unseen-compound means unseen molecular connectivity, not necessarily a novel
  Murcko scaffold.
- Source and evidence tier remain potential label correlates and must be
  analysed separately.
- Architecture superiority requires paired confidence intervals, not a single
  seed or a single fold.
