# Frozen experiment contract: pairing control ladder

Date frozen: 2026-08-17

## Question

How much allosteric-versus-orthosteric pair discrimination remains as protein
and ligand reuse are removed, and does explicit protein-ligand interaction
modeling improve over simple concatenation under each control level?

This is a one-seed architecture/control screen. It is not a repeated final
benchmark and does not support significance or superiority claims.

## Common input contract

All five models use the same rows within a stage. The CPU preparation reads
only explicitly named protein, coordinate, and cache files; it performs no
recursive directory scan and no all-versus-all sequence or structure search.

The cleaned uncontrolled pairing resource contains 20,640 exact
`(UniProt, full InChIKey)` pairs on 7,384 proteins. Requiring a canonical ESM3
embedding, a nonempty frozen pocket mask, a canonical sequence, and a named
coordinate candidate produces 6,030 pregraph pairs on 793 proteins. Targeted
coordinate validation gives 4,812 structure-ready pairs on 633 proteins:

- 1,311 allosteric and 3,501 orthosteric rows;
- 318 proteins carrying both labels;
- 318 previously validated graphs reused;
- 315 new graphs built;
- 160 proteins excluded by explicit coordinate/alignment checks.

GPU preparation resolves or generates exact UniMol atomic embeddings and
removes rows with unavailable/invalid ligand, protein, or graph tensors. The
resulting base table is then frozen before any split is constructed.

## Three control stages

1. **Uncontrolled.** Use the broad GPU-ready cohort. Perform a deterministic
   70/10/20 row-level stratified-random split. Protein, Pfam, and ligand reuse
   are allowed and audited.
2. **Protein-controlled.** Use only the already frozen 3,496-pair,
   313-protein common model universe in which every protein carries both
   labels. Preserve the split constructed on the larger 4,770-pair
   pocket-ready cohort: proteins sharing any exact Pfam accession belong to
   one connected component, and components are disjoint across splits. No C9,
   sequence-similarity, whole-chain Foldseek, LB05, or local-pocket 3Di edge is
   used.
3. **Fully controlled (operational name).** Start from the identical
   protein-controlled train/validation/test assignment. Remove validation rows
   whose ligand connectivity key occurs in train; then remove test rows whose
   connectivity key occurs in retained train or validation. The scientific
   description in text and figures must be **Pfam-disjoint with ligand-novel
   evaluation**. It is not a joint protein-ligand connected-component split.

The frozen protein-controlled counts before ligand removal are train 2,177,
validation 624, and test 695. The connectivity-novel diagnostic is expected to
retain train 2,177, validation 321, and test 364; within the latter evaluation
sets, 189 validation rows on 24 proteins and 249 test rows on 23 proteins
retain both labels. GPU validation must reproduce these counts exactly.

## Models

- `c1`: legacy-aligned simple fusion. Mean-pool UniMol atoms, independently
  attention-pool ESM3 residues in the fpocket-union pocket, concatenate the two
  vectors, and classify. There is no protein-ligand cross-attention.
- `c2`: a pooled ligand query attends to the whole ESM3 protein sequence.
- `c3`: the C2 one-way interaction is restricted to the fpocket-union residues.
- `d1`: UniMol atoms and pocket residues attend to one another in both directions.
- `graph_cross_pair`: two C-alpha neighbor/cross-attention blocks over the same
  pocket residues. This is a reduced LABind-style pair adaptation, not LABind.

Relative to historical C1, LayerNorm replaces BatchNorm so that small final
batches are valid; the defining operation remains independent pooling followed
by direct concatenation.

## Matched training and endpoints

All stages and models use the same seed, optimizer, early-stopping rule, hidden
width, and maximum epochs. Within training, every `(protein, label)` group has
equal total weight and the two labels then have equal total weight. This keeps
the uncontrolled stage's many single-label proteins from turning the loss into
a trivial class-prevalence objective.

Report for every stage and model:

- allosteric-positive AP, orthosteric-positive AP, and their mean (symmetric AP);
- pooled, protein-macro, and Pfam-component-macro endpoints;
- AUROC as a diagnostic;
- the subset of test proteins retaining both labels after each control;
- C2/C3/D1/graph-cross deltas against C1 and interaction-model deltas against
  one another.

Early stopping uses validation Pfam-component-macro symmetric AP, with pooled
symmetric AP only as a fallback if no two-label validation component exists.

## Interpretation limits

- Differences from uncontrolled to protein-controlled combine dataset
  anchoring and split control by design; they are a control-ladder result, not
  a pure causal estimate of one split variable.
- Protein-controlled versus fully controlled keeps the protein split and
  training rows fixed, so this comparison isolates ligand novelty in the
  evaluation universe more directly.
- The fully controlled two-label test subset is small and remains a diagnostic
  until repeated runs or outer-fold evaluation are completed.
- The graph model omits LABind's Ankh/MolFormer encoders, DSSP/MSMS features,
  full residue geometry, and residue-level site supervision.
