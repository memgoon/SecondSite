# Frozen experiment contract

## Scientific purpose

The main stress test asks whether a joint protein–ligand model retains signal
when neither exact protein identity nor exact ligand identity deterministically
specifies the allosteric/orthosteric label. It is explicitly restricted to a
biochemical/metabolite-dominated chemical domain and is not claimed to represent all
drug-like allosteric modulators.

## Data arms

The input source and counts are fail-closed:

- every pair: 6,854 rows / 940 proteins;
- protein anchored: 4,637 rows / 426 proteins;
- protein-and-ligand role complete: 395 rows / 98 proteins / 40 exact ligands /
  47 family components / 143 allosteric / 252 orthosteric.

The strict cohort is constructed from the protein-anchored table by repeatedly
removing proteins and standardized full InChIKeys that do not carry both
labels, until the set stops changing. In the frozen result, the 40 exact
ligands also map one-to-one to 40 connectivity groups.

No outcome from a fitted model enters cohort construction.

Role completeness is an identity control, not a source-provenance control.
The positive and comparison labels continue to originate from different
curation pipelines, and no result may be described as eliminating that
confounding.

The broad-only 2,217 rows are training-only augmentation. The every-pair arm
is evaluated on the same 4,637 protein-anchored rows as the protein-anchored
arm; the 6,854 rows are not described as a complete self-OOF evaluation.

The strict cohort is not ligand-ratio balanced. Ligand-specific allosteric
prevalence spans 0.10–0.833 and the prespecified in-sample ligand-prevalence
oracle AUROC is 0.7566. Fold-restricted within-ligand AUROC is therefore the
primary ligand-control statistic. No global balancing weight computed using
held-out labels is introduced.

## Split regimes

All three cohorts are fit under `row_random`, `unseen_family`, and
`unseen_ligand`. The ligand connectivity groups are assigned with a
deterministic row/class-balancing rule. In the two general cohorts,
ligand-held-out CV is a secondary ligand-generalization axis; it does not
replace the harder double-unseen endpoint described below.

For fold `f`, validation is `(f + 1) mod 5` and the other three folds train.
Family overlap must be zero in `unseen_family`; connectivity overlap must be
zero in `unseen_ligand`.

For the two general cohorts, predictions from the family-held-out fit are
also restricted to rows whose ligand connectivity is absent from training
and validation. Simultaneous family-and-ligand
component holdout is not constructed as a separate partition. The
largest joint component contains 96.0%, 98.6%, and 100% of rows in the three
cohorts. For every-pair and protein-anchored training, the hardest endpoint is
the family-held-out OOF subset whose connectivity is absent from train plus
validation. The arm-specific counts are 2,721 and 2,854; direct comparison is
frozen to the common 2,721 rows. The role-complete arm has only 11 rows and
three empty folds, so no double-unseen metric is emitted for that arm.

## Training

- eight models: ligand, protein, C1–C3, D1–D3;
- five outer folds;
- seeds 20260817, 20260818, 20260819;
- 25 maximum epochs, patience 5;
- hidden dimension 256, four attention heads, dropout 0.30;
- AdamW, learning rate 1e-4, weight decay 1e-4;
- mixed-precision forward pass;
- sigmoid after casting logits to FP32;
- equal total training weight per protein–label group and then per label;
- no ligand-specific weighting and no ligand-ratio downsampling.

Checkpoint selection is validation-only and model-aware. Ligand-only always
uses within-protein AUROC and protein-only always uses within-ligand AUROC, so
neither single-input control is selected by a statistic on which its input is
constant. Joint models use within-ligand AUROC for family holdout,
within-protein AUROC for ligand holdout, and the unweighted mean of both for
row-random fits. Every requested conditional statistic must contain at least
eight valid two-class groups; a row-random joint fit must satisfy that rule for
both statistics rather than silently using only one. Otherwise the
prespecified fallback is pooled symmetric AP. The eight-group guard was fixed
from label-support audits without inspecting model performance. Requested
support, fallback reason, and fallback use are recorded for every epoch and
fit. A best epoch of one after fallback is distinguished from epoch-one
selection by a structurally constant control statistic, which is prohibited.
The frozen support audit predicts 132 fallback fits among 1,080 (12.2%), all
in the role-complete arm; this is 132/360 (36.7%) within that arm. The exact
cell-by-cell incidence is stored in `CPU_CONTRACT.json` and must match the fit
reports. Because pooled fallback can use between-identity prevalence for epoch
selection, this incidence is reported as a small-cohort limitation even though
final primary evaluation remains conditional.

Total: 9 cohort-regime cells x 8 models x 5 folds x 3 seeds = 1,080
newly harmonized fits. Existing checkpoints are not mixed into the primary
matrix; all cells use this model implementation and numerical contract.

Trainable parameter counts at hidden dimension 256 are approximately 0.132M
(ligand), 0.591M (protein), 0.722M (concat), 1.052M (one-way attention), and
1.315M (bidirectional attention). Seed dispersion and best-epoch distributions
must be retained; repeated fits do not erase the small-sample limitation of
the 395-row cohort.

## Evaluation

Reported metrics include pooled allosteric AP, orthosteric AP, symmetric AP,
and AUROC, plus fold-restricted protein and ligand macro metrics and family
macro metrics. Pooled metrics are secondary. Family-generalization ladders,
lifts, and retention use fold-restricted within-ligand AUROC; ligand-
generalization analyses use fold-restricted within-protein AUROC. Three-arm
prediction tables use the identical 395 strict rows, but conditional AUROC
uses only fold-by-group blocks containing both labels. Consequently, every
conditional result records input rows, rows used, rows skipped, groups used,
and groups skipped. In the frozen strict cohort the relevant support is
249/395 rows for row-random within-ligand, 240/395 for family-held-out
within-ligand, 129/395 for row-random within-protein, and 87/395 for
ligand-held-out within-protein.

Single-input predictions are deterministic functions of one exact input, but
mixed-precision reductions over batches with different padding shapes can
produce last-digit score differences for repeated inputs. Before computing
metrics, ligand-only scores are therefore averaged within outer fold and full
InChIKey, and protein-only scores within outer fold and UniProt selected-chain
identity. This removes artificial within-identity ranks while retaining
differences between stereoisomers in one connectivity block. Raw within-
identity spans are audited by arm and regime; any probability span above 0.01
fails aggregation. This numerical canonicalization does not change joint-
model predictions or checkpoint selection.

Because those effective conditional row sets differ between row-random and
held-out regimes, their numerical difference is reported as a descriptive
conditional change with both support counts, not as an unqualified causal
split penalty or recovery estimate. A separate pooled split-change table uses
the identical 395 rows but remains secondary because it retains between-
identity prevalence information.

The two general-domain arms additionally use the identical 2,721
double-unseen prediction rows. Their primary hard endpoint is fold-restricted
within-protein AUROC under family holdout, which uses 1,805 rows in 112
two-class protein-by-fold groups and compares joint models with ligand-only.
This asks whether unseen ligands can be ordered within proteins belonging to
unseen families. Fold-restricted within-ligand AUROC uses only 37 rows in 14
groups and is therefore a limited-support sensitivity. Pooled AUROC over all
2,721 identical rows is retained only as a secondary matched-row split-change
sensitivity because it also reflects between-protein prevalence and score
scale. The row-random within-protein comparator uses 654 rows in 94 groups, so
its numerical change versus family holdout is descriptive rather than a
same-effective-row retention estimate.

For family-held-out claims, paired bootstrap resampling uses the held-out
family component and joint models are compared primarily with protein-only;
ligand-only is the structurally constant 0.5 implementation control within a
ligand. For ligand-held-out claims, resampling uses the held-out ligand
connectivity and joint models are compared primarily with ligand-only;
protein-only is the structurally constant 0.5 control within a protein. The
bootstrap unit follows the unit of generalization rather than the macro-group
label. The double-unseen within-protein claim also resamples held-out family
components and uses ligand-only as its nonconstant comparator. Within every
replicate, undefined single-class fold-by-group blocks are excluded using the
same estimator as the observed statistic; valid/invalid replicate counts and
the minimum, median, and maximum effective support are recorded. Ten thousand
replicates are used, and at least 95% must yield a defined paired statistic.
When a metric group is nested in the resampled cluster (notably protein within
family for the hard endpoint), every repeated cluster occurrence receives a
draw identifier so cluster multiplicity is retained in the macro mean. An
embedded A,A,B synthetic check requires the expected 2/3 weighted delta. ATP
and ADP are reported separately and an ATP/ADP-
excluded sensitivity is mandatory.

## External deployment

Deployment epochs are frozen as median CV best epochs:

- every-pair sensitivity ensemble: every-pair unseen-family fits;
- general protein-anchored ensemble: unseen-family fits;
- biochemical role-complete ensemble: unseen-family fits.

Thus every deployment epoch is the median best epoch from the 15
unseen-family CV fits for the same cohort and model; epochs from different
generalization regimes are never pooled.

Full ChEMBL inference scores the every-pair and protein-anchored ensembles on
identical rows and excludes every exact UniProt–connectivity pair in the
6,854-row broad benchmark. Text-derived labels never enter training and remain
weak evidence rather than ground truth.

Full screen models for both general training arms: ligand, protein, C1, C2,
D1–D3. Ligand-only is scored on every row. Every non-ligand model is scored
only where the validated selected target-chain tensor exists; C1/C2 and
protein-only receive that whole-chain tensor, while D1–D3 receive pocket
residues extracted from the same tensor. Missing rows are NA and all model
comparisons use the identical selected-chain subset. The mixed legacy
UniProt/PDB external protein tensor is never used. The source-linked
biochemical screen also includes C3. Whole-chain C3 is omitted from the full
screen because bidirectional atom–residue attention has disproportionate
memory and time cost at that scale, not because its input is unavailable.
Its 850 keys derive only from orthosteric evidence provenance and are not
called an independent metabolite catalog. Exact protein novelty and Pfam
overlap novelty are reported separately; missing Pfam annotations are never
classified as unseen.

Every-pair training is a coverage sensitivity analysis, not a ligand-control
upgrade. The frozen Pfam cache is complete for protein-anchored training but
misses 365 of 940 every-pair proteins; an every-pair Pfam nonmatch is therefore
labeled `training_reference_incomplete` rather than family-unseen.
Protein-anchored training remains the prespecified primary general model. The
every-pair arm is not promoted to primary after observing ChEMBL weak-label
enrichment, because doing so would turn the external analysis into model
selection rather than validation.

ChEMBL input caches, selected-chain tensors, deployment checkpoints, and the
inference implementation are SHA256-pinned into a run fingerprint. Resume is
allowed only when that fingerprint and the output hash match. Returned
archives include every full-screen shard so the aggregate can be rebuilt and
validated independently on the CPU server.

The reference selected-chain universe covers 254 of 568 reference proteins
(44.7%), whereas the full screen covers 804 of 837 proteins (96.1%). Therefore
the new selected-chain reference metrics are not directly comparable with
legacy metrics obtained from mixed full-UniProt/PDB tensors and a different
row universe. Exact-target, ligand-connectivity, double-novel, and Pfam strata
are emitted for both the reference and full screen.

Deploy-epoch provenance is fail-closed: all 15 unseen-family source fit
reports must match cohort, regime, model, seed, fold, validation fold, complete
training contract, checkpoint hash, and prediction hash. Their report hashes
and identities are incorporated into the deploy checkpoint fingerprint and
resume gate.

The 25-epoch/patience-5 optimization contract is retained for comparability.
The fraction of fits whose selected checkpoint is epoch 1 is reported for
every cohort-regime-model cell. Cells at or above one third are flagged for
small-sample stability review, but this flag is descriptive and is never used
to select seeds, epochs, or models after seeing results.
