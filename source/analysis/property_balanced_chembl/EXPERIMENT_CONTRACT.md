# Property-balanced ChEMBL deployment contract

## Purpose

This run completes the prespecified random-draw stability extension and then
tests a deploy ensemble trained on the full property-balanced cohort. ChEMBL
text-derived labels never enter training or model selection. The random-draw
extension is a secondary/SI robustness analysis, not a gate for the main
chemistry-balanced model comparison; it is nevertheless run once as frozen.

## Prespecified 30-fit extension

The first run trained selection seeds 20260824 and 20260825 only at model seed
20260817. Protein-macro AUROC for C2 minus ligand-only changed sign across the
three negative-row draws. The frozen follow-up therefore adds model seed
20260818 to those two draws for ligand-only, C2, and C3 across five family
folds: 2 draws x 3 models x 5 folds = 30 fits. This is a stability analysis; it
does not alter the primary three-seed property-balanced result and it does not
trigger further adaptive training.

## Property-balanced deploy ensemble

The deploy cohort is exactly 3,022 rows, 426 proteins, 165 family components,
1,511 allosteric rows, and 1,511 orthosteric rows. Its compressed-file SHA-256
is locked in `config.json`. Ligand-only, C2, and C3 are trained on all rows with
seeds 20260817, 20260818, and 20260819. Fixed deploy epochs are the median best
epochs from the corresponding 15 completed family-held-out fits: five epochs
for ligand-only, five for C2, and six for C3. The 45 fit-report hashes and the
derived medians are frozen in `data/C3_DEPLOY_EPOCH_AUDIT.json`; no ChEMBL
score or label participates in epoch selection. No outer-fold checkpoint is
called a deploy model.

The existing source-trained ligand-only, protein-only, and C2 deploy weights
are retained as comparators. New property-balanced checkpoints and predictions
use separate output roots and never overwrite the source analysis.

C2 remains the prespecified principal property-trained pair model, but C3 is
now included as a subset-scoped sensitivity comparator. In the completed
chemistry-balanced, family-held-out comparison, pooled AUROC was 0.677350 for
C2 and 0.611032 for C3, whereas protein-macro AUROC was 0.786591 versus
0.788002 and family-macro AUROC was 0.745684 versus 0.752653. These internal
results, not ChEMBL results, explain why C2 remains principal.

The earlier statement that the external cache had no consistent pocket mask
was correct for the legacy full-screen tensors but is no longer a reason to
omit C3. The frozen target-chain procedure was applied to the union of the
reference and full-screen UniProt accessions. Importantly, its pocket indices
are zero-based positions in the selected observed PDB-chain sequence, not in
the canonical UniProt sequence. Consequently, C3 uses new ESM3 tensors made
from those exact selected-chain FASTAs; it never applies those indices to the
legacy full-screen protein tensors.

The outcome-blind audit covers 804 of 837 full-screen proteins (96.06%). All
837 have a frozen PDB/fpocket input; zero are absent for lack of structure.
Thirty-three fail the previously frozen alignment or minimum-pocket criteria.
Reference coverage is lower: 254 of 568 proteins overall and 226 of 493
two-label-primary proteins. Of the 314 unavailable reference targets, 254 lack
a row in the frozen one-PDB-per-UniProt map, 51 have a map row without a PDB,
one lacks its coordinate file, and eight fail alignment or pocket criteria.
No new PDB-selection heuristic is introduced merely to increase reference
coverage.

A property-trained protein-only deploy ensemble is not added in this package.
Unlike ligand-only and C2, protein-only did not yet have the matched 15
property-balanced family-held-out fits needed to freeze its deploy epoch. Those
fits belong to the separately frozen 165-fit matrix-completion experiment.
After that experiment, a protein-only deploy comparator can be added by an
append-only extension with its epoch selected without reference to ChEMBL.

## External inference

Six ensembles are evaluated in the same FP32 inference workflow:

- source ligand-only, protein-only, and C2;
- property-balanced ligand-only, C2, and C3.

Forward passes remain mixed precision, but sigmoid is applied after casting
logits to FP32. Any exact UniProt--InChIKey-connectivity pair present in the
current benchmark blacklist is excluded. ChEMBL labels are joined only after
scores have been generated.

The Burggraaff 2020 two-label reference is evaluated with pooled AUPRC/AUROC,
target-macro metrics, and paired target bootstrap. The positive-only local 5A
reference is evaluated with fixed-fraction enrichment both across the pooled
universe and within each target.

Ligand-only and C2 are always reported on the complete eligible reference or
screen. C3 is never imputed for a target without a validated selected-chain
pocket tensor: its score is `NA`. Every C3 comparison and metric is computed
only on `pocket_available_matched`, and ligand-only, C2, and the source
comparators are recomputed on that identical row subset. Full-row and
pocket-matched results carry an explicit `evaluation_subset` column and must
not be compared across subsets.

Novelty is reported as exact target identity unseen, ligand connectivity
unseen, and both simultaneously. An append-only Pfam-overlap stratum is also
reported. The frozen UniProt 2026_02 cache covers the union of the 426 property
training proteins and 568 ChEMBL-reference proteins. `family_seen_property`
means that at least one exact Pfam accession assigned to the evaluated target
occurs among the Pfam annotations of the property-training proteins. Annotated
targets with no such overlap are `family_unseen_property`; targets with no
Pfam annotation or absent from the frozen cache are
`annotation_unavailable`, never unseen.

This is an exact Pfam-overlap audit, not a claim of sequence-family novelty.
The reference scope has complete frozen-accession coverage. The much larger
full-screen target universe is not locally available in this package, so full
rows absent from the cache remain explicitly annotation-unavailable until a
separately frozen cache extension is made. Adding these annotations to an
already scored output does not rerun model inference.

## Interpretation

The principal external contrast is property C2 minus property ligand-only.
Property C2 minus source C2 measures the effect of changing the training
cohort. Pooled enrichment without corresponding within-target enrichment is
interpreted as target-level prioritization, not compound-level discrimination.
Weak labels are independent evidence routes, not ground truth, and high-ranked
pairs are candidates rather than discoveries.
