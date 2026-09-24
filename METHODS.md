# Computational methods

## Pair identities and dataset construction

An exact pair combines UniProt accession and full InChIKey. Duplicate source records can support one pair; contradictory class assignments are quarantined. Connectivity keys are used for chemical-identity holdout, while full InChIKeys define ligand role completeness. Protein anchoring retains the frozen evaluation population; ligand anchoring selects two-role ligands from the broader every-pair set. Double anchoring starts from the protein-anchored set and repeatedly removes single-role proteins and ligands until no further rows are removed.

## Model inputs and architectures

Protein representations are fixed 1,536-dimensional ESM3 open-small per-residue embeddings of a selected structure-matched chain. Ligand representations are fixed 512-dimensional Uni-Mol heavy-atom embeddings. D-series models use the union of the five highest-ranked fpocket pockets of that chain. The compact code starts from these prepared representations.

Ligand-only uses masked atom averaging. Protein-only uses learned attention pooling. C1/D1 concatenate independently pooled ligand/protein representations. C2/D2 use a pooled ligand query attending to projected protein residues. C3/D3 use atom-to-residue and residue-to-atom attention with masked maximum pooling. Attention heads use a 256-dimensional projection and four heads; dropout is 0.30. Fusion includes the ligand and protein vectors, their elementwise product and their absolute difference. At most 120 ligand atoms are used; long full-chain inputs follow the recorded 4,096-residue uniform-index limit. Pocket indices refer to the original tensor before full-chain subsampling.

## Splits, fitting and deployment

Frozen five-fold assignments define test fold f, validation fold (f+1) mod 5 and the remaining training folds. Every-pair evaluation uses the protein-anchored test population, while the extra every-pair rows can contribute to training where eligible. Double held-out keeps the Pfam test fold, removes its ligand connectivities from validation, and removes connectivities in test or retained validation from training. The optional scaffold sensitivity additionally removes all overlapping nonempty Murcko variants, without changing test rows.

AdamW uses learning rate and weight decay 1e-4. Models run for at most 25 epochs, with five epochs of early-stopping patience. Weights equalize protein–label groups and then label totals using only the training partition. Five-fold loss divides the weighted loss sum by minibatch weight sum. The 47-fold double-anchored protocol uses mean weighted loss and the fixed final epoch 25. GPU forward passes use mixed precision, with sigmoid applied after FP32 conversion.

Checkpoint selection uses validation within-protein AUROC for ligand-only, within-ligand AUROC for protein-only, and the split-specific conditional statistic for pair models. Random-split pair models require both conditional statistics. Fewer than eight eligible groups triggers symmetric average-precision fallback. The general-dataset implementation groups ligand selection statistics by connectivity; the ligand-anchored implementation uses full InChIKey, as in its source experiment. This distinction is explicit in the training code. Reported within-exact-ligand evaluation remains available separately.

Deployment starts from initialization and uses the integer median best epoch of the corresponding 15 Pfam-held-out fits. External outcomes do not select an epoch. The deployment reports used by the concise runner are its own fit records; conversion of older execution-specific checkpoint reports is an input-preparation task.

## Evaluation and external rankings

Three seed scores are averaged before evaluation. Pooled AUROC and average precision summarize the complete selected population. Conditional AUROCs use two-class groups within the same outer fold and report eligible support. Paired cluster resampling preserves repeated-draw multiplicity and compares models on common rows. Undefined bootstrap replicates are recorded; the general evaluator requires at least 95% defined replicates.

ChEMBL reference-label discrimination and allosteric assay-keyword enrichment are separate evaluations. The top 0.1% contains ceil(0.001 × population size) records. Within-protein selection retains at least one record per eligible protein; eligible proteins contain at least one keyword hit. Ties use record identifiers. Single-input external predictions are evaluated once per identical input, so protein-only ties within a protein are retained. These ties have no ligand-discrimination interpretation.

## Structural rankings

BioLiP observations are grouped by exact protein–ligand pair. Features use a median within each PDB and then a median across PDBs. Contact frequencies average observations within a PDB and then weight PDB entries equally. Frequencies at least 0.60 define the contact summary; empty annotations remain in the denominator. Different observations and ligand instances remain traceable through their IDs.

Four features—orthosteric-site distance, molecular weight, cLogP and aromatic-ring count—form 15 nonempty combinations. Three estimators yield 45 likelihood rankings, supplemented by direct distance. Quantile-bin naïve Bayes uses cube-root bin counts and Jeffreys 0.5 smoothing. Gaussian KDE uses Scott bandwidths, clips evaluation values to the training feature range and adds 1e-12 to density values. Aromatic rings use categorical frequencies with an unseen-category bin. Feature-specific KDE caps are 0.01–100 for distance and 0.1–10 for chemistry; common-scale caps are 0.01–100 for all components. Scores sum component log likelihood ratios. Bins and densities are fitted inside each held-out training fold. Candidate scoring refits on the complete reference. Incomplete candidates retain NA scores.

The optional review funnel applies spatial, compound and known-role checks before agreement across four ranking views and repeated-PDB support. These flags organize review; they do not establish novelty or a functional mechanism. The direct-distance reference result is linked to how known sites were selected and should not be interpreted as prospective allostery accuracy.

## Scope of this code release

The eight modules expose the core scientific operations with explicit table inputs. Database-specific extraction, manual identity decisions, pretrained encoder execution, structure preparation, environment transfer, manuscript rendering and web deployment are outside this concise source release. The supplied input schemas make these boundaries explicit. The publication's frozen input tables and representations are necessary to reproduce its exact numerical populations.
