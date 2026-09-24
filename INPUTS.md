# Inputs and outputs

Files are tab-separated UTF-8 tables, optionally gzip-compressed. `NA` denotes missing information. Pair labels are `binary_label=1` for an allosteric assignment and `0` for an orthosteric assignment. A missing label is not a negative label.

## 1. Reference annotations

`prepare_references.py` consumes identity-resolved source exports with:

`source`, `source_record_id`, `uniprot`, `full_inchikey`, `canonical_smiles`, `binary_label`.

Accepted sources are ASD, GtoPdb, KLIFS, BRENDA, UniProt, ChEBI and KEGG. Source-specific assessment of binding role, protein identity, species and chemical standardization must already have been performed. In particular, enzyme EC membership or fuzzy compound-name similarity cannot establish an exact protein–ligand record. Canonical SMILES must reproduce the full InChIKey. Salt, fragment, charge and stereochemistry decisions must follow the source preparation used for the dataset; this integrator does not silently revise them.

Outputs: one exact-pair table, source lineage, class conflicts and unresolved structures. Existing row IDs and frozen split tables must be retained when reproducing a published benchmark rather than replaced with newly generated IDs.

## 2. Model-ready pairs and splits

The model-ready table requires:

`main_row_id`, `uniprot`, `full_inchikey`, `connectivity_key`, `binary_label`, `canonical_smiles`, `family_component_id`, `ligand_embedding_path`, `protein_embedding_path`.

Training additionally requires frozen `matrix_row_fold`, `matrix_family_fold` and `matrix_ligand_fold` columns with assignments 0–4, and `matrix_evaluation_eligible`. Where the ligand-anchored extension supplies completed `run_row_fold` and `run_family_fold` columns, these supersede the corresponding shared assignments. Fold assignments can differ between dataset-specific experiments; supply their frozen table for exact reproduction. Unassigned rows must be assigned under the documented split protocol before fitting. Dataset construction does not infer new Pfam annotations or select folds from model performance.

`build_datasets.py` writes all four dataset tables and counts. Supply `matrix_evaluation_eligible` already at this stage: it identifies the frozen 4,637-row protein-anchored reference within the broader input. Re-selecting every two-role protein from the broader input would instead retain 4,645 rows, including eight later-added rows from one additional protein, and would change the published evaluation population. Double anchoring starts from the frozen protein-anchored reference. The optional `murcko_scaffolds_json` field contains a union of nonempty atom/bond-specific Murcko scaffolds per connectivity. Acyclic compounds are retained under connectivity exclusion, without placing every acyclic molecule in a shared empty-scaffold group.

Protein inputs are tensors of shape residues × 1,536; ligand inputs are heavy atoms × 512. `.pt` files may contain one tensor or a dictionary with a single compatible tensor. Paths resolve relative to `--tensor-root`. `pocket_indices.json` maps a UniProt accession to zero-based indices in its selected chain tensor. These are not canonical UniProt residue numbers or PDB author residue IDs. Prepared selected-chain tensors and pocket masks must agree exactly. Biological assemblies are not represented by a single chain tensor.

Outputs from training: `checkpoint.pt`, `fit.json` and held-out `predictions.tsv.gz`. Each job represents one model, one seed and one outer fold. Evaluation combines exactly three seed predictions per row with matching labels, identities and folds. To evaluate a particular published population, restrict prediction inputs to that frozen row set before aggregation.

For the 47-fold double-anchored protocol, invoke `train_models.py` once for each family with `--dataset double_anchored --regime double_unseen --leave-family FAMILY_ID`. This uses epoch 25 and has no validation partition. Other regimes use validation-based selection. For deployment, supply all 15 corresponding Pfam-held-out `fit.json` files using `--deployment-fit-reports`; the model is then initialized afresh and trained on the complete supplied dataset for their median best epoch.

## 3. External screening

ChEMBL pair inputs use `record_id`, `uniprot`, `full_inchikey`, `connectivity_key`, `ligand_embedding_path` and `protein_embedding_path`. Scores do not require external labels. Supply the matching training pair table to exclude previously observed exact pairs. `screen_chembl.py` accepts three deployment checkpoints from the same dataset and architecture, plus the same pocket mapping and tensor root used by those representations.

An optional annotation file contains `record_id`, `published_label` and/or `keyword_positive`. `published_label` is the binary annotation from the published text-mining reference; `keyword_positive` is a separately prepared 0/1 assay-text search result. Keyword hits do not fill missing published labels. These fields are joined after scores are produced.

Output rankings retain seed scores, ensemble mean/SD, missing-input status, novelty flags, global rank and within-protein rank. Reference metrics use labelled available rows. Model comparisons require the intersection of available rows across the compared models; do not compare metrics calculated on different populations. Pfam novelty annotations require a separately complete family reference and must not equate missing annotations with unseen families.

## 4. Structural follow-up

`structural_followup.py` consumes a prespecified job table with:

`job_id`, `system_id`, `ligand_id`, `ligand_role`, `pocket_id`, `receptor`, `ligand`, `center_x/y/z`, `size_x/y/z`, `scoring`, `seed`, `exhaustiveness`, `cpu`, `num_modes`.

Coordinates and box sizes are in Å. Receptor and ligand paths identify prepared PDBQT files. Each ligand/site combination must have five seeds for each of `vina` and `vinardo`; ligand roles are candidate, background or reference. Vina is supplied with `--vina`. Receptor biological assembly, cofactors, protonation, ligand stereochemistry and charge, and site-box definitions are preparation inputs requiring structural review. This stage does not invent a receptor assembly or infer an allosteric label from a docking score.

Outputs retain poses, logs, input hashes and site-score summaries. Use deposited ligand coordinates rather than redocking when illustrating an existing BioLiP co-crystal observation.

## 5. BioLiP observations

Prepared reference and candidate tables require:

`observation_id`, `pdb_id`, `uniprot`, `full_inchikey`, `binding_uniprot_positions`, `distance`, `molecular_weight`, `clogp`, `aromatic_ring_count`.

Reference observations also require `binary_label`, `family_component_id`, `protein_fold`, `family_fold`. Contacts are semicolon-separated canonical UniProt positions; empty contacts remain in the frequency denominator. Source receptor-chain and ligand-instance identifiers should accompany the table and its archived observation records. Distances are measured from ligand heavy-atom centroid to a known orthosteric-site Cα centroid in a valid common coordinate frame, using the nearest defined site for candidate observations. The ranking script consumes these measured distances; it does not reconstruct geometry from residue numbers alone.

Optional candidate review fields are `min_heavy_distance` (Å), `residue_overlap` (count), `compound_pass` (0/1) and `known_role_pass` (0/1). The latter flags must be supplied from the frozen compound/site/role review; unknown values fail the automated pass. Similarity to known orthosteric compounds is a review criterion, not an added likelihood feature. This concise code contains pair reconstruction and ranking; source-specific structural linkage, biological-assembly review and detailed cross-structure coordinate audits remain upstream preparation.

Outputs include aggregated pairs, residue contact frequencies, held-out reference scores, candidate rankings and, when the review fields are supplied, a review funnel. Every candidate pair remains present even when complete ranking inputs are unavailable.

## 6. Ranking comparison

Supply one neural dataset/regime's seed-ensemble scores and one structural held-out regime, with `uniprot`, `full_inchikey`, `binary_label`, `family_component_id`, `score`, and respectively `model` or `method`. Neural input includes all eight models. The four default structural methods are direct distance, KDE distance + aromatic rings, KDE distance + molecular weight and quantile-bin distance + aromatic rings.

Matching uses exact pair identities and consistent reference labels. All selected methods must have scores for the same pair. Output tables contain paired scores, per-protein agreement, cluster-bootstrap intervals and differences against ligand-only. The two source family partitions are joined for resampling; held-out score agreement is not a new external validation experiment.
