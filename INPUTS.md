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

Training additionally requires frozen `matrix_row_fold`, `matrix_family_fold` and `matrix_ligand_fold` columns with assignments 0–4, and `matrix_evaluation_eligible`. Where the Both-class ligands extension supplies completed `run_row_fold` and `run_family_fold` columns, these supersede the corresponding shared assignments. Fold assignments can differ between dataset-specific experiments; supply their frozen table for exact reproduction. Unassigned rows must be assigned under the documented split protocol before fitting. Dataset construction does not infer new Pfam annotations or select folds from model performance.

`build_datasets.py` writes all four dataset tables and counts. Supply `matrix_evaluation_eligible` already at this stage: it identifies the frozen 4,637-row Both-class proteins reference within the broader input. Preserve that supplied membership to reproduce the published evaluation population. Both-class proteins & ligands is constructed from this frozen reference. The optional `murcko_scaffolds_json` field contains a union of nonempty atom/bond-specific Murcko scaffolds per connectivity. Acyclic compounds are retained under connectivity exclusion, without placing every acyclic molecule in a shared empty-scaffold group.

Protein inputs are tensors of shape residues × 1,536; ligand inputs are heavy atoms × 512. `.pt` files may contain one tensor or a dictionary with a single compatible tensor. Paths resolve relative to `--tensor-root`. `pocket_indices.json` maps a UniProt accession to zero-based indices in its selected chain tensor. These are not canonical UniProt residue numbers or PDB author residue IDs. Prepared selected-chain tensors and pocket masks must agree exactly. Biological assemblies are not represented by a single chain tensor.

Outputs from training: `checkpoint.pt`, `fit.json` and held-out `predictions.tsv.gz`. Each job represents one model, one seed and one outer fold. Evaluation combines exactly three seed predictions per row with matching labels, identities and folds. To evaluate a particular published population, restrict prediction inputs to that frozen row set before aggregation.

For the 47-fold double held-out protocol on Both-class proteins & ligands, invoke `train_models.py` once for each family with `--dataset both_class_proteins_ligands --regime double_held_out --leave-family FAMILY_ID`. This uses epoch 25 and has no validation partition. Pfam held-out evaluation on this same dataset remains five-fold. For deployment, supply all 15 corresponding Pfam-held-out `fit.json` files using `--deployment-fit-reports`; the model is then initialized afresh and trained on the complete supplied dataset for their median best epoch. Historical CLI values and report identifiers remain accepted; see the compatibility table in README.md.

## 3. External screening

ChEMBL pair inputs use `record_id`, `uniprot`, `full_inchikey`, `connectivity_key`, `ligand_embedding_path` and `protein_embedding_path`. Scores do not require external labels. Supply the matching training pair table to exclude previously observed exact pairs. `screen_chembl.py` accepts three deployment checkpoints from the same dataset and architecture, plus the same pocket mapping and tensor root used by those representations.

An optional annotation file contains `record_id`, `published_label` and/or `keyword_positive`. `published_label` is the binary annotation from the published text-mining reference; `keyword_positive` is a separately prepared 0/1 assay-text search result. Keyword hits do not fill missing published labels. These fields are joined after scores are produced.

Output rankings retain seed scores, ensemble mean/SD, missing-input status, novelty flags, global rank and within-protein rank. Reference metrics use labelled available rows. Model comparisons require the intersection of available rows across the compared models; do not compare metrics calculated on different populations. Pfam novelty annotations require a separately complete family reference and must not equate missing annotations with unseen families.

## 4. Structural follow-up

`structural_followup.py` consumes a prespecified job table with:

`job_id`, `system_id`, `ligand_id`, `ligand_role`, `pocket_id`, `receptor`, `ligand`, `center_x/y/z`, `size_x/y/z`, `scoring`, `seed`, `exhaustiveness`, `cpu`, `num_modes`.

Coordinates and box sizes are in Å. Receptor and ligand paths identify prepared PDBQT files. Each ligand/site combination must have five seeds for each of `vina` and `vinardo`; ligand roles are candidate, comparison, background or reference. Vina is supplied with `--vina`. Receptor biological assembly, cofactors, protonation, ligand stereochemistry and charge, and site-box definitions are preparation inputs requiring structural review.

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

## 7. Time-split validation

The dated model-ready table adds `role_year` to the fields in section 2. Keep the
associated source identifiers and date evidence with each record. Missing years
are represented by NA, not by zero. The build command reconstructs all four
datasets from earlier records and writes a shared later curated test:

```bash
python time_split.py build --dated-pairs inputs/dated_pairs.tsv.gz --cutoff 2015 --output results/time_split
python time_split.py fit --pairs results/time_split/both_class_proteins.tsv.gz --future results/time_split/future_curated.tsv.gz --pockets inputs/pocket_indices.json --tensor-root inputs/tensors --dataset both_class_proteins --model c2 --seed 20260817 --device cuda:0 --output results/time_fits/c2_seed20260817
```

Repeat fits for the four datasets, eight models and three seeds. Output predictions
use `score`, `dataset`, `model`, `seed` and `outer_fold`, and can be combined by
`evaluate_models.py`. The selected epoch is determined only from earlier data.

For external ChEMBL recovery, supply a common metadata table with `main_row_id`,
`uniprot` and 0/1 columns identifying the assay-keyword and reviewed literature
sets. Prediction tables contain `main_row_id`, `cohort`, `model`, `score`; scores
are the three-seed ensemble means. All conditions must contain identical rows.
An indicator of zero means absence from that evidence set, not a negative
binding-role label. The frozen external population already excludes earlier and
undated protein–connectivity pairs.

`time_split.py infer` applies three checkpoints produced by `time_split.py fit`
to external rows with `main_row_id`, `first_pair_year`, protein/ligand identities
and individual embedding paths. Supply `--blocked-pairs` with the complete
`uniprot`, `connectivity_key` list excluded by the time-split design, including
earlier and undated records from all sources, not only the selected cohort.

```bash
python time_split.py infer --pairs inputs/later_chembl_pairs.tsv.gz --blocked-pairs inputs/earlier_and_undated_pairs.tsv.gz --checkpoints results/time_fits/c2_seed20260817/checkpoint.pt results/time_fits/c2_seed20260818/checkpoint.pt results/time_fits/c2_seed20260819/checkpoint.pt --pockets inputs/chembl_pocket_indices.json --tensor-root inputs/tensors --output results/time_screen/c2
```

Intersect scored rows across all compared models before recovery evaluation.

```bash
python ranking_recovery.py --metadata inputs/later_chembl_evidence.tsv.gz --predictions inputs/later_chembl_scores.tsv.gz --evidence-columns allosteric_assay_keyword literature_supported_function --output results/later_recovery
```

## 8. Matched docking and site correspondence

`docking_sites.py select` uses one ranking condition at a time. Its ranking input
contains `OOD_Row_ID`, `uniprot`, `ligand_chembl_id`, `connectivity_key`,
`model_score`, `MW`, `LogP`, `TPSA`, `has_allosteric_text` and
`has_orthosteric_text`. The two text fields are explicit 0/1 or true/false values.
Rank percentiles are calculated before chemical-preparation exclusions.

The chemical-preparation table has one row per `ligand_chembl_id`, with `status`
(`complete` for usable records), `computed_connectivity`, `source_formal_charge`,
`ph74_formal_charge`, `smiles` and `ph74_smiles`. Populate it from the same
standardization and pH 7.4 preparation used for the study. Failed preparations
remain in the ranking population but cannot be selected. The protein-panel file
lists eligible UniProt accessions separated by whitespace.

```bash
python docking_sites.py select --rankings inputs/ranking_condition.tsv.gz --chemistry inputs/prepared_compounds.tsv.gz --proteins inputs/protein_panel.txt --count 1000 --per-protein 100 --output results/docking_selection
```

Use the archived per-protein cap for each experiment; the additional-protein
panel used 10 matches per protein and condition. `selected_pairs.tsv` preserves
the original record IDs and labels high-ranked compounds `candidate` and
low-ranked matched compounds `comparison`. These roles are ranking groups, not
experimental binding labels. Use `structural_followup.py` with prepared PDBQT
inputs and the complete job table to execute docking.

For site classification, supply:

- `--systems`: the complete-case compound table with `system_id`, `match_id`,
  `uniprot`, `role`, `heavy_atoms`, `cohort`, `model` and `reference_kind`.
- `--jobs`: the complete planned grid with `job_id`, `system_id`, `pocket_id`,
  `scoring`, `seed`, `pose_path`, `pose_sha256`. Every retained matched comparison
  must include both compounds and every planned box/scoring/seed combination.
- `--references`: JSON keyed by UniProt accession. Each entry contains
  `coordinates_frame: "same reviewed docking assembly"`, `reference_kind`,
  `receptor_atom_xyz`, `receptor_atom_residue_keys`, `target_atom_xyz`,
  `target_atom_residue_keys`, `primary_xyz`, `primary_contact_keys`, and
  `prepared_receptor` (PDBQT path). Each `boxes` entry has `pocket_id`, `center`
  and `size`. Each `allosteric_sites` entry has `site_id`, `xyz` and `contact_keys`.
- `--root`: base directory for pose and receptor paths. Coordinates are in
  angstroms; residue keys include assembly-copy identity, residue number and name.

Reference entries can set `strict_distinct_evaluable` and `known_site_evaluable`
independently to false, with corresponding `*_exclusion_reason` fields. A limited
catalytic-residue reference cannot define full orthosteric-site separation.
Transferred sites must already be mapped into the receptor frame. Keep excluded
comparisons and their preparation/completion reasons in the accompanying data.

```bash
python docking_sites.py classify --systems inputs/complete_compounds.tsv --jobs inputs/pose_jobs.tsv --references inputs/site_geometry.json --root inputs/docking --output results/sites
python docking_sites.py summarize --endpoints inputs/all_conditions_endpoints.tsv.gz --output results/docking_comparison
```

The classifier reports results without an energy cutoff and at −5 kcal/mol by
default; `--energy-cutoffs` selects additional thresholds if required. Summary
inputs may also be archived endpoint tables, with `cohort` and `model` added.
Match/system IDs must be unique within a condition; prefix them when joining
separate docking batches. Protein-cluster resampling does not treat seeds or
search boxes as independent compounds. Do not merge overlapping selected pairs
from different training conditions into a single independent sample.

## 9. Residue-attention analysis

Use the frozen pair annotations, site masks and C2/C3 attention maps from the
Both-class ligands, Pfam held-out experiment. `--pairs` is a JSON list with
`pair_id`, `uniprot`, `full_inchikey`, `family`, `binary_label`, `masks` and
`mask_status`. Masks include `fpocket_union` and `cross_structure_consensus60`;
the latter is used for orthosteric pairs when its status is `available`.
`--allosteric-masks` is keyed by pair ID, with `status` and `mask` for the exact
AlloBench ligand site. All masks use the input protein-residue indexing.

`--maps` is an NPZ with one normalized residue vector per
`<model>__<pair_id>__mean`, averaged across the three seeds. Optional
`--single-maps` contains head-averaged vectors named `<model>__<seed>__<pair_id>`.
These maps are extracted from the corresponding held-out checkpoints, not from
deployment weights. The module reports enrichment against uniform attention,
protein/family-cluster summaries and attention at the same site when another
ligand of that protein is supplied.

```bash
python attention_analysis.py --pairs inputs/attention_pairs.json --allosteric-masks inputs/allosteric_site_masks.json --maps inputs/ensemble_attention.npz --output results/attention
```
