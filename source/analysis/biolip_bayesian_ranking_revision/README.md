# BioLiP Bayesian ranking revision

This package revises the legacy BioLiP allosteric-likeness ranking without
modifying any prior package or `/shared_data` source.

Work is checkpointed.  A later checkpoint is not started until the preceding
checkpoint has been reviewed by the user.

## Checkpoint 1

Freeze the complete pre-audit orthosteric pair/source universe used upstream of
the final pair benchmark.  The source is the 20,668-row participant/KEGG-
augmented allosteric/orthosteric table.  Only its orthosteric rows are copied
into the checkpoint-1 pair master.  Manual BioLiP multisite dispositions are
deliberately deferred to checkpoint 2.

Run:

```bash
python3 analysis/biolip_bayesian_ranking_revision/scripts/01_freeze_orthosteric_sources.py
```

The script also verifies that every orthosteric pair used by the frozen main
and broad-superset benchmark cohorts is present in the checkpoint-1 master.

## Checkpoint 2

Apply the frozen 266-pair BioLiP manual audit at the exact
`(UniProt, full InChIKey)` grain.  The 247 retained pairs remain in the active
orthosteric reference; 13 excluded and 6 held pairs are preserved in separate
quarantine files.  Pair-wide application prevents an excluded/held BioLiP pair
from re-entering through another provenance row.

Run:

```bash
python3 analysis/biolip_bayesian_ranking_revision/scripts/02_apply_biolip_manual_audit.py
```

The active checkpoint-2 master is
`data/ORTHOSTERIC_PAIR_MASTER_AUDITED.tsv.gz`.  The complete audit crosswalk is
`data/BIOLIP_MANUAL_AUDIT_PAIR_CROSSWALK.tsv.gz`, and no quarantined pair is
deleted from the package.

## Checkpoint 3

Inventory existing BioLiP observation, coordinate, SIFTS, mapping, and pilot
site-cluster assets before any new mapping is started.  The script enumerates
only four explicitly named local cache directories and reads named manifests
and completed logs; it does not recursively scan the large legacy BioLiP
coordinate repository.

Run with the workspace Python that provides DuckDB:

```bash
miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/03_inventory_existing_structure_assets.py
```

The principal outputs are
`data/CHECKPOINT3_CACHE_PDB_MANIFEST.tsv.gz`,
`data/CHECKPOINT3_EXISTING_MAPPING_SUMMARY.tsv`, and
`data/CHECKPOINT3_REUSE_CLASSIFICATION.tsv`.  This checkpoint performs no new
structure download, residue mapping, feature extraction, or site clustering.

## Checkpoint 4

Rebuild the BioLiP table at exact observed-site grain and map every reported
binding residue to UniProt coordinates.  The source grain is preserved as
`PDB + receptor chain + binding-site code + CCD + ligand chain + ligand serial
+ ligand author residue ID`; rows are not collapsed to protein–ligand pairs.

Run in order:

```bash
miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/04a_build_exact_observation_master.py
miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/04b_validate_cached_author_mapping.py
miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/04c_finalize_checkpoint4.py
```

The primary mapping path is BioLiP observed-sequence position to the exact CIF
polymer position and then through SIFTS to UniProt.  Canonical UniProt sequence
alignment is an independent consistency check and controlled rescue.  Existing
local mmCIF/SIFTS pairs provide a second check from raw author residue numbers.

The full 989,058-row observation table is
`data/BIOLIP_EXACT_OBSERVATION_MASTER.tsv.gz`; the separate 989,058-row final
eligibility ledger is `data/BIOLIP_SITE_MAPPING_ELIGIBILITY.tsv.gz`.  Final
site-mapping eligibility is 927,697 rows.  This checkpoint does not yet form
site clusters or compute the BioLiP ranking.

## Checkpoint 5

Link exact observed BioLiP sites to the two reference classes without
promoting pair-level biochemical annotations to exact structural sites.

Run and validate:

```bash
miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/05_link_exact_site_references.py
python3 \
  analysis/biolip_bayesian_ranking_revision/scripts/05_validate_exact_site_references.py
```

The allosteric linkage requires exact UniProt, PDB, CCD, ligand chain and
ligand author-residue identity plus overlap between a chain-qualified
AlloBench site residue and the exact BioLiP receptor-chain binding residues.
This retains 2,053 site-supported observations; 2,028 pass checkpoint-4
mapping.  The manually retained 247 BioLiP orthosteric pairs link to 4,093
exact observations, of which 4,090 pass mapping.

After excluding one observation with conflicting allosteric and biochemical
role annotations, the frozen reference contains 6,117 exact observations:
2,027 allosteric and 4,090 orthosteric.  Independent identity validation found
zero missing observations, zero identity mismatches, and zero mapping-ineligible
rows in the unified table.  See
`validation/CHECKPOINT5_VALIDATION.json` and
`reports/CHECKPOINT5_EXACT_SITE_REFERENCE_LINKAGE.md`.

GtoP, KLIFS, BRENDA, ChEBI, KEGG, and UniProt remain available as broad
protein–ligand role annotations.  They are not treated as exact binding-site
labels unless the corresponding BioLiP observation is among the manually
audited exact controls.

## Checkpoint 6

Compute seven exact-structure distance definitions on the checkpoint-5 reference
observations. No distance is selected until the checkpoint-6 comparison is reviewed.

Run and validate:

```bash
CHECKPOINT6_WORKERS=16 miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/06_compare_and_freeze_distance.py
miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/06_validate_distance_definition.py
```

The initial bounded run found 804 exact-reference PDB IDs missing from all
named local caches.  These were fetched by exact ID with
`06a_fetch_missing_reference_coordinates.py`; no coordinate directory was
recursively scanned.  The final run preserves all 6,117 reference rows and
computes valid geometry for 6,054 (98.97%): all 4,090 orthosteric observations
and 1,964/2,027 allosteric observations.

The candidates include heavy-atom and C-alpha minimum distances, candidate-site
centroid distance, and two distances from the exact ligand heavy-atom centroid to
the orthosteric-site heavy-atom or C-alpha centroid. All seven are reported on the
same 5,798-row comparison universe; no ranking cutoff or preferred definition is
selected at this checkpoint.

AlloBench reports 1,708 geometry-computable observations as being at a distinct
site, but 669 share at least one mapped binding residue with the supplied active
site.  The source annotation is preserved, and these rows are exposed in
`data/CHECKPOINT6_REPORTED_DISTINCT_RESIDUE_OVERLAP_AUDIT.tsv.gz` rather than
silently relabeled or removed.  See
`validation/CHECKPOINT6_VALIDATION.json`,
`manifests/CHECKPOINT6_DISTANCE_SPEC.json`, and
`reports/CHECKPOINT6_DISTANCE_DEFINITION.md`.

After user review, the selected ranking distance was frozen separately in
`manifests/CHECKPOINT6_DISTANCE_SELECTION.json` as the distance between the exact
ligand heavy-atom centroid and the orthosteric-site residue C-alpha centroid.

## Checkpoint 7

Compare the complete set of 15 non-empty subsets of four ranking features:
selected structural distance, molecular weight, calculated LogP, and aromatic
ring count. Tanimoto similarity is excluded from the ranking score and reserved
for later substrate/cofactor-similarity QC.

Run and validate:

```bash
analysis/structure_known_candidate_rebuild/tools/rdkit_env/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/07a_compute_reference_ligand_descriptors.py
miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/07_compare_ranking_features.py
miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/07_validate_feature_ablation.py
```

The frozen comparison contains 5,798 exact observations and 3,549 exact
site-ligand signatures. A direct monotonic distance rank is reported alongside
the 15 likelihood-ratio feature subsets because the KDE transformation itself
can alter the ordering of very distant observations. All checkpoint-7 metrics
are descriptive full-reference diagnostics. No preferred feature subset is
selected before the protein/family-held-out evaluation in checkpoint 8.

### Checkpoint 7 likelihood-estimator addendum

The same 5,798 observations and 15 feature subsets were also evaluated with
three explicitly separated likelihood estimators, plus the direct-distance
baseline:

- quantile-bin Naive Bayes with Jeffreys 0.5 smoothing and no manually imposed
  likelihood-ratio limit;
- the legacy KDE convention (distance limited to 0.01--100 and each chemistry
  feature limited to 0.1--10);
- KDE with every feature limited to 0.01--100, matching the first checkpoint-7
  implementation;
- the selected structural distance ranked directly, without a Bayesian
  transformation.

Run and validate:

```bash
miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/07b_compare_likelihood_estimators.py
miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/07b_validate_likelihood_estimators.py
```

This produces 46 ranking definitions: three estimators times 15 feature subsets
plus one direct-distance baseline. Independent validation recomputes the bin
counts, smoothed likelihood ratios, KDE components, capped-row counts, component
sums, and all reported metrics. The direct distance has the highest descriptive
within-protein macro AUROC (0.9823); the highest descriptive pooled AUROC is the
quantile-bin `D+LP+AR` score (0.9824). The contrast is not used to select a final
method because chemistry-heavy pooled performance can reflect protein, source,
or ligand-composition differences. Method selection remains deferred to
checkpoint 8 held-out evaluation.

## Checkpoint 8

Evaluate all 46 frozen ranking definitions under protein-held-out and
Pfam-family-held-out five-fold splits. Every likelihood transformation is fitted
only on training-fold rows; the direct-distance ranking requires no fitting.

Run and validate:

```bash
miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/08a_prepare_heldout_splits.py
miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/08a_validate_heldout_splits.py
miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/08_evaluate_heldout_rankings.py
miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/08_validate_heldout_rankings.py
```

UniProt release 2026_02 provides Pfam annotations for all 370 reference
proteins. The 481 Pfam accessions form 157 connected family components; the
largest contains 31 proteins. Protein and family-component overlap between
training and test is zero. Each split regime produces 5,798 OOF predictions
for all 46 rankings.

The validated primary statistic is site-collapsed within-protein macro AUROC,
supplemented by MRR, normalized first-allosteric rank, Recall@1/3/5, pooled
metrics, fold-level metrics, and 10,000 paired cluster-bootstrap replicates.
No single ranking is selected. The direct distance is the strongest exact-site
retrieval rule, while different KDE feature combinations lead among the
fold-fitted rankings depending on whether the goal is within-protein AUROC,
first-candidate retrieval, or pooled cross-protein ranking. See
`reports/CHECKPOINT8_VALIDATED_RESULTS.md` and
`validation/CHECKPOINT8_VALIDATION.json`.

## Checkpoint 9

Apply all 46 frozen ranking definitions to BioLiP observations that were not
used as exact allosteric or broad orthosteric reference pairs. Structural
ranking is limited to proteins with at least one exact manually audited
orthosteric-site observation.

Run in order:

```bash
miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/09a_prepare_unlabelled_universe.py
miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/09b_fetch_candidate_assets.py \
  --run --workers 20
CHECKPOINT9_WORKERS=48 miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/09c_compute_candidate_geometry.py
analysis/structure_known_candidate_rebuild/tools/rdkit_env/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/09d_compute_candidate_descriptors.py
miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/09e_score_unlabelled_rankings.py
miniconda3/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/09_validate_unlabelled_rankings.py
```

The frozen pre-geometry universe contains 19,920 observations. Exact targeted
coordinate retrieval and mapping produce 18,859 geometry-complete observations,
2,890 exact protein-ligand pairs, and 5,150 protein-ligand-site ranking rows on
153 proteins. Repeated mapped binding-residue sets are grouped by complete
linkage with a minimum pairwise Jaccard similarity of 0.50; single-linkage
chaining is not used.

The principal public table is
`data/CHECKPOINT9_SITE_RANKINGS_LONG.tsv.gz`. Pair-level and observation-level
tables are also retained. Direct distance is continuous; 8, 10, and 12 A flags
are descriptive filters rather than automatic allosteric labels. Scores are
uncalibrated rankings, not probabilities. See
`reports/CHECKPOINT9_VALIDATED_RESULTS.md` and
`validation/CHECKPOINT9_VALIDATION.json`.

## Checkpoint 10

Apply post-ranking chemical-role and structure-context checks without fitting
or changing any ranking model:

```bash
analysis/structure_known_candidate_rebuild/tools/rdkit_env/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/10_filter_ranked_candidates.py
analysis/structure_known_candidate_rebuild/tools/rdkit_env/bin/python \
  analysis/biolip_bayesian_ranking_revision/scripts/10_validate_candidate_filtering.py
```

The conservative review filter requires a spatially separate site, a
substantial single-component carbon-containing ligand, no broad AlloBench
protein-ligand link, and Morgan-fingerprint similarity below 0.70 to known
same-protein orthosteric compounds. Low similarity is neutral rather than
positive allostery evidence. Four prespecified ranking views are then used to
form a transparent agreement filter; a final structural-repeat filter requires
the site to occur in at least two PDB entries.

The validated funnel is 5,150 site rows -> 1,768 spatially separate -> 1,448
after the compound filter -> 1,343 after known-role checks -> 79 ranking-agreed
sites -> 13 repeatedly observed sites representing six protein-ligand pairs.
Manual inspection of the six actual structure contexts excludes one
experimental additive (1PS) and retains five pairs for database-expansion
review. These five are not claimed to be newly discovered mechanisms. See
`reports/CHECKPOINT10_FILTERED_CANDIDATES.md` and
`validation/CHECKPOINT10_VALIDATION.json`.
