# Analysis workflow

Run stages in a new workspace only after supplying and verifying the required data. Commands below identify entrypoints; use each script's argument definitions and frozen contract for the run. Historical READMEs retain historical terminology. The primary manuscript metric is classification AUROC; conditional metrics are additional analyses with distinct support.

## 01. Reference preprocessing

`allosteric_orthosteric_preprocessing_v1/scripts/parse_gtop_orthosteric_references.py` parses source annotations. `resolve_brenda_exact_structures.py` resolves cached BRENDA compounds; its verification mode does not authorize reretrieving all proteins. `build_uncontrolled_allosteric_orthosteric.py` and its validator integrate exact pairs and record conflicts. The participant/KEGG package retains the subsequent biochemical-source expansion. Manual participant decisions and later source exclusions remain frozen inputs. Do not treat a historical BioLiP coverage audit as an additional current benchmark label source.

## 02. Protein and ligand representations

The `allosteric_pair_benchmark_main` and `allosteric_pair_benchmark_broad_superset` packages contain exact input preparation, chain selection/alignment, pocket indexing and molecular-embedding interfaces. `new_pairing_control_ladder` preserves upstream dataset construction. Structure-matched chain inputs and complete UniProt inputs are different representations and must not be interchanged in an existing contract.

## 03. Benchmark training

1. Prepare and validate the base `role_complete_pair_matrix` contract.
2. Train its every-pair, protein-anchored and double-anchored random/family/ligand splits with `train_matrix.py`; aggregate and validate the results.
3. Use `explicit_double_unseen` for newly trained every-pair and protein-anchored double held-out fits.
4. Use `double_anchored_lofo` for the 47-fold small-data protocol, not the five-fold validation-selected protocol.
5. Use `ligand_anchored_benchmark` for the fourth dataset and its four splits, followed by `matrix_overview.py` to collect results with their protocol provenance.

The base 1,080 fits do not represent the full final four-by-four design. Preserve source experiment, fold, seed, dataset, selected epoch and eligible test-row identities when merging extensions. Never substitute a filtered old test set for double held-out retraining.

## 04. Sensitivity and uncertainty

`murcko_holdout_extension` performs scaffold exclusion on fixed test rows. `full_sequence_representation_pilot` isolates protein representation. `double_anchored_pooled_bootstrap/bootstrap_pooled.py` supplies paired uncertainty for the pooled small-data result. Its `--self-test-only` mode uses synthetic data. Do not infer absence of leakage or equivalence merely from a confidence interval crossing zero.

## 05. ChEMBL predictions

The base deployment functions are in `role_complete_pair_matrix`; earlier cache and selected-chain utilities are retained in the ChEMBL support packages. `ligand_anchored_chembl` adds deployment for the fourth benchmark dataset. `chembl_four_cohort_completion` supplies the final four-dataset/eight-model result, including full-screen C3 and double-anchored predictions. It reuses deployment checkpoints and is inference-only. Run `assemble.py`, the execution/validation procedures, `aggregate.py` and `bootstrap.py` in the order specified by its contract. Novelty and model comparisons require the same eligible rows; a missing representation is not a zero score.

## 06. Structural follow-up

`chembl_competitive_docking_v2`: `prepare_inputs.py` → `build_jobs.py` → `run_docking_parallel.py` → `analyze_results.py`. These are source entrypoints, not instructions to launch a new screen during release validation. Binding-competent assemblies, ligand charge and receptor preparation remain essential inputs.

## 07. BioLiP pair ranking

`biolip_bayesian_ranking_revision` contains the exact structural linkage, geometry and frozen estimator implementations. Its old standalone candidate counts are superseded by `biolip_consensus60_pipeline_v2`.

For a new compatible contract:

```bash
python analysis/biolip_consensus60_pipeline_v2/scripts/run_pipeline.py \
  --lock /path/to/new/INPUT_LOCK.json --output /path/to/new/run \
  --workers 8 --io-workers 2
python analysis/biolip_consensus60_pipeline_v2/scripts/validate_pipeline.py \
  --run-dir /path/to/new/run --verify-final
```

The primary record is the exact protein–ligand pair. A 60% per-PDB-weighted contacting-residue consensus describes its structural evidence; multiple observed locations remain nested reviewable records. Refit likelihood estimators and rerun held-out metrics at the pair level. Computational validation does not replace candidate recuration or ASD novelty checks.

## 08. Compare ranking approaches

`biolip_neural_ranking_comparison/scripts/prepare.py` freezes matched inputs. `run_comparison.py --workers 8` calculates ranking agreement and uncertainty. `test_statistics.py` exercises tie handling, resampling multiplicity and undefined cases. The comparison uses existing out-of-fold predictions; it does not perform new model training.
