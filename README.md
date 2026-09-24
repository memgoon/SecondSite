# SecondSite

### Benchmarking allosteric–orthosteric classification and ranking candidate protein–ligand interactions

**Heesu Jeong, Hanshin David Shin, So Yun Jhang, Lina Yi, Jeongwoo Seo and Heebal Kim**  
Seoul National University · Correspondence: Heebal Kim

[Web resource](https://secondsite.snu.ac.kr) · [Analysis workflow](#analysis-workflow) · [Input formats](INPUTS.md) · [Methods](METHODS.md)

## Research question

Can a model distinguish an allosteric protein–ligand interaction from an orthosteric one, and how much of its performance depends on familiar proteins or compounds?

SecondSite examines this question using complementary reference datasets and held-out evaluation settings. We combine curated allosteric records with primary-ligand, substrate, cofactor and catalytic-site annotations, then compare models that receive ligand information, protein information or both. We also investigate whether these models can prioritize interactions in ChEMBL and whether experimentally observed binding positions in BioLiP provide complementary rankings for follow-up research.

This repository contains the core analysis in eight Python files. Each file implements a scientific stage, with shared model and statistical functions imported directly where needed.

## Benchmark design

Allosteric annotations come from ASD. Orthosteric references come from GtoPdb, KLIFS, BRENDA, UniProt, ChEBI and KEGG. Records are linked by exact UniProt accession and standardized full InChIKey; conflicting binding-role assignments are kept out of the benchmark. The integrated reference contains 20,640 pairs across 7,384 proteins. Of these, 6,854 pairs have the representations required by all eight models.

| Dataset | Selection rule | Pairs | Proteins | Ligands |
|---|---|---:|---:|---:|
| Every-pair | All model-ready pairs | 6,854 | 940 | 4,418 |
| Protein-anchored | Frozen evaluation set; each protein has both binding roles | 4,637 | 426 | 2,910 |
| Ligand-anchored | Each retained ligand has both binding roles | 1,365 | 543 | 122 |
| Double-anchored | Both requirements hold after iterative filtering | 395 | 98 | 40 |

The protein-anchored set preserves the frozen evaluation population; the broader every-pair set adds training records. Ligand anchoring is selected from the every-pair dataset, while double anchoring iteratively filters the protein-anchored set. Anchoring requires both roles to be represented; it does not impose a 1:1 class ratio.

The eight models are ligand-only, protein-only, C1–C3 and D1–D3. C models receive the selected PDB-chain sequence; D models receive residues from its predicted pockets. The numeric suffix denotes concatenation, unidirectional cross-attention or bidirectional cross-attention. Protein and ligand encoders are held fixed while the prediction heads are trained.

Models are assessed under random, Pfam-family-held-out, ligand-held-out and double-held-out evaluation. Ligand exclusion uses molecular connectivity. An additional Murcko scaffold analysis tests a broader structural exclusion. The double-anchored double-held-out experiment uses 47 leave-one-family-out partitions because the standard five-fold design leaves too little training data.

## Main findings

- **Dataset composition matters.** On 4,637 common test pairs, every-pair cross-attention models reached AUROCs of 0.927–0.942 under random splitting and 0.790–0.809 when both protein families and ligand identities were held out. The corresponding single-input controls had lower discrimination in their anchored datasets.
- **The strongest identity constraints leave a small benchmark.** Double anchoring retained 395 pairs. Under double-held-out evaluation, the 95% AUROC intervals of all eight models included 0.5.
- **ChEMBL rankings recover text-supported candidates unevenly across models.** Allosteric keywords appeared in 1,442 of 781,532 screened records (0.18%). For every-pair-trained D2, 39 of the top 782 records contained these keywords: 5.0%, or 27-fold enrichment. A separate evaluation uses published positive and negative binding-role annotations.
- **Observed structures provide another basis for ranking.** BioLiP reconstruction produced 1,322 labelled reference pairs and 2,890 candidate pairs; 2,874 candidate pairs had complete ranking inputs. Distance and ligand-property likelihood ratios recover known reference classes, although distance is closely related to the structural reference definitions.

SecondSite makes model-specific ChEMBL and BioLiP rankings searchable alongside activity and structural records. These collections support selection of interactions for follow-up experiments. Ranking scores are not calibrated probabilities of allostery, and source-database absence is not proof of biological novelty.

## Analysis workflow

| Step | File | Main calculation |
|---|---|---|
| 1 | [prepare_references.py](prepare_references.py) | Validate exact chemical identities, combine source annotations and separate conflicting pairs |
| 2 | [build_datasets.py](build_datasets.py) | Construct the four role-based datasets; apply and audit held-out partitions and scaffold exclusion |
| 3 | [train_models.py](train_models.py) | Define all eight architectures, fit prediction heads and select deployment durations from cross-validation |
| 4 | [evaluate_models.py](evaluate_models.py) | Combine seeds, calculate pooled and conditional metrics, and compare fixed predictions using paired cluster bootstrap |
| 5 | [screen_chembl.py](screen_chembl.py) | Score external pairs, exclude training pairs, rank within proteins and evaluate labels and assay keywords |
| 6 | [structural_followup.py](structural_followup.py) | Run prespecified Vina/Vinardo comparisons and summarize candidate and background-ligand poses |
| 7 | [rank_biolip.py](rank_biolip.py) | Aggregate structural observations per exact pair, summarize contacts at 60% frequency and fit 46 rankings |
| 8 | [compare_rankings.py](compare_rankings.py) | Compare existing neural and structural scores on shared reference pairs with paired uncertainty |

Steps 1–5 form the predictive workflow. Step 6 is an optional structural follow-up to a nominated interaction. Step 7 uses a separate structural reference, and step 8 connects its rankings with the neural benchmark. The scripts do not retrieve restricted databases or assign scientific labels from compound names.

## Installation and use

Python 3.10 or later is recommended. Install PyTorch for the available CPU/GPU first, then the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Inputs are supplied explicitly through command-line arguments. See [INPUTS.md](INPUTS.md) for required columns, identity conventions and prepared representations.

```bash
# Integrate identity-resolved database exports.
python prepare_references.py --sources inputs/source_annotations.tsv --output results/references

# Build datasets from the model-ready reference and supplied frozen fold assignments.
python build_datasets.py --model-ready-pairs inputs/model_ready_pairs.tsv.gz \
  --include-scaffolds --output results/datasets

# Fit one model, seed and outer fold. Repeat using the frozen experiment design.
python train_models.py --pairs results/datasets/protein_anchored.tsv.gz \
  --pockets inputs/pocket_indices.json --tensor-root inputs/tensors \
  --dataset protein_anchored --regime unseen_family --model c2 \
  --fold 0 --seed 20260817 --device cuda:0 --output results/fits/c2_seed20260817_fold0

# Evaluate predictions from three seeds across all outer folds.
python evaluate_models.py --predictions results/fits/*/predictions.tsv.gz \
  --output results/benchmark

# Fit structure-based rankings using prepared exact observations.
python rank_biolip.py --reference-observations inputs/reference_observations.tsv.gz \
  --candidate-observations inputs/candidate_observations.tsv.gz --output results/biolip
```

Each script provides `--help`. Output directories must be new. Training jobs are individual fits so they can be scheduled independently across available GPUs. Data preparation, reference identity decisions and encoder inputs are described in [METHODS.md](METHODS.md); supplementary experiments use the same model and evaluation functions with their specified inputs.

## Data and reproducibility

This is the analysis-code repository. Source database snapshots, frozen fold assignments, prepared embeddings, trained checkpoints and ranking tables are separate research assets. Redistribution of source data is subject to the respective database terms. The example commands require those inputs and do not reconstruct the paper from an empty machine.

The code consolidates the mathematical operations used in the study into reader-facing modules. Model forward passes, dataset membership, split exclusions and likelihood calculations are checked against the research implementations. Consolidating the execution code does not establish bit-identical GPU training across software versions. Historical encoder package and weight revisions were not fully pinned; archived tensor hashes identify the actual representations used in the experiments.

The web application, database export/deployment tools and manuscript figure sources are maintained separately. This repository does not distribute them.

## Citation

Jeong H, Shin HD, Jhang SY, Yi L, Seo J, Kim H. *SecondSite: a benchmark and web resource for prioritizing candidate allosteric protein–ligand interactions*. Manuscript in preparation.

Software citation metadata are provided in [CITATION.cff](CITATION.cff). A publication DOI and archived software DOI will be added when available. Questions about the analysis can be raised through this repository's issues.

## License

A software license has not yet been assigned. Upstream database and pretrained-model licenses apply separately to their respective materials.
