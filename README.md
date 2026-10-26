# SecondSite

### Benchmarking allosteric–orthosteric classification and ranking candidate protein–ligand interactions

**Heesu Jeong, Hanshin David Shin, So Yun Jhang, Lina Yi, Jeongwoo Seo and Heebal Kim**  
Seoul National University · Correspondence: Heebal Kim

[Web resource](https://secondsite.snu.ac.kr) · [Model weights](https://huggingface.co/memgoon/SecondSite-models) · [Study data](https://huggingface.co/datasets/memgoon/SecondSite-data) · [Input formats](INPUTS.md) · [Methods](METHODS.md)

## Research question

Can a model distinguish an allosteric protein–ligand interaction from an orthosteric one, and how much of its performance depends on familiar proteins or compounds?

SecondSite examines this question using complementary reference datasets and held-out evaluation settings. We combine curated allosteric records with primary-ligand, substrate, cofactor and catalytic-site annotations, then compare models that receive ligand information, protein information or both. We also investigate whether these models can prioritize interactions in ChEMBL and whether experimentally observed binding positions in BioLiP provide complementary rankings for follow-up research.

This repository provides the benchmark, external ranking and structural follow-up workflows. Modules share the same model definitions and statistical functions.

## Benchmark design

Allosteric annotations come from ASD. Orthosteric references come from GtoPdb, KLIFS, BRENDA, UniProt, ChEBI and KEGG. Records are linked by exact UniProt accession and standardized full InChIKey; conflicting binding-role assignments are kept out of the benchmark. The integrated reference contains 20,640 pairs across 7,384 proteins. Of these, 6,854 pairs have the representations required by all eight models.

| Dataset | Selection rule | Pairs | Proteins | Ligands |
|---|---|---:|---:|---:|
| Every-pair | All model-ready pairs | 6,854 | 940 | 4,418 |
| Both-class proteins | Frozen evaluation set; each protein has both class labels | 4,637 | 426 | 2,910 |
| Both-class ligands | Each retained ligand has both class labels | 1,365 | 543 | 122 |
| Both-class proteins & ligands | Both requirements hold after iterative filtering | 395 | 98 | 40 |

Both-class means that allosteric and orthosteric class labels are both represented for each retained protein or ligand. Both-class proteins preserves the frozen evaluation population; the broader every-pair set adds training records. Both-class ligands is selected from every-pair, while Both-class proteins & ligands iteratively filters Both-class proteins. The requirement does not impose a 1:1 class ratio. Dataset selection is based on individual protein or ligand identities; Pfam-family grouping is a separate evaluation-split operation.

The eight models are ligand-only, protein-only, C1–C3 and D1–D3. C models receive the selected PDB-chain sequence; D models receive residues from its predicted pockets. The numeric suffix denotes concatenation, unidirectional cross-attention or bidirectional cross-attention. Protein and ligand encoders are held fixed while the prediction heads are trained.

Models are assessed under random, Pfam held-out, ligand held-out and double held-out evaluation. Ligand exclusion uses molecular connectivity. An additional Murcko scaffold analysis tests a broader structural exclusion. Only double held-out evaluation on Both-class proteins & ligands uses 47 leave-one-family-out partitions; its Pfam held-out evaluation remains five-fold.

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
| 9 | [time_split.py](time_split.py) | Reconstruct all four datasets from records through 2015; select epochs on earlier validation records, refit and predict later records |
| 10 | [ranking_recovery.py](ranking_recovery.py) | Evaluate fixed ChEMBL rankings against assay annotations and literature-supported pairs, with fractional boundary ties and protein-cluster intervals |
| 11 | [docking_sites.py](docking_sites.py) | Match top/bottom-decile compounds, classify first poses by residue contacts, and compare paired site-support frequencies |
| 12 | [attention_analysis.py](attention_analysis.py) | Measure residue attention at annotated binding sites and its variation across ligands and seeds |

Steps 1–5 provide the main benchmark and screening workflow. Steps 7–8 provide structure-based rankings and their comparison with model scores. Time-split validation uses steps 9–10. Structural follow-up uses step 11 for selection, step 6 for docking, and step 11 for site classification and paired analysis.

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
python train_models.py --pairs results/datasets/both_class_proteins.tsv.gz \
  --pockets inputs/pocket_indices.json --tensor-root inputs/tensors \
  --dataset both_class_proteins --regime pfam_held_out --model c2 \
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

Prepared inputs are described in [INPUTS.md](INPUTS.md). Preserve the frozen row IDs,
fold assignments, embedding tensors and pocket indices when reproducing the study.
Model weights are maintained in [SecondSite-models](https://huggingface.co/memgoon/SecondSite-models);
deployment and held-out evaluation checkpoints have separate directories.
The companion data repository is
[SecondSite-data](https://huggingface.co/datasets/memgoon/SecondSite-data).
It is being prepared for release. Its manifest identifies the available files
and their source-specific licenses.

Docking analyses use reviewed receptor assemblies and site coordinates in the same
coordinate frame. Site-contact agreement is a structural follow-up measure, not an
experimental demonstration of allosteric modulation.

## Citation

Jeong H, Shin HD, Jhang SY, Yi L, Seo J, Kim H. *SecondSite: a benchmark and web resource for prioritizing candidate allosteric protein–ligand interactions*. Manuscript in preparation.

Software citation metadata are provided in [CITATION.cff](CITATION.cff). Questions about the analysis can be raised through this repository's issues.

## License

The analysis code is released under the [MIT License](LICENSE). Source databases
and pretrained models retain their respective licenses.
