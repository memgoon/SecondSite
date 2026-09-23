# Protein-and-ligand role-complete benchmark matrix

This package replaces the property-matched cohort as the strongest ligand-side
control. It keeps the broader protein-anchored benchmark as the general-domain
analysis and treats property matching as Supporting Information.

## Primary design

Three nested training cohorts are evaluated with the same code:

| training cohort | training rows | OOF evaluation rows | proteins | exact ligands | role contract |
|---|---:|---:|---:|---:|---|
| every pair | 6,854 | 4,637 | 940 | 4,418 | 2,217 broad-only rows are training augmentation |
| protein anchored | 4,637 | 4,637 | 426 | 2,910 | every protein has both labels |
| protein-and-ligand role complete | 395 | 395 | 98 | 40 | every protein and exact ligand has both labels |

“Role complete” means presence of both labels, not equal row counts. No rows
are discarded to force 1:1 ligand ratios and no ligand-specific training
weights are introduced. The pre-existing fold-local protein–label weighting is
retained.

This control removes deterministic protein and ligand identity rules; it does
not make the evidence sources interchangeable. Allosteric and orthosteric
annotations still originate from different curation pipelines, so residual
source provenance remains an explicit limitation.

Role completeness does not equalize ligand-specific label prevalence. The 40
ligands range from 0.10 to 0.833 allosteric prevalence, and an in-sample
ligand-prevalence lookup has AUROC 0.7566. Therefore fold-restricted
within-ligand AUROC is the primary ligand-control endpoint; pooled metrics are
reported with this structural diagnostic and are not described as ligand-free.

All three cohorts are trained under row-random, related-family-held-out, and
separate ligand-held-out splits. For the two general cohorts, ligand-held-out
CV is a secondary ligand-generalization axis. Their hardest endpoint remains
different: family-held-out predictions are restricted to test rows whose
ligand connectivity is also absent from training and validation.

A true joint family-and-ligand partition degenerates in every cohort: the
largest bipartite component contains 96.0% of every-pair rows, 98.6% of
protein-anchored rows, and 100% of role-complete rows. Instead, the two broad
training arms have a strict double-unseen endpoint obtained by restricting
family-held-out OOF predictions to compounds absent from train plus validation.
This leaves 2,721 common rows for an identical-row comparison. The
role-complete arm has only 11 such rows and three empty folds, so double-unseen
performance is explicitly unavailable there. The rectangular matrix contains
nine cohort-regime cells, or 1,080 fits across eight models, five folds, and
three seeds.

## Models

The internal matrix contains two single-input baselines and a 2 x 3 factorial
set of joint models:

| model | protein representation | interaction |
|---|---|---|
| C1 | whole selected target chain | concat |
| C2 | whole selected target chain | one-way ligand-to-protein attention |
| C3 | whole selected target chain | bidirectional atom–residue attention |
| D1 | pocket residues | concat |
| D2 | pocket residues | one-way ligand-to-pocket attention |
| D3 | pocket residues | bidirectional atom–residue attention |

Legacy names are mapped without overwriting prior results in
`data/MODEL_NAME_MAPPING.tsv`.

## Cross-cohort interpretation

Each arm has complete predictions on its contracted evaluation rows. The
every-pair arm is broad-training augmentation evaluated on the identical 4,637
protein-anchored rows, not a 6,854-row self-OOF benchmark. The primary
three-arm comparison uses the same 395 role-complete rows.

The claim-aligned comparisons are:

- family held out: within-ligand AUROC, each joint model versus protein-only;
  ligand-only is the expected 0.5 identity-constant check;
- ligand held out (all three cohorts): within-protein AUROC, each joint model
  versus ligand-only; protein-only is the expected 0.5 identity-constant check;
- double unseen (two general cohorts): family-held-out models on the common
  2,721 ligand-novel rows, using fold-restricted within-protein AUROC
  (1,805 rows / 112 groups) versus ligand-only as the main hard endpoint;
- within-ligand metrics are computed inside an outer fold;
- within-protein metrics are computed inside an outer fold.

Ligand-only within a ligand and protein-only within a protein are expected to
equal 0.5 by construction; these are implementation checks, not evidence of
model success.

Repeated exact inputs can acquire last-digit score differences when mixed-
precision reductions use different batch-padding shapes. Aggregate metrics
therefore average ligand-only scores within outer fold and full InChIKey and
protein-only scores within outer fold and UniProt before ranking. An audit
retains the raw spans and fails if any exceeds 0.01; connectivity-level
stereoisomer variation is not collapsed.

Conditional AUROC does not necessarily use every prediction row: single-class
fold-by-ligand or fold-by-protein blocks are undefined and skipped. Every
metric row therefore reports the input/used/skipped row and group counts. In
particular, conditional within-ligand analysis on the common 2,721-row hard
set has very sparse support (37 rows / 14 groups) and is sensitivity-only.
Pooled AUROC on all 2,721 rows is an identical-row split-change sensitivity,
not the primary hard endpoint, because it retains between-protein prevalence.
Likewise, row-random versus held-out conditional changes are labeled
descriptive when their effective support differs. An identical-395-row pooled
split-change table is supplied as a secondary sensitivity, not as a replacement
for the conditional endpoint.

Checkpoint selection is model-aware. Ligand-only uses validation within-
protein AUROC, protein-only uses validation within-ligand AUROC, and joint
models use the claim-aligned conditional statistic (or both for row-random).
Every requested statistic requires at least eight valid groups; otherwise a
recorded pooled-symmetric-AP fallback is used. Row-random joint fits never
silently substitute one available conditional statistic for the required
two-statistic mean.
The frozen audit expects 132/1,080 fallback fits, all in the 395-row arm
(132/360 there); actual incidence must match this cell-level contract and is
reported as a limitation because pooled epoch selection can retain between-
identity prevalence.

Cluster bootstrap resampling preserves repeated-cluster multiplicity for
nested macro groups by assigning a draw ID to every occurrence. The built-in
synthetic A,A,B check must return a 2/3 delta, and at least 95% of requested
replicates must have a defined paired statistic.

## ChEMBL stages

Three deploy ensembles are trained only after the internal matrix is complete:

- **every-pair sensitivity screen:** all 6,854 training pairs, applied to the
  same full ChEMBL OOD rows as the primary model;
- **primary general screen:** protein-anchored training, applied to the full
  ChEMBL OOD universe;
- **source-linked biochemical screen:** the 395-row role-complete training set,
  applied to ChEMBL molecules whose connectivity key entered through
  orthosteric KEGG, BRENDA, ChEBI, or UniProt provenance.

The 850-key list is not an independent metabolite catalog: all 2,321 source
rows are orthosteric. It is retained under the explicit source-linked name as
a candidate screen, not a metabolite validation or label-enrichment test.

Every-pair training is not treated as stronger ligand control: its extra 2,217
rows come from single-label proteins. It tests whether broader training
coverage improves external prioritization despite that additional confounding.
Every-pair and protein-anchored predictions are evaluated on identical rows,
with enrichment and score deltas written directly.
Protein-anchored remains the prespecified primary arm; choosing the better arm
after viewing ChEMBL weak labels would make that external set a model-selection
set rather than an independent validation.

The full ChEMBL screen uses ligand-only, protein-only, C1, C2, and D1–D3 for
both general training arms.
Ligand-only is evaluated on every row. Every other model is evaluated only on
the validated selected-chain subset
(804/837 full-screen proteins and 254/568 reference proteins), with all
baselines recomputed on identical rows. Protein-only/C1/C2 receive the same
selected target-chain tensors as internal training, and D1–D3 receive pockets
extracted from those tensors; legacy mixed full-UniProt/PDB tensors are not
used. Whole-chain C3 is retained only for the small biochemical screen because
its bidirectional atom–residue computation is disproportionate at full scale.
Reference and full-screen results are stratified into exact-target unseen,
ligand-connectivity unseen, double-novel, Pfam seen, Pfam unseen, and
annotation-unavailable groups relative to the primary protein-anchored
training reference. Both deployment arms use identical rows in direct deltas.
The frozen Pfam cache covers all 426 protein-anchored proteins but only 575 of
940 every-pair proteins. Consequently, a match can establish Pfam `seen` for
every-pair training, whereas a nonmatch is labeled
`training_reference_incomplete`, not `unseen`.

Selected-chain availability differs sharply by external endpoint: 254/568
reference proteins (44.7%) versus 804/837 full-screen proteins (96.1%). The
new reference metrics therefore must not be placed beside legacy metrics that
used mixed full-UniProt/PDB tensors and a different evaluation universe as if
they were directly comparable.

## Run

From the CPU server:

```bash
bash '/disk9/13.Heesu_Allostery/analysis/role_complete_pair_matrix/scripts/send_gpu_input.sh'
```

On the GPU server, internal benchmark first:

```bash
GPU_IDS=0,1 RUN_STAGE=benchmark bash '/disk1/11.HS_allostery/analysis/role_complete_pair_matrix/scripts/run_gpu.sh'
```

After reviewing the benchmark, run ChEMBL deployment and inference:

```bash
GPU_IDS=0,1 RUN_STAGE=chembl bash '/disk1/11.HS_allostery/analysis/role_complete_pair_matrix/scripts/run_gpu.sh'
```

Both stages are resumable. One GPU is also supported with `GPU_IDS=0`.
Resume requires matching source-cache, selected-chain, implementation, and
deployment-checkpoint fingerprints; changing any contracted artifact forces
rescoring.

Fetch on the CPU server:

```bash
bash '/disk9/13.Heesu_Allostery/analysis/role_complete_pair_matrix/scripts/fetch_gpu_return.sh'
```

Fetch runs the 10,000-replicate paired cluster bootstrap locally.
