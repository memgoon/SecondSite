# Corrected competitive docking for CD73 and legacy controls

This package performs structural triage, not experimental validation of
allostery.  Its primary new question is whether `CHEMBL4549571` preferentially
docks to the literature-defined CD73 dimer-interface site rather than the
AMPCP-defined catalytic site.  It also reruns TDO2, MPO, and HSP90AA1 after
fixing two defects discovered in the first docking pass.

## What is corrected

1. **Exact site annotation.** Site atoms are selected by
   `source chain + residue name + residue number`.  A polymer tryptophan can no
   longer be mistaken for a crystallographic ligand named `TRP`.  TDO2 is
   explicitly split into catalytic `HEM402 + hetero TRP403` and exo-site
   `hetero TRP404`.
2. **pH-aware ligands.** Open Babel generates the pH 7.4 protonation state
   before RDKit 3-D embedding.  Preparation fails unless L-tryptophan is a
   zwitterion and ATP has net charge -4.
3. **Direct CD73 comparison.** The primary contrast is one prespecified box
   against one prespecified box: `known_dimer_interface - active_site_A`.
   The symmetry-related `active_site_B` is a separate sensitivity analysis.
4. **Matched background.** Twelve outcome-blind compounds matched on MW,
   cLogP, TPSA, charge, and rotatable-bond range are docked to CD73.  They
   estimate whether the hydrophobic interface generically scores well.  They
   are not experimentally proven CD73 inactives and are never called biological
   decoys.
5. **Best-of-N is secondary.** Fpocket cavities are retained for exploration,
   but their best-of-many score is not the primary CD73 result.

CD73 uses the biological dimer of the closed AMPCP-bound structure 4H2I.  The
dimer-interface box is centered on the two Glu543 residues, following the
published CD73 allosteric-screening definition.  Catalytic zinc ions are
retained.  A single rigid receptor and Vina/Vinardo remain approximations;
even a consistent interface preference only nominates a follow-up experiment.

## Recommended full run

Choose the CPU budget on the server.  Two Vina threads per process is usually
a reasonable balance:

```bash
cd /disk9/13.Heesu_Allostery
CPU_BUDGET=224 CPU_PER_JOB=2 RUN_PROFILE=full \
nohup bash analysis/chembl_competitive_docking_v2/scripts/run_cpu.sh \
  > analysis/chembl_competitive_docking_v2/logs/run.nohup.log 2>&1 &
```

For a 128-core allocation, for example:

```bash
CPU_BUDGET=128 CPU_PER_JOB=2 RUN_PROFILE=full \
nohup bash analysis/chembl_competitive_docking_v2/scripts/run_cpu.sh \
  > analysis/chembl_competitive_docking_v2/logs/run.nohup.log 2>&1 &
```

The workflow is resumable.  Reissuing the same command only reruns jobs whose
receptor, ligand, Vina binary, or complete job contract does not match.

Monitor with:

```bash
tail -F analysis/chembl_competitive_docking_v2/logs/run.nohup.log
tail -F analysis/chembl_competitive_docking_v2/logs/docking_progress.log

# Set WORKERS to CPU_BUDGET / CPU_PER_JOB, matching the launch command.
analysis/chembl_candidate_docking_v1/tools/dock_env/bin/python \
  analysis/chembl_competitive_docking_v2/scripts/status.py \
  --root analysis/chembl_competitive_docking_v2 --workers 112
```

Run individual stages with:

```bash
RUN_STAGE=prepare bash analysis/chembl_competitive_docking_v2/scripts/run_cpu.sh
RUN_STAGE=dock CPU_BUDGET=128 CPU_PER_JOB=2 RUN_PROFILE=full \
  bash analysis/chembl_competitive_docking_v2/scripts/run_cpu.sh
RUN_STAGE=analyze RUN_PROFILE=full \
  bash analysis/chembl_competitive_docking_v2/scripts/run_cpu.sh
```

For a fast candidate-only pilot, use `RUN_PROFILE=cd73_candidate`.  For the
final analysis rerun with `RUN_PROFILE=full`; completed candidate jobs are
reused because their job fingerprints are unchanged.

## Primary outputs

- `primary_named_site_results.tsv`: direct CD73 and corrected TDO2 results.
- `cd73_matched_background_null.tsv`: candidate rank against 12 matched
  background ligands for each scoring function.
- `named_site_comparison_summary.tsv`: all direct named-site contrasts.
- `de_novo_best_of_n_summary.tsv`: explicitly secondary fpocket exploration.
- `representative_pose_contacts.tsv`: residue contacts and distances to every
  named site.
- `VALIDATION.json`: completion and interpretation contract.

Interpret a negative `test_minus_reference_kcal_mol` as the named test site
(CD73 dimer interface or TDO2 exo site) scoring better than the reference site.
Do not compare absolute docking scores across unrelated proteins.
