# BioLiP consensus-60: downstream reanalysis v2

This package supersedes v1 **as an execution package**, without modifying it. It runs the affected computational chain, not merely a pair-score sensitivity. Preparation tests use temporary synthetic data. Real production has NOT been run by the code-preparation session.

## Execute and resume

```bash
bash /disk9/13.Heesu_Allostery/analysis/biolip_consensus60_pipeline_v2/scripts/run_local.sh --workers 16 --io-workers 2
```

Run exactly the same command after interruption. The CPU worker count can be changed on resume; it is an execution setting, not a scientific parameter. Defaults: up to 8 CPU processes and 2 coordinate-reading processes. `--workers 16` uses 16 independent processes, each with one BLAS/OpenMP thread. Do not just increase `OMP_NUM_THREADS` on the old v1 script. Final independent validation and merging contain sequential work; no claim of linear speedup is made.

Input-only check:

```bash
bash /disk9/13.Heesu_Allostery/analysis/biolip_consensus60_pipeline_v2/scripts/run_local.sh --preflight-only
```

To survive an agent/terminal session ending:

```bash
cd /disk9/13.Heesu_Allostery/analysis/biolip_consensus60_pipeline_v2
RUN_LOG="$(mktemp /tmp/biolip60_v2.XXXXXX.log)"
nohup bash scripts/run_local.sh --workers 16 --io-workers 2 >"$RUN_LOG" 2>&1 < /dev/null &
echo "PID=$! LOG=$RUN_LOG"
```

Outputs: `runs/complete_reanalysis/`. Progress: `STATUS.json`, per-task commit files in `tasks/`, and per-stage records in `stages/`. A 5-hour agent token limit does not stop an independently launched local process. System/process interruptions are handled by resume. The coordinator is the only output writer; workers are terminated by Linux parent-death signal if the coordinator is hard-killed, avoiding orphan workers retaining its run lock. Do not delete `RUN.lock` or checkpoints. Uncommitted outputs are preserved as `.orphan.*`; committed corruption stops execution. Failed-attempt JSONs are retained. A hard kill can leave the last status RUNNING; that is not completion evidence.

Every task is keyed by its input hash. Completed tasks are hash-validated and skipped. Bounded in-flight work prevents submission of all coordinate jobs at once. Source files, runtime scripts, scientific parameters and library versions are pinned. Code/source changes require a new package/run; never remove a fingerprint to bypass a failure. Completed runs are independently validated read-only and skipped without re-exporting their data/plots.

## Included stages

1. **Source gates.** Verify pinned reference features, original fold assignments, candidate features, CP10 chemistry/known-role audit, old comparison rankings and coordinate/sequence manifests. Only listed files and coordinate paths explicitly recorded in the frozen manifest are accessed. No repository recursion or network retrieval.
2. **Reference AND candidate reconstruction.** One UniProt/full-InChIKey pair, with all source observations in lineage. Residue frequency is the mean within-PDB contacting fraction across PDBs, threshold >=3/5. Feature rows use the median of within-PDB medians consistently on both sides. This is an explicit aggregation/training-weight amendment. Original protein/family folds are retained and a pair crossing labels or folds is a hard failure, not silently removed. The 60% threshold is not Jaccard >=0.60.
3. **Cached-coordinate spatial audit.** Verify each used coordinate file; map the exact receptor chain to its frozen UniProt sequence using the existing alignment utility. Recover the exact ligand CCD/chain/residue occurrence. Compare distinct structural instances after common-CA alignment. Diagnostic same-location support requires >=20 common CA atoms, >=95% mapped identity, >=80% chain coverage, CA RMSD <=2.5 A, ligand-centroid separation <=4 A and ligand-heavy-atom minimum separation <=2 A. Distinct-location support requires the same alignment gates with centroid separation >=10 A and minimum separation >=4 A. Intermediate/failed cases stay unresolved. These are conservative **new diagnostic gates**, not fitted biological boundaries and not tuned using rankings. A group must satisfy ALL within-group same-location and ALL cross-group distinct-location comparisons; no single-linkage chain becomes a final location. One pair remains one primary record even with several nested locations. A singleton is marked separately from multi-observation concordance. Geometry support is not manual or biological-assembly validation.
4. **Recompute contact-summary geometry where meaningful.** Derive >=60% residues within geometrically coherent locations. Map these to each real structure; require >=80% CA recovery for ligand-to-consensus-centroid distances. No virtual merged coordinates. Unresolved pairs retain observation evidence and a review entry rather than forcing one site. The original ligand-centroid-to-frozen-known-orthosteric-site distance is intentionally reused: that measurement is independent of the old candidate complete-linkage grouping. The orthosteric reference-site definition is not changed by stealth. Residue observability does not change the frozen frequency denominator.
5. **Full refitting and external scoring.** Fit QNB, KDE_LEGACY and KDE100 over all 15 feature subsets, plus raw distance: 46 ranking definitions. Preserve smoothing, Scott KDE bandwidth rule, caps and rounding from the frozen estimator implementation. Refit on pair-level inputs in each of the existing 2 regimes x 5 folds; separately refit on all reference pairs for candidate scoring. No old OOF score or old model weight is substituted. Incomplete candidate features remain explicit NA rows; no observation/pair disappears silently. Singular/invalid fits stop; no fallback estimator is inserted to force PASS.
6. **Held-out pair evaluation.** Recompute pooled AUROC, within-protein macro AUROC/AUPRC, Recall@1 and MRR. Run 10,000 paired protein- or family-cluster bootstrap replicates for each method/regime, including delta AUROC/MRR versus raw distance. Methods share resampling draws within each regime. Save support counts, cluster units, draw hashes, OOF scores and training provenance. These pair-based metrics are NOT directly comparable to old site-signature AUROCs as an improvement claim.
7. **Ranking, filters, comparisons.** Recompute global/within-protein pair ranks and the four-view agreement. Reuse unchanged, hashed CP10 chemistry/known-role flags conservatively (all old records must pass; disagreement flagged), not old final eligibility or rank decisions. Recompute spatial thresholds and repeated-PDB support. Preserve old best-site ranks/scores alongside new values. Shortlists are provisional; the six previous manual outcomes are not automatically transferred.
8. **Plots and offline SecondSite handoff.** Export matplotlib PNG/PDF plots and their TSV source data, report, pair ranking/record tables and nested location records. These are replacement-ready numerical/visual source assets, NOT a rewrite of the existing manuscript PPTX/DOCX or a live database migration. The actual website is not changed.
9. **Independent validation.** A separate validator does not import production grouping or model functions. It reconstructs pairs/contacts and independently fits the bin/KDE formulas, checks every model score, all OOF coverage, all candidate ranks/filters and all bootstrap intervals. It checks spatial numeric gates/location consensus and coordinate hashes, but does NOT independently reimplement the upstream sequence-to-coordinate mapping. Manual assembly/crystal-contact review remains outstanding. Successful completion is `PASS_COMPUTATIONAL_PIPELINE`, never “novel allostery proven” or “manual review complete”.

## Files to inspect after a run

- `data/REFERENCE_PAIRS.tsv`, `CANDIDATE_PAIRS.tsv`, `OBSERVATION_LINEAGE.tsv.gz`, `RESIDUE_FREQUENCIES.tsv.gz`.
- `data/SPATIAL_COMPARISONS.tsv.gz`, `NESTED_LOCATION_CONSENSUS.tsv`, `CONSENSUS_GEOMETRY.tsv.gz`, `SPATIAL_REVIEW_QUEUE.tsv`.
- `data/OOF_PAIR_SCORES.tsv.gz`, `HELDOUT_PAIR_METRICS_AND_CI.tsv`, `HELDOUT_PER_PROTEIN_METRICS.tsv`.
- `data/CANDIDATE_RANKINGS.tsv.gz`, `FILTER_COUNTS.tsv`, `PROVISIONAL_PAIR_SHORTLIST.tsv`.
- `figure_source/heldout_pair_recovery.{png,pdf}`, `pair_filter_counts.{png,pdf}`; this is not the final edited main Figure 4.
- `secondsite_handoff/PAIR_RANKINGS.tsv.gz`, `PAIR_RECORDS.tsv`, `NESTED_LOCATIONS.tsv`, `MANIFEST.json`; not automatically uploaded.
- `SUMMARY.json`, `REPORT.md`, `VALIDATION.json`, `STATUS.json`, `FINAL_CHECKSUMS.sha256`.

## Validate after execution

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /disk9/13.Heesu_Allostery/miniconda3/bin/python \
 /disk9/13.Heesu_Allostery/analysis/biolip_consensus60_pipeline_v2/scripts/validate_pipeline.py \
 --run-dir /disk9/13.Heesu_Allostery/analysis/biolip_consensus60_pipeline_v2/runs/complete_reanalysis --verify-final
```

```bash
cd /disk9/13.Heesu_Allostery/analysis/biolip_consensus60_pipeline_v2/runs/complete_reanalysis
sha256sum --check --quiet FINAL_CHECKSUMS.sha256
```

## Limitations and timing

The frozen eligible reference and candidate observation universes are retained; previously failed source observations are not magically recovered by residue consensus. Reference structural distances and known orthosteric-site definitions are unchanged. Site geometry uses the selected receptor chain, not a complete biological-assembly/crystal-symmetry analysis. A single protein chain may not represent a binding-competent assembly. Geometry-consistent candidates still require structure-context review before a scientific or web-release claim. A pair median may conceal a minority real binding site; nested evidence is retained and multi-site cases are explicitly reviewable. This is not chemical or functional evidence for allostery.

A measured real-data runtime is not yet available. The earlier 20–60 minute estimate applied to the smaller v1 scope, not this expanded pipeline. v2 adds real-coordinate comparisons, full pair-level refitting and independent refitting/CI checks. Runtime depends on repeated-instance counts and CIF access; inspect stage progress rather than assuming linear speedup or a fixed finish time. The code checkpoints through long runs.

Tests are synthetic and must not be presented as successful real-data reanalysis. Run `scripts/test_pipeline.py` to repeat them. `scripts/freeze_package.py` is maintainer-only one-time preparation; it pins inputs/code, records tests and preflight, and refuses to overwrite an existing lock.
