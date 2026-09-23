# Broad-superset training-pool experiment

This package tests one question while leaving the completed main benchmark
unchanged:

> On the same frozen validation and test pairs, does training on every
> input-ready pair improve over training only on proteins that contain both
> allosteric and orthosteric labels?

Arm A is the completed `allosteric_pair_benchmark_main` cohort (4,637 pairs,
426 proteins). Arm B is constructed as a strict superset, `Arm A + E`, under
the same graph-free target-chain, fpocket-union, ESM3, and exact Uni-Mol input
contract. The CPU-frozen pre-GPU pool contains 6,975 pairs and 950 proteins;
the final Arm B count is frozen only after exact ligand-tensor validation.

The test and validation rows never change. Only Arm B is newly trained:

- six heads: ligand, protein, C1, C2, C3, and D1;
- row-random and unseen-family regimes;
- five folds and three seeds;
- 180 additional fits.

For unseen-family training, additional rows are excluded if their exact
protein or any Pfam accession overlaps the locked validation/test proteins.
The unseen-compound analysis is a test subset, not another trained model. Its
rows are defined against Arm B training plus the locked validation fold and
then applied identically to Arm A and Arm B predictions.

## Run order

CPU preparation has already been completed locally. To rebuild it explicitly:

```bash
python3 analysis/allosteric_pair_benchmark_broad_superset/scripts/build_broad_superset_cpu.py
```

Transfer with one authenticated SSH stream:

```bash
bash analysis/allosteric_pair_benchmark_broad_superset/scripts/send_gpu_input.sh
```

Then run one command on the GPU server:

```bash
bash /disk1/11.HS_allostery/analysis/allosteric_pair_benchmark_broad_superset/scripts/run_gpu_broad_superset.sh
```

After completion, fetch and validate locally:

```bash
bash analysis/allosteric_pair_benchmark_broad_superset/scripts/fetch_gpu_return.sh
```

No script recursively scans the project tree. The handoff and return archives
contain explicit paths only. ESM3 and exact ligand embeddings are generated
only when they are absent; completed fits resume safely.

## Principal outputs

- `gpu_cache/BROAD_MODEL_READY.tsv.gz`: final Arm B training universe.
- `validation/SPLIT_AUDIT.tsv`: per-fold additions and zero-overlap gates.
- `gpu_output/broad_superset/aggregate/POOL_MODEL_SUMMARY.tsv`: Arm A/B metrics.
- `gpu_output/broad_superset/aggregate/PAIRED_DELTAS_ARM_B_MINUS_A.tsv`: seed-level paired differences.
- `gpu_output/broad_superset/aggregate/PAIRED_CLUSTER_BOOTSTRAP_ARM_B_MINUS_A.tsv`: 10,000-replicate Pfam-component bootstrap.

Figure rendering is intentionally outside this package.
