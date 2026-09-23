#!/usr/bin/env python3
"""Write checksums for the explicit GPU handoff files without tree traversal."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery"))
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    package = args.project_root / "analysis/allosteric_pair_benchmark_broad_superset"
    paths = [
        package / "README.md",
        package / "EXPERIMENT_CONTRACT.md",
        package / "CODEX_GPU_TASK.md",
        package / "requirements-cpu.txt",
        package / "requirements-gpu.txt",
        package / "methods/MATERIALS_AND_METHODS.md",
        package / "docs/GPU_HANDOFF_COMMANDS.txt",
        package / "data/BROAD_ALIGNED_POOL.tsv.gz",
        package / "data/UNIFIED_TARGET_CHAIN_SEQUENCES.tsv",
        package / "data/EXTRA_TARGET_CHAIN_SEQUENCES.tsv",
        package / "data/POCKET_INDICES.json",
        package / "data/UNIPROT_SEQUENCE_CACHE.json",
        package / "data/REFERENCE_ARM_A_MODEL_READY.tsv.gz",
        package / "data/REFERENCE_ARM_A_OOF_PREDICTIONS.tsv.gz",
        package / "data/REFERENCE_ARM_A_TRAINING_INDEX.json",
        package / "validation/CPU_INPUT_VALIDATION.json",
        package / "validation/TARGET_CHAIN_ALIGNMENT_AUDIT.tsv",
        package / "validation/PRE_GPU_SPLIT_DESIGN.tsv",
        package / "validation/PRE_GPU_SPLIT_VALIDATION.json",
    ]
    paths.extend(sorted((package / "scripts").glob("*.py")))
    paths.extend(sorted((package / "scripts").glob("*.sh")))
    paths.extend(sorted((package / "data/extra_target_chain_fasta").glob("*.fasta")))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing manifest inputs: {}".format(missing))
    lines = [
        "{}  {}".format(sha256(path), path.relative_to(args.project_root))
        for path in paths
    ]
    output = package / "MANIFEST.sha256"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote {} entries to {}".format(len(lines), output))


if __name__ == "__main__":
    main()
