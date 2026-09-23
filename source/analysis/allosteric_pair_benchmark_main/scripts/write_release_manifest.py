#!/usr/bin/env python3
"""Write checksums for the explicit public-release file set."""

import argparse
import hashlib
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery"))
    args = parser.parse_args()
    package = args.project_root / "analysis/allosteric_pair_benchmark_main"
    relative = [
        "README.md", "EXPERIMENT_CONTRACT.md", "CODEX_GPU_TASK.md", "requirements-gpu.txt",
        "requirements-cpu.txt",
        "config/main_benchmark.json", "methods/MATERIALS_AND_METHODS.md",
        "docs/DATA_DICTIONARY.md", "docs/GITHUB_RELEASE_CHECKLIST.md",
        "scripts/build_aligned_target_chains.py", "scripts/prepare_cpu_release.py",
        "scripts/validate_target_chain_embeddings.py", "scripts/embed_target_chains_gpu.sh",
        "scripts/prepare_exact_gpu_inputs.py",
        "scripts/train_main_benchmark.py", "scripts/aggregate_main_results.py",
        "scripts/bootstrap_oof.py", "scripts/render_main_figure.py", "scripts/write_release_manifest.py",
        "scripts/send_gpu_main.sh", "scripts/run_gpu_main.sh", "scripts/fetch_gpu_main.sh",
        "tests/test_metric_consistency.py", "tests/test_coordinate_contract.py",
        "data/MAIN_COHORT.tsv.gz", "data/POCKET_INDICES.json", "data/PFAM_COMPONENTS.tsv",
        "data/TARGET_CHAIN_SEQUENCES.tsv", "data/UNIPROT_SEQUENCE_CACHE.json",
        "validation/CPU_INPUT_VALIDATION.json", "validation/POCKET_ALIGNMENT_AUDIT.tsv",
        "validation/POCKET_ALIGNMENT_VALIDATION.json",
    ]
    missing = [value for value in relative if not (package / value).is_file()]
    if missing:
        raise FileNotFoundError("release files are missing: {}".format(missing))
    lines = ["{}  {}".format(sha256(package / value), value) for value in relative]
    (package / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote {} checksums".format(len(lines)))


if __name__ == "__main__":
    main()
