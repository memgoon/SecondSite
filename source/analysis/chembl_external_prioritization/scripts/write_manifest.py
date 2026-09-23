#!/usr/bin/env python3
"""Write checksums for the explicit CPU-to-GPU handoff files."""

import argparse
import hashlib
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery"))
    args = parser.parse_args()
    relative_package = Path("analysis/chembl_external_prioritization")
    package = args.project_root / relative_package
    files = [
        package / "README.md", package / "EXPERIMENT_CONTRACT.md",
        package / "CODEX_GPU_TASK.md", package / "config.json",
        package / "validation/CPU_INPUT_VALIDATION.json",
    ]
    for directory in ["data", "methods", "scripts"]:
        files.extend(sorted(path for path in (package / directory).glob("*") if path.is_file() and path.name != "MANIFEST.sha256"))
    output = package / "MANIFEST.sha256"
    lines = []
    for path in sorted(set(files)):
        if not path.is_file():
            raise FileNotFoundError(path)
        lines.append("{}  {}".format(digest(path), path.relative_to(args.project_root)))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote {} checksums to {}".format(len(lines), output))


if __name__ == "__main__":
    main()
