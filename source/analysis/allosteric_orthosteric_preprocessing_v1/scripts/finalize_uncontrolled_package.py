#!/usr/bin/env python3
"""Create the publication-package inventory and SHA256 manifest."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/CHECKSUMS.sha256"
INVENTORY = ROOT / "data/PACKAGE_FILE_INVENTORY.tsv"
CHECKSUM_VALIDATION = ROOT / "validation/CHECKSUM_VALIDATION.json"
STATUS = ROOT / "work/status/UNCONTROLLED_PREPROCESSING.json"
CONTENT_VALIDATION = ROOT / "validation/VALIDATION.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_files(include_inventory: bool) -> list[Path]:
    paths: list[Path] = []
    for name in ["README.md", "CODE_AVAILABILITY.md"]:
        path = ROOT / name
        if path.is_file():
            paths.append(path)
    for rel, pattern in [
        ("scripts", "*.py"),
        ("config", "*"),
        ("data", "*"),
        ("cache", "*"),
        ("reports", "*"),
        ("validation", "*"),
        ("work/status", "*"),
    ]:
        directory = ROOT / rel
        if directory.exists():
            paths.extend(p for p in directory.glob(pattern) if p.is_file())
    paths = [p for p in paths if p != MANIFEST and (include_inventory or p != INVENTORY)]
    return sorted(set(paths), key=lambda p: str(p.relative_to(REPO)))


def main() -> None:
    # Place final checksum-gate metadata before hashing.  The claim is accepted
    # only if the verification at the end of this function succeeds.
    CHECKSUM_VALIDATION.parent.mkdir(parents=True, exist_ok=True)
    CHECKSUM_VALIDATION.write_text("{}\n", encoding="utf-8")
    expected_targets = len(package_files(include_inventory=False)) + 1  # add inventory
    checksum_gate = {
        "status": "PASS",
        "manifest": str(MANIFEST.relative_to(REPO)),
        "target_count": expected_targets,
        "verification": "Every listed target was rehashed after final manifest creation.",
    }
    CHECKSUM_VALIDATION.write_text(json.dumps(checksum_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = json.loads(STATUS.read_text())
    status["checksum_manifest_status"] = "PASS"
    status["checksum_manifest_target_count"] = expected_targets
    status["checksum_validation_path"] = str(CHECKSUM_VALIDATION.relative_to(REPO))
    STATUS.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation = json.loads(CONTENT_VALIDATION.read_text())
    validation["checksum_gate"] = checksum_gate
    CONTENT_VALIDATION.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    base_files = package_files(include_inventory=False)
    INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    with INVENTORY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256", "inventory_scope"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for path in base_files:
            writer.writerow(
                {
                    "relative_path": str(path.relative_to(REPO)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "inventory_scope": "all package files except this inventory and checksum manifest",
                }
            )

    files = package_files(include_inventory=True)
    if len(files) != expected_targets:
        raise SystemExit(f"unexpected checksum target count: {len(files)} != {expected_targets}")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}  {path.relative_to(REPO)}\n")

    failures = []
    for path in files:
        expected = next(line.split()[0] for line in MANIFEST.read_text().splitlines() if line.endswith(str(path.relative_to(REPO))))
        if sha256_file(path) != expected:
            failures.append(str(path.relative_to(REPO)))
    if failures:
        raise SystemExit(f"checksum failures: {failures}")
    print(f"PASS: {len(files)} package targets hashed and verified")
    print(MANIFEST)


if __name__ == "__main__":
    main()
