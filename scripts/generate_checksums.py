#!/usr/bin/env python
"""
Generate or verify SHA256 checksums for files in data/frozen/.

Two modes:

    python scripts/generate_checksums.py            # write data/CHECKSUMS.txt
    python scripts/generate_checksums.py --verify   # compare disk against CHECKSUMS.txt

Output format (one record per line):
    <sha256>  <relative_path>  <size_bytes>

Why this matters for MCIS:
    Every training run logs the contents of CHECKSUMS.txt to W&B. If two team
    members report different B0 numbers, the checksums let us instantly tell
    whether the data on disk drifted, or whether the divergence is in code.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# ---- Paths ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN_DIR = REPO_ROOT / "data" / "frozen"
CHECKSUMS_FILE = REPO_ROOT / "data" / "CHECKSUMS.txt"

# Files we expect under data/frozen/. Missing files are flagged but not fatal,
# because some files (cv_folds.csv) are produced later in Week 1 by Malak.
EXPECTED_FILES = [
    "baheya_m1_primary_ready.csv",
    "baheya_m2_secondary_ready.csv",
    "baheya_uncoded_patients.csv",
    "cv_folds.csv",  # produced Day 2 by Malak
]


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA256 of a file, streaming so huge files don't OOM."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_files(root: Path) -> list[Path]:
    """All regular files under root, sorted for deterministic output."""
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file())


def write_checksums(out_file: Path, root: Path) -> int:
    files = discover_files(root)
    if not files:
        print(f"[WARN] No files found under {root}. Nothing to checksum.", file=sys.stderr)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text("")
        return 1

    lines = []
    print(f"Hashing {len(files)} file(s) under {root}...")
    for f in files:
        digest = sha256_of(f)
        rel = f.relative_to(REPO_ROOT)
        size = f.stat().st_size
        lines.append(f"{digest}  {rel}  {size}")
        print(f"  {digest[:12]}…  {rel}  ({size:,} bytes)")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {len(lines)} checksum(s) to {out_file.relative_to(REPO_ROOT)}")

    # Warn (not error) on any expected-but-missing files
    found_names = {f.name for f in files}
    missing = [name for name in EXPECTED_FILES if name not in found_names]
    if missing:
        print("\n[NOTE] The following expected files are missing from data/frozen/:")
        for name in missing:
            print(f"  - {name}")
        print("This is fine if those files have not been produced yet.")
    return 0


def verify_checksums(checksums_file: Path, root: Path) -> int:
    if not checksums_file.exists():
        print(
            f"[ERROR] {checksums_file} does not exist. Run without --verify first.", file=sys.stderr
        )
        return 2

    expected = {}
    for line in checksums_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("  ", 2)
        if len(parts) < 2:
            print(f"[ERROR] Malformed line in CHECKSUMS.txt: {line!r}", file=sys.stderr)
            return 2
        digest, rel = parts[0], parts[1]
        expected[rel] = digest

    on_disk = {str(p.relative_to(REPO_ROOT)): p for p in discover_files(root)}

    mismatches: list[tuple[str, str]] = []
    missing: list[str] = []
    extra: list[str] = []

    for rel, expected_digest in expected.items():
        if rel not in on_disk:
            missing.append(rel)
            continue
        actual_digest = sha256_of(on_disk[rel])
        if actual_digest != expected_digest:
            mismatches.append((rel, actual_digest))

    for rel in on_disk:
        if rel not in expected:
            extra.append(rel)

    if mismatches:
        print("[FAIL] Checksum MISMATCH detected on:")
        for rel, actual in mismatches:
            print(f"  - {rel}\n      expected: {expected[rel]}\n      actual:   {actual}")
    if missing:
        print("[FAIL] File(s) listed in CHECKSUMS.txt but missing from disk:")
        for rel in missing:
            print(f"  - {rel}")
    if extra:
        print("[WARN] File(s) on disk but not in CHECKSUMS.txt (regenerate?):")
        for rel in extra:
            print(f"  - {rel}")

    if mismatches or missing:
        print("\nVERIFICATION FAILED. Do not train until resolved.")
        return 1

    print(f"OK  {len(expected)} file(s) verified against {checksums_file.relative_to(REPO_ROOT)}.")
    if extra:
        print(
            "    (Some new files on disk are not yet checksummed; run without --verify to update.)"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify on-disk files against existing CHECKSUMS.txt instead of writing it.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=FROZEN_DIR,
        help=f"Directory to hash (default: {FROZEN_DIR.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=CHECKSUMS_FILE,
        help=f"Path to CHECKSUMS.txt (default: {CHECKSUMS_FILE.relative_to(REPO_ROOT)}).",
    )
    args = parser.parse_args()

    if args.verify:
        return verify_checksums(args.out, args.root)
    return write_checksums(args.out, args.root)


if __name__ == "__main__":
    sys.exit(main())
