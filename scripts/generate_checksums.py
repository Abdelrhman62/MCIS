#!/usr/bin/env python3
"""Phase 0.1 — Generate or verify SHA256 checksums for all frozen data.

Walks ``data/frozen/`` recursively and hashes every regular file. Writes
``data/CHECKSUMS.txt`` with one line per file, sorted by relative path so
diffs are deterministic across runs.

The data files themselves stay gitignored. Only ``data/CHECKSUMS.txt`` is
tracked, so the team can confirm everyone is training on the same bytes.

Coverage:
    - data/frozen/m1_model_ready.parquet            (Baheya M1)
    - data/frozen/tcga_pretrain_model_ready/...     (8453 reports + vocab)
    - data/frozen/tcga_brca_external_model_ready/...(1025 BRCA external)
    - data/frozen/seer_rare_augmentation_model_ready/...(619 templates)
    - any future frozen artifact — just drop it under data/frozen/

Per-bundle CHECKSUMS.txt files inside subdirectories are checksummed too.
Changing one is meaningful (it would indicate the bundle was modified
post-freeze) and we want the master CHECKSUMS.txt to catch that.

Usage:
    python scripts/generate_checksums.py             # generate
    python scripts/generate_checksums.py --verify    # verify
    python scripts/generate_checksums.py --root /custom/path  # override root

Exit codes:
    0  success (generate) or all OK (verify)
    1  setup error (no frozen dir, or no files to hash)
    2  verify failure (mismatch or missing)
    3  verify warning (new files on disk not in CHECKSUMS.txt)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "frozen"
DEFAULT_CHECKSUM_FILE = REPO_ROOT / "data" / "CHECKSUMS.txt"

# Files to never include in the master checksum manifest.
# Hidden files (.DS_Store, .gitkeep) and our own output.
SKIP_NAMES = {".DS_Store", ".gitkeep"}


def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """SHA256 hex digest of a file, computed in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def collect_files(root: Path, manifest_path: Path) -> list[tuple[Path, Path]]:
    """Walk root recursively. Return [(absolute_path, relative_path)] sorted by relpath.

    Skips:
    - The manifest file itself (avoid self-reference)
    - Hidden files in SKIP_NAMES
    - Symlinks (only follow regular files)
    """
    if not root.exists():
        return []

    items: list[tuple[Path, Path]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.is_symlink():
            continue
        if p.name in SKIP_NAMES:
            continue
        if p.resolve() == manifest_path.resolve():
            continue
        rel = p.relative_to(root)
        items.append((p, rel))

    # Sort by relative path string for deterministic output across OSes
    items.sort(key=lambda x: str(x[1]))
    return items


def parse_manifest(path: Path) -> dict[str, str]:
    """Parse an existing CHECKSUMS.txt file. Returns relpath -> sha256."""
    expected: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: "<sha256>  <relpath>  (<bytes>)"
        # Split on at least 2 spaces to separate digest from relpath
        parts = line.split("  ", 2)
        if len(parts) < 2:
            continue
        digest = parts[0].strip()
        relpath = parts[1].strip()
        expected[relpath] = digest
    return expected


def cmd_generate(root: Path, manifest_path: Path) -> int:
    files = collect_files(root, manifest_path)
    if not files:
        print(f"[error] no frozen data files under {root}", file=sys.stderr)
        print("        is data/frozen/ populated?", file=sys.stderr)
        return 1

    lines = []
    print(f"Hashing {len(files)} file(s) under {root}:")
    for abs_path, rel_path in files:
        digest = sha256_of_file(abs_path)
        size = abs_path.stat().st_size
        lines.append(f"{digest}  {rel_path}  ({size:,} bytes)")
        # Pretty-print: pad relpath for alignment up to a reasonable width
        pretty_rel = str(rel_path)
        if len(pretty_rel) > 60:
            pretty_rel = "..." + pretty_rel[-57:]
        print(f"  {pretty_rel:60s}  {digest[:16]}...")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {len(files)} checksum(s) to "
          f"{manifest_path.relative_to(REPO_ROOT) if manifest_path.is_relative_to(REPO_ROOT) else manifest_path}")
    return 0


def cmd_verify(root: Path, manifest_path: Path) -> int:
    if not manifest_path.exists():
        print(f"[error] {manifest_path} does not exist; run without --verify first.",
              file=sys.stderr)
        return 1

    expected = parse_manifest(manifest_path)
    if not expected:
        print(f"[error] {manifest_path} is empty or unparseable.", file=sys.stderr)
        return 1

    files_on_disk = collect_files(root, manifest_path)
    on_disk_relpaths = {str(rel): abs_p for abs_p, rel in files_on_disk}

    n_ok = 0
    n_mismatch = 0
    n_missing = 0
    n_new = 0

    print(f"Verifying {len(expected)} expected file(s) against {root}:")

    # Check each expected entry
    for rel_str, expected_digest in sorted(expected.items()):
        if rel_str not in on_disk_relpaths:
            print(f"  {rel_str:60s}  [MISSING]")
            n_missing += 1
            continue
        actual = sha256_of_file(on_disk_relpaths[rel_str])
        if actual == expected_digest:
            print(f"  {rel_str:60s}  OK")
            n_ok += 1
        else:
            print(f"  {rel_str:60s}  [MISMATCH]")
            print(f"    expected: {expected_digest}")
            print(f"    actual:   {actual}")
            n_mismatch += 1

    # Check for files on disk that aren't in the manifest
    new_files = [
        rel_str for rel_str in on_disk_relpaths
        if rel_str not in expected
    ]
    for rel_str in sorted(new_files):
        print(f"  {rel_str:60s}  [NEW — not in manifest]")
        n_new += 1

    print()
    print(f"Summary: {n_ok} OK, {n_mismatch} mismatch, "
          f"{n_missing} missing, {n_new} new")

    if n_mismatch or n_missing:
        return 2
    if n_new:
        return 3
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify against existing CHECKSUMS.txt instead of regenerating",
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_DATA_ROOT),
        help=f"Data root to walk (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_CHECKSUM_FILE),
        help=f"Manifest path (default: {DEFAULT_CHECKSUM_FILE})",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifest = Path(args.manifest).resolve()

    if args.verify:
        return cmd_verify(root, manifest)
    return cmd_generate(root, manifest)


if __name__ == "__main__":
    sys.exit(main())
