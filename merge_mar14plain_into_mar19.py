#!/usr/bin/env python3
"""
Merge src1 into src2 in-place.

Usage:
    python merge_into_dir2.py <src1> <src2>

- Files/dirs only in src1 are copied into src2.
- Files/dirs only in src2 are left untouched.
- Conflicting files: src2 wins by default (use --overwrite to let src1 win).
- Conflicting types (file vs dir): skipped with a warning.
"""

import argparse
import shutil
import sys
from pathlib import Path


def merge_into(src: Path, dest: Path, overwrite: bool = False, dry_run: bool = False) -> None:
    """Recursively merge src into dest (dest is modified in-place)."""
    for entry in sorted(src.iterdir(), key=lambda p: p.name):
        target = dest / entry.name

        if not target.exists():
            # Only in src → copy it over
            _copy(entry, target, dry_run)

        elif entry.is_dir() and target.is_dir():
            # Both dirs → recurse
            merge_into(entry, target, overwrite=overwrite, dry_run=dry_run)

        elif entry.is_file() and target.is_file():
            # Both files → conflict
            if overwrite:
                print(f"  [conflict → src1 wins] {target}")
                if not dry_run:
                    shutil.copy2(entry, target)
            else:
                print(f"  [conflict → src2 keeps] {target}")

        else:
            # Type mismatch
            print(f"  [SKIP — type mismatch] {entry.name}: "
                  f"{'dir' if entry.is_dir() else 'file'} vs "
                  f"{'dir' if target.is_dir() else 'file'}", file=sys.stderr)


def _copy(src: Path, dst: Path, dry_run: bool) -> None:
    kind = "dir " if src.is_dir() else "file"
    print(f"  [copy {kind}] {src} → {dst}")
    if dry_run:
        return
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Merge src1 into src2 in-place.")
    parser.add_argument("src1", help="Source directory to merge from")
    parser.add_argument("src2", help="Destination directory to merge into (modified in-place)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="On file conflict, overwrite src2's file with src1's (default: keep src2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without making any changes",
    )
    args = parser.parse_args()

    src1, src2 = Path(args.src1), Path(args.src2)

    for p, label in [(src1, "src1"), (src2, "src2")]:
        if not p.is_dir():
            print(f"Error: {label} is not a valid directory: {p}", file=sys.stderr)
            sys.exit(1)

    mode = "DRY RUN — " if args.dry_run else ""
    print(f"\n{mode}Merging '{src1}' into '{src2}'\n")

    merge_into(src1, src2, overwrite=args.overwrite, dry_run=args.dry_run)
    print("\nDone." if not args.dry_run else "\nDry run complete. No files were changed.")


if __name__ == "__main__":
    main()