# main.py

"""
Orchestrator — run all processors to build the results JSON files.
Usage:
    python main.py [--method cc|bc|rb|all] [--dry-run]
"""

import argparse
import time
import warnings
from pathlib import Path
from collections import defaultdict

from config import TASK_TO_SHORT
from io_utils import (
    iter_cc_files, iter_bc_files, iter_rb_files,
    results_json_path, _parse_transfer_filename,
)
from processors.cc_processor import process_cc
from processors.bc_processor import process_bc
from processors.rb_processor import process_rb


# ── Dry-run ───────────────────────────────────────────────────────────────────

def dry_run(method: str) -> None:
    """
    Traverse the filesystem and report what would be processed,
    without loading any CSVs or writing any JSON files.
    """
    print(f"\n{'='*60}")
    print(f"DRY RUN — method: {method.upper()}")
    print(f"{'='*60}")

    iterators = {
        "cc": ("CC", iter_cc_files),
        "bc": ("BC", iter_bc_files),
        "rb": ("RB", iter_rb_files),
    }

    to_run = list(iterators.keys()) if method == "all" else [method]

    # Capture warnings during traversal so we can summarise them
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        for key in to_run:
            label, iter_fn = iterators[key]
            print(f"\n── {label} ──────────────────────────────────────────────")

            records = list(iter_fn())

            # Summary counters
            total          = len(records)
            by_task        = defaultdict(int)
            by_prompt      = defaultdict(int)
            by_family      = defaultdict(int)
            by_batch       = defaultdict(int)
            transfer_count = 0
            transfer_by_type = defaultdict(int)
            output_paths   = set()

            for rec in records:
                by_task[rec["task_type"]]   += 1
                by_prompt[rec["prompt"]]    += 1
                by_family[rec["family"]]    += 1
                if "batch_size" in rec:
                    by_batch[rec["batch_size"]] += 1
                if rec.get("is_transfer"):
                    transfer_count += 1
                    transfer_by_type[rec["transfer_type"]] += 1

                out = results_json_path(
                    key, rec["task_type"],
                    rec["prompt"], rec["family"], rec["dataset"]
                )
                output_paths.add(out)

            print(f"  Files found       : {total}")
            print(f"  By task type      : {dict(by_task)}")
            print(f"  By prompt level   : {dict(by_prompt)}")
            print(f"  By model family   : {dict(by_family)}")
            if by_batch:
                print(f"  By batch size     : {dict(sorted(by_batch.items()))}")
            if key == "rb":
                print(f"  Transfer files    : {transfer_count}")
                if transfer_by_type:
                    print(f"  Transfer by type  : {dict(transfer_by_type)}")
            print(f"  Output JSON files : {len(output_paths)}")

            # Show a sample of records (first 5)
            print(f"\n  Sample records (first 5):")
            for rec in records[:5]:
                _print_record(rec, key)

    # Summarise warnings (missing files)
    missing = [str(w.message) for w in caught_warnings
               if issubclass(w.category, UserWarning)]
    if missing:
        print(f"\n── WARNINGS ({len(missing)} missing files) ──────────────────")
        for m in missing[:20]:
            print(f"  ⚠  {m}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    else:
        print(f"\n  ✓ No missing files detected")

    print(f"\n{'='*60}")
    print("Dry run complete. No files were read or written.")
    print(f"{'='*60}\n")


def _print_record(rec: dict, method: str) -> None:
    parts = [
        f"prompt={rec['prompt']}",
        f"dataset={rec['dataset']}",
        f"family={rec['family']}",
        f"model={rec['model']}",
    ]
    if "batch_size" in rec:
        parts.append(f"n={rec['batch_size']}")
    if rec.get("is_transfer"):
        parts.append(f"transfer={rec['transfer_type']}→{rec.get('source_model') or rec.get('source_dataset') or rec.get('source_prompt')}")
    parts.append(f"path={rec['path']}")
    print(f"    • {' | '.join(parts)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build LLM logprob metrics JSONs")
    parser.add_argument(
        "--method", choices=["cc", "bc", "rb", "all"], default="all",
        help="Which correction method to process (default: all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Traverse filesystem and report what would be processed, without reading CSVs or writing JSONs"
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.method)
        return

    runners = {
        "cc": ("CC", process_cc),
        "bc": ("BC", process_bc),
        "rb": ("RB", process_rb),
    }

    to_run = list(runners.keys()) if args.method == "all" else [args.method]

    for key in to_run:
        label, fn = runners[key]
        print(f"\n{'='*50}")
        print(f"Processing {label}...")
        print(f"{'='*50}")
        t0 = time.time()
        fn()
        elapsed = time.time() - t0
        print(f"Done ({elapsed:.1f}s)")

    print("\nAll done.")


if __name__ == "__main__":
    main()