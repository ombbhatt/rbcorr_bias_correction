# comparison_table.py
"""
Generate a methods comparison table for zeroshot prompt level, batch/calib size n=40.

Columns:
  Dataset | Model | Raw Acc | Raw TVD | Raw RSD
        | CC ΔAcc | CC ΔTVD | CC ΔRSD
        | BC ΔAcc | BC ΔTVD | BC ΔRSD
        | RB ΔAcc | RB ΔTVD | RB ΔRSD

Rows: grouped by dataset (yesno first, then nli, then mcq),
      within each dataset grouped by model family order.

Usage:
    python comparison_table.py               # prints to stdout
    python comparison_table.py --csv out.csv # also saves CSV
"""

import argparse
import json
import math
from pathlib import Path

# ── Config ──────────────────────────────────────

RESULTS_DIR = Path("../results/Mar-23-2026_N60K20")

PROMPT_LEVEL = "fewshot"
BATCH_SIZE   = 60

DATASET_ORDER = [
    # yesno
    "ARITH", "BABI", "COMPS", "EWOK",
    # nli  — no zeroshot, will be skipped
    "SNLI", "MNLI",
    # mcq
    "MMLU-HUMANITIES", "MMLU-OTHERS", "MMLU-SOCIAL_SCI", "MMLU-STEM",
]

DATASET_TASK = {
    "ARITH":           "yesno",
    "BABI":            "yesno",
    "COMPS":           "yesno",
    "EWOK":            "yesno",
    "SNLI":            "nli",
    "MNLI":            "nli",
    "MMLU-HUMANITIES": "mcq",
    "MMLU-OTHERS":     "mcq",
    "MMLU-SOCIAL_SCI": "mcq",
    "MMLU-STEM":       "mcq",
}

TASK_TO_SHORT = {
    "yesno": "yn",
    "nli":   "nli",
    "mcq":   "mcq",
}

NO_ZEROSHOT = {"SNLI", "MNLI"}

MODEL_FAMILIES = {
    # "GPT2": ["gpt2-medium", "gpt2-large"],
    "Falcon": [
        "Falcon3-3B-Base", "Falcon3-3B-Instruct",
        "Falcon3-10B-Base", "Falcon3-10B-Instruct",
    ],
    "Llama3": [
        "Llama-3.1-8B", "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B", "Llama-3.1-70B-Instruct",
    ],
    "Gemma3": [
        "gemma-3-27b-pt", "gemma-3-27b-it",
        "gemma-3-12b-pt", "gemma-3-12b-it",
    ],
}

# Ordered list of (family, model) pairs for row ordering
MODEL_ORDER = [
    (family, model)
    for family, models in MODEL_FAMILIES.items()
    for model in models
]

BATCH_KEY   = str(BATCH_SIZE)
TOTAL       = 1200
# K           = TOTAL // BATCH_SIZE   # 30
K          = 20  


# ── JSON loading ──────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def get_results_json(method: str, task_type: str, family: str, dataset: str) -> dict:
    short = TASK_TO_SHORT[task_type]
    path  = RESULTS_DIR / f"{method}_{short}" / f"{PROMPT_LEVEL}_{family}_{dataset}.json"
    return load_json(path)


# ── Metric extraction helpers ─────────────────────────────────────────────────

def _get(d: dict, *keys):
    """Safe nested dict access; returns None if any key is missing."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def extract_raw(data: dict, family: str, model: str,
                dataset: str, task_type: str) -> dict | None:
    """
    Raw metrics live inside every method's JSON under the same structure.
    We use the CC file as the canonical source for raw metrics since it
    has no batch-size dimension and is the simplest to navigate.
    Falls back to BC if CC is missing.
    """
    qtype = _qtype(task_type)

    # CC path: prompt -> qtype -> dataset -> family -> model -> "cc" -> raw_*
    block = _get(data, PROMPT_LEVEL, qtype, dataset, family, model, "cc")
    if block and _defined(block.get("raw_acc")):
        return {
            "acc": block["raw_acc"],
            "tvd": block["raw_tvd"],
            "rsd": block["raw_rsd"],
        }
    return None


def extract_cc(data: dict, family: str, model: str,
               dataset: str, task_type: str) -> dict | None:
    qtype = _qtype(task_type)
    block = _get(data, PROMPT_LEVEL, qtype, dataset, family, model, "cc")
    if block and _defined(block.get("acc")):
        return {"acc": block["acc"], "tvd": block["tvd"], "rsd": block["rsd"]}
    return None


def extract_bc(data: dict, family: str, model: str,
               dataset: str, task_type: str) -> dict | None:
    qtype = _qtype(task_type)
    block = _get(data, PROMPT_LEVEL, qtype, dataset, family, model, BATCH_KEY)
    if block and _defined(block.get("acc")):
        return {"acc": block["acc"], "tvd": block["tvd"], "rsd": block["rsd"]}
    return None


def extract_rb(data: dict, family: str, model: str,
               dataset: str, task_type: str) -> dict | None:
    qtype       = _qtype(task_type)
    transfer_key = f"{dataset}-from{dataset}"
    block = _get(data, PROMPT_LEVEL, qtype, transfer_key, family, model, BATCH_KEY)
    if block and _defined(block.get("acc")):
        return {"acc": block["acc"], "tvd": block["tvd"], "rsd": block["rsd"]}
    return None


def _qtype(task_type: str) -> str:
    return {"yesno": "YESNO", "nli": "NLI", "mcq": "MCQ"}[task_type]
    # return {"yesno": "YESNO", "nli": "NLI"}[task_type]


def _defined(v) -> bool:
    """True if v is a real number (not None or NaN)."""
    if v is None:
        return False
    try:
        return not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


# ── Delta formatting ──────────────────────────────────────────────────────────

def fmt_delta(corrected, baseline, higher_is_better=True) -> str:
    """
    Format the delta of a corrected metric vs baseline.
    For accuracy: higher is better → positive delta is good.
    For TVD/RSD:  lower is better  → negative delta is good.
    Returns a signed string to 4 decimal places, e.g. '+0.0432' or '-1.7260'.
    """
    if corrected is None or baseline is None:
        return "N/A"
    if not (_defined(corrected) and _defined(baseline)):
        return "N/A"
    delta = float(corrected) - float(baseline)
    sign  = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


def fmt_val(v) -> str:
    if v is None or not _defined(v):
        return "N/A"
    return f"{float(v):.4f}"


# ── Build rows ────────────────────────────────────────────────────────────────

def build_rows() -> list[dict]:
    rows = []

    for dataset in DATASET_ORDER:
        # if dataset in NO_ZEROSHOT:
            # continue

        task_type = DATASET_TASK[dataset]
        short     = TASK_TO_SHORT[task_type]

        # Load each method's JSON once per dataset
        cc_data = {}
        bc_data = {}
        rb_data = {}

        for family, _ in MODEL_FAMILIES.items():
            if not cc_data.get(family):
                cc_data[family] = get_results_json("cc", task_type, family, dataset)
            if not bc_data.get(family):
                bc_data[family] = get_results_json("bc", task_type, family, dataset)
            if not rb_data.get(family):
                rb_data[family] = get_results_json("rb", task_type, family, dataset)

        for family, model in MODEL_ORDER:
            raw = extract_raw(cc_data[family], family, model, dataset, task_type)
            cc  = extract_cc( cc_data[family], family, model, dataset, task_type)
            bc  = extract_bc( bc_data[family], family, model, dataset, task_type)
            rb  = extract_rb( rb_data[family], family, model, dataset, task_type)

            raw_acc = raw["acc"] if raw else None
            raw_tvd = raw["tvd"] if raw else None
            raw_rsd = raw["rsd"] if raw else None

            rows.append({
                "dataset": dataset,
                "model":   model,
                # Baseline
                "raw_acc": fmt_val(raw_acc),
                "raw_tvd": fmt_val(raw_tvd),
                "raw_rsd": fmt_val(raw_rsd),
                # CC deltas
                "cc_acc":  fmt_delta(cc["acc"]  if cc else None, raw_acc, higher_is_better=True),
                "cc_tvd":  fmt_delta(cc["tvd"]  if cc else None, raw_tvd, higher_is_better=False),
                "cc_rsd":  fmt_delta(cc["rsd"]  if cc else None, raw_rsd, higher_is_better=False),
                # BC deltas
                "bc_acc":  fmt_delta(bc["acc"]  if bc else None, raw_acc, higher_is_better=True),
                "bc_tvd":  fmt_delta(bc["tvd"]  if bc else None, raw_tvd, higher_is_better=False),
                "bc_rsd":  fmt_delta(bc["rsd"]  if bc else None, raw_rsd, higher_is_better=False),
                # RB deltas
                "rb_acc":  fmt_delta(rb["acc"]  if rb else None, raw_acc, higher_is_better=True),
                "rb_tvd":  fmt_delta(rb["tvd"]  if rb else None, raw_tvd, higher_is_better=False),
                "rb_rsd":  fmt_delta(rb["rsd"]  if rb else None, raw_rsd, higher_is_better=False),
            })

    return rows


# ── Printing ──────────────────────────────────────────────────────────────────

COLUMNS = [
    ("Dataset",  "dataset", 18),
    ("Model",    "model",   28),
    # Baseline
    ("Raw Acc",  "raw_acc",  8),
    ("Raw TVD",  "raw_tvd",  8),
    ("Raw RSD",  "raw_rsd",  8),
    # CC
    ("CC ΔAcc",  "cc_acc",   9),
    ("CC ΔTVD",  "cc_tvd",   9),
    ("CC ΔRSD",  "cc_rsd",   9),
    # BC
    ("BC ΔAcc",  "bc_acc",   9),
    ("BC ΔTVD",  "bc_tvd",   9),
    ("BC ΔRSD",  "bc_rsd",   9),
    # RB
    ("RB ΔAcc",  "rb_acc",   9),
    ("RB ΔTVD",  "rb_tvd",   9),
    ("RB ΔRSD",  "rb_rsd",   9),
]


def print_table(rows: list[dict]) -> None:
    # Header
    header = "  ".join(label.ljust(w) for label, _, w in COLUMNS)
    sep    = "  ".join("-" * w       for _, _, w       in COLUMNS)
    print(header)
    print(sep)

    prev_dataset = None
    for row in rows:
        # Blank line between dataset groups
        if prev_dataset is not None and row["dataset"] != prev_dataset:
            print()
        prev_dataset = row["dataset"]

        line = "  ".join(str(row[key]).ljust(w) for _, key, w in COLUMNS)
        print(line)


def save_csv(rows: list[dict], path: Path) -> None:
    import csv
    fieldnames = [key for _, key, _ in COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    rows = build_rows()
    print_table(rows)

    # if args.csv:
    save_csv(rows, Path(f"../results/comparison_mar23_{PROMPT_LEVEL}_n{BATCH_SIZE}-N60K20.csv"))


if __name__ == "__main__":
    main()