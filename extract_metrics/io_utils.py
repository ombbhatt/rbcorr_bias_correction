# io_utils.py

"""
Filesystem traversal, file discovery, CSV loading, and JSON I/O.
"""

import json
import re
import warnings
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterator

from config import (
    ROOT, RESULTS_DIR,
    PROMPT_LEVELS, NO_ZEROSHOT,
    DATASETS, MODEL_FAMILIES, MODEL_TO_FAMILY,
    CC_FOLDERS, BC_BASE_NAMES, RB_BASE_NAMES,
    BATCH_SIZES, BATCH_K, TRANSFER_BATCH_SIZE,
    RAW_LOGPROB_COLS, GT_COL,
    TASK_TO_SHORT,
)


# ── JSON I/O ─────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge update into base (in-place on base)."""
    for k, v in update.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


# ── Output path helpers ───────────────────────────────────────────────────────

def results_json_path(method: str, task_type: str,
                      prompt_level: str, family: str, dataset: str) -> Path:
    """
    e.g. results/rb_yn/fewshot_Falcon_ARITH.json
    """
    short = TASK_TO_SHORT[task_type]
    folder = RESULTS_DIR / f"{method}_{short}"
    fname  = f"{prompt_level}_{family}_{dataset}.json"
    return folder / fname


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception as e:
        warnings.warn(f"Could not load {path}: {e}")
        return None


# ── Parse RB corrected logprob list columns ───────────────────────────────────

def parse_rb_list_col(series: pd.Series) -> np.ndarray:
    parsed = []
    for val in series:
        if isinstance(val, float) and np.isnan(val):
            parsed.append(None)
            continue
        try:
            cleaned = str(val)
            # Handle np.float64(...) representations from transfer CSVs
            cleaned = re.sub(r'np\.float64\(([^)]+)\)', r'\1', cleaned)
            # Handle bare nan
            cleaned = cleaned.replace("nan", "None")
            lst = ast.literal_eval(cleaned)
            parsed.append([float("nan") if x is None else float(x) for x in lst])
        except Exception:
            parsed.append(None)

    k = next((len(r) for r in parsed if r is not None), 0)
    arr = np.full((len(parsed), k), np.nan)
    for i, row in enumerate(parsed):
        if row is not None:
            arr[i, :len(row)] = row
    return arr


# ── Build logprob matrix from DataFrame ──────────────────────────────────────

def raw_logprob_matrix(df: pd.DataFrame, task_type: str) -> np.ndarray:
    """(n_items, n_options) matrix of raw logprobs."""
    cols = RAW_LOGPROB_COLS[task_type]
    return df[cols].to_numpy(dtype=float)


def corrected_logprob_matrix_cc_bc(df: pd.DataFrame, task_type: str) -> np.ndarray:
    """(n_items, n_options) matrix of CC/BC corrected logprobs (single floats)."""
    cols = [f"corrected_{c}" for c in RAW_LOGPROB_COLS[task_type]]
    return df[cols].to_numpy(dtype=float)


def avg_corrected_logprob_matrix_rb(df: pd.DataFrame, task_type: str) -> np.ndarray:
    """(n_items, n_options) matrix of RB avg_corrected logprobs."""
    cols = [f"avg_corrected_{c}" for c in RAW_LOGPROB_COLS[task_type]]
    return df[cols].to_numpy(dtype=float)


def rb_fold_logprob_matrices(df: pd.DataFrame, task_type: str) -> list[np.ndarray]:
    """
    Returns a list of k (n_valid_items, n_options) matrices, one per fold.
    Items where ALL option logprobs are NaN at fold i are excluded.
    Also returns the corresponding gt label lists.
    """
    raw_cols = RAW_LOGPROB_COLS[task_type]
    corr_cols = [f"corrected_{c}" for c in raw_cols]
    n_options = len(raw_cols)

    # Parse each corrected column into (n_items, k) array
    fold_arrays = [parse_rb_list_col(df[c]) for c in corr_cols]
    # fold_arrays[opt_idx] has shape (n_items, k)

    if not fold_arrays or fold_arrays[0].size == 0:
        return [], []

    k = fold_arrays[0].shape[1]
    gt_col = GT_COL[task_type]
    gt_labels = df[gt_col].tolist()

    run_matrices = []   # list of k arrays, each (n_valid, n_options)
    run_gt       = []   # list of k gt label lists

    for fold_i in range(k):
        # For this fold, collect logprob vector per item
        lp = np.column_stack([fold_arrays[opt][:, fold_i]
                               for opt in range(n_options)])  # (n_items, n_options)
        # Valid rows: at least one non-NaN logprob
        valid_mask = ~np.all(np.isnan(lp), axis=1)
        run_matrices.append(lp[valid_mask])
        run_gt.append([gt_labels[i] for i, v in enumerate(valid_mask) if v])

    return run_matrices, run_gt


# ── File discovery ────────────────────────────────────────────────────────────

def _method_folder(method: str, task_type: str, n: int | None = None) -> str:
    """Return the folder name for a given method/task/size combo."""
    if method == "cc":
        return CC_FOLDERS[task_type]
    if method == "rb":
        k = BATCH_K[n]
    elif method == "bc":
        k = 1200 // n
    base = BC_BASE_NAMES[task_type] if method == "bc" else RB_BASE_NAMES[task_type]
    return f"{base}_n{n}_k{k}"


def iter_cc_files() -> Iterator[dict]:
    """Yield one record per CC results CSV found on disk."""
    for prompt in PROMPT_LEVELS:
        for dataset, (task_type, domain) in DATASETS.items():
            if prompt == "zeroshot" and dataset in NO_ZEROSHOT:
                continue
            folder_name = CC_FOLDERS[task_type]
            for family, models in MODEL_FAMILIES.items():
                for model in models:
                    path = (ROOT / prompt / dataset / folder_name
                            / family / domain / f"{model}_results.csv")
                    if path.exists():
                        yield dict(prompt=prompt, dataset=dataset,
                                   task_type=task_type, family=family,
                                   model=model, path=path)
                    else:
                        warnings.warn(f"Missing CC file: {path}")


def iter_bc_files() -> Iterator[dict]:
    """Yield one record per BC results CSV found on disk."""
    for prompt in PROMPT_LEVELS:
        for dataset, (task_type, domain) in DATASETS.items():
            if prompt == "zeroshot" and dataset in NO_ZEROSHOT:
                continue
            # for n in BATCH_SIZES:
            for n in [60]:  # Only look for n=60 BC files for now, since those are the only ones we have 
                folder_name = _method_folder("bc", task_type, n)
                for family, models in MODEL_FAMILIES.items():
                    for model in models:
                        path = (ROOT / prompt / dataset / folder_name
                                / family / domain / f"{model}_results.csv")
                        if path.exists():
                            yield dict(prompt=prompt, dataset=dataset,
                                       task_type=task_type, family=family,
                                       model=model, batch_size=n, path=path)
                        else:
                            warnings.warn(f"Missing BC file: {path}")


def iter_rb_files() -> Iterator[dict]:
    """
    Yield one record per RB results CSV found on disk,
    including transfer runs (only at n=40).

    Each record includes a `transfer_key` field describing the transfer type,
    and `source_*` fields for parsed source configuration.
    """
    for prompt in PROMPT_LEVELS:
        for dataset, (task_type, domain) in DATASETS.items():
            if prompt == "zeroshot" and dataset in NO_ZEROSHOT:
                continue
            for n in BATCH_SIZES + [TRANSFER_BATCH_SIZE]:  # Look for all batch sizes, but only TRANSFER_BATCH_SIZE will have transfer files
                folder_name = _method_folder("rb", task_type, n)
                for family, models in MODEL_FAMILIES.items():
                    for model in models:
                        base_dir = ROOT / prompt / dataset / folder_name / family / domain

                        if not base_dir.exists():
                            warnings.warn(f"Missing RB dir: {base_dir}")
                            continue

                        # Standard (non-transfer) file
                        if n != TRANSFER_BATCH_SIZE:
                            std_path = base_dir / f"{model}_results.csv"
                            if std_path.exists():
                                yield dict(
                                    prompt=prompt, dataset=dataset,
                                    task_type=task_type, family=family,
                                    model=model, batch_size=n,
                                    path=std_path,
                                    is_transfer=False,
                                    transfer_type=None,
                                    source_model=None,
                                    source_dataset=None,
                                    source_prompt=None,
                                )
                            else:
                                warnings.warn(f"Missing RB file: {std_path}")
                            continue

                        # Transfer files — only at n=40
                        # if n != TRANSFER_BATCH_SIZE:
                        #     continue

                        for tf_path in sorted(base_dir.glob(f"{model}_results_from_*.csv")):
                            parsed = _parse_transfer_filename(tf_path.name, model,
                                                              prompt, dataset, family)
                            if parsed is None:
                                warnings.warn(f"Could not parse transfer filename: {tf_path}")
                                continue
                            yield dict(
                                prompt=prompt, dataset=dataset,
                                task_type=task_type, family=family,
                                model=model, batch_size=n,
                                path=tf_path,
                                is_transfer=True,
                                **parsed,
                            )


def _parse_transfer_filename(filename: str, target_model: str,
                              target_prompt: str, target_dataset: str,
                              target_family: str) -> dict | None:
    """
    Parse a transfer RB filename like:
      {model}_results_from_{source}.csv
    where source is one of:
      - a dataset name        → cross-dataset
      - a prompt level        → cross-prompt
      - a model name          → cross-model (same family guaranteed)

    Returns dict with keys:
      transfer_type, source_model, source_dataset, source_prompt
    """
    stem = filename.replace("_results_from_", "_FROM_")
    stem = stem.replace(".csv", "")
    parts = stem.split("_FROM_")
    if len(parts) != 2:
        return None
    source_str = parts[1]

    all_datasets    = set(DATASETS.keys())
    all_prompts     = set(PROMPT_LEVELS)
    family_models   = set(MODEL_FAMILIES.get(target_family, []))

    if source_str in all_datasets:
        return dict(transfer_type="dataset",
                    source_dataset=source_str,
                    source_model=None,
                    source_prompt=None)
    elif source_str in all_prompts:
        return dict(transfer_type="prompt",
                    source_prompt=source_str,
                    source_dataset=None,
                    source_model=None)
    elif source_str in family_models:
        return dict(transfer_type="model",
                    source_model=source_str,
                    source_dataset=None,
                    source_prompt=None)
    else:
        return None


# ── JSON key builders ─────────────────────────────────────────────────────────

def build_rb_transfer_key(record: dict) -> str:
    """
    Build the third-level JSON key for an RB record.
    Format:
      "{target_dataset}-from{source_dataset}"            (always present)
      + "_from{source_prompt}"                           (cross-prompt)
      + "_{target_model}_from{source_model}"             (cross-model)
    """
    target_dataset = record["dataset"]
    transfer_type  = record.get("transfer_type")

    # Dataset part
    if transfer_type == "dataset":
        src_dataset = record["source_dataset"]
    else:
        src_dataset = target_dataset
    dataset_part = f"{target_dataset}-from{src_dataset}"

    # Optional suffixes
    suffix = ""
    if transfer_type == "model":
        suffix = f"_{record['model']}_from{record['source_model']}"
    elif transfer_type == "prompt":
        suffix = f"_{record['prompt']}_from{record['source_prompt']}"

    return dataset_part + suffix