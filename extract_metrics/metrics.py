# metrics.py

"""
Pure metric calculations — no I/O, no pandas, just numpy + plain Python.
All functions operate on lists/arrays of predictions and ground truth labels.
"""

import numpy as np
from collections import Counter
from config import ANSWER_OPTIONS


# ── Prediction distribution ───────────────────────────────────────────────────

def prediction_distribution(predictions, task_type: str) -> dict:
    """Fraction of times each answer option was predicted."""
    options = ANSWER_OPTIONS[task_type]
    n = len(predictions)
    counts = Counter(predictions)
    return {str(opt): counts.get(opt, 0) / n for opt in options}


def gt_distribution(gt_labels, task_type: str) -> dict:
    """Empirical ground truth label distribution."""
    options = ANSWER_OPTIONS[task_type]
    n = len(gt_labels)
    counts = Counter(gt_labels)
    return {str(opt): counts.get(opt, 0) / n for opt in options}


# ── Accuracy ──────────────────────────────────────────────────────────────────

def accuracy(predictions, gt_labels) -> float:
    """Fraction of correct predictions."""
    if len(predictions) == 0:
        return float("nan")
    return float(np.mean([p == g for p, g in zip(predictions, gt_labels)]))


# ── TVD ───────────────────────────────────────────────────────────────────────

def tvd(pred_dist: dict, task_type: str) -> float:
    """
    Total Variation Distance between model prediction distribution
    and the uniform (ground truth) distribution.
    TVD = 0.5 * sum(|p_i - 1/|Y||)
    """
    options = ANSWER_OPTIONS[task_type]
    uniform = 1.0 / len(options)
    return float(0.5 * sum(abs(pred_dist.get(str(opt), 0.0) - uniform)
                           for opt in options))


# ── RSD ───────────────────────────────────────────────────────────────────────

def rsd(predictions, gt_labels, task_type: str) -> float:
    """
    Relative Standard Deviation of class-wise accuracy.
    RSD = population_std(class_accs) / mean(class_accs)
    Uses ddof=0 consistent with the paper's formula.
    """
    options = ANSWER_OPTIONS[task_type]

    # Group indices by ground truth label
    groups = {opt: [] for opt in options}
    for pred, gt in zip(predictions, gt_labels):
        if gt in groups:
            groups[gt].append(pred == gt)

    class_accs = []
    for opt in options:
        items = groups[opt]
        if len(items) == 0:
            continue
        class_accs.append(float(np.mean(items)))

    if len(class_accs) == 0:
        return float("nan")

    mean_acc = float(np.mean(class_accs))
    if mean_acc == 0:
        return float("nan")

    std_acc = float(np.std(class_accs, ddof=0))
    return std_acc / mean_acc


# ── Argmax helpers ────────────────────────────────────────────────────────────

def argmax_predictions(logprob_matrix: np.ndarray, task_type: str) -> list:
    """
    Given a (n_items, n_options) matrix of logprobs,
    return the predicted answer label for each item.
    """
    options = ANSWER_OPTIONS[task_type]
    indices = np.argmax(logprob_matrix, axis=1)
    return [options[i] for i in indices]


# ── Single-snapshot metrics (CC / BC) ────────────────────────────────────────

def compute_snapshot_metrics(predictions, gt_labels, task_type: str) -> dict:
    """
    Compute acc, TVD, RSD, and model_dist for a single set of predictions.
    Used for CC and BC corrected values, and for raw values across all methods.
    """
    pred_dist = prediction_distribution(predictions, task_type)
    return {
        "acc":        accuracy(predictions, gt_labels),
        "tvd":        tvd(pred_dist, task_type),
        "rsd":        rsd(predictions, gt_labels, task_type),
        "model_dist": str(pred_dist),
    }


# ── Multi-run metrics (RB) ────────────────────────────────────────────────────

def compute_rb_run_metrics(
    run_predictions: list[list],
    run_gt_labels:   list[list],
    task_type: str,
) -> dict:
    """
    Compute aggregated metrics across k RB runs.

    Parameters
    ----------
    run_predictions : list of k lists, each containing predicted labels
                      for items valid in that run (NaNs excluded)
    run_gt_labels   : list of k lists, parallel to run_predictions
    task_type       : "yesno" | "nli" | "mcq"

    Returns
    -------
    dict with mean/median/std for acc, tvd, rsd;
    best_run_acc, worst_run_acc, num_calib_sets
    """
    accs, tvds, rsds = [], [], []

    for preds, gts in zip(run_predictions, run_gt_labels):
        if len(preds) == 0:
            continue
        pred_dist = prediction_distribution(preds, task_type)
        accs.append(accuracy(preds, gts))
        tvds.append(tvd(pred_dist, task_type))
        rsds.append(rsd(preds, gts, task_type))

    def _agg(vals, fn):
        return float(fn(vals)) if vals else float("nan")

    return {
        "mean_acc":      _agg(accs, np.mean),
        "median_acc":    _agg(accs, np.median),
        "std_acc":       _agg(accs, lambda x: np.std(x, ddof=0)),
        "best_run_acc":  _agg(accs, np.max),
        "worst_run_acc": _agg(accs, np.min),
        "mean_tvd":      _agg(tvds, np.mean),
        "median_tvd":    _agg(tvds, np.median),
        "std_tvd":       _agg(tvds, lambda x: np.std(x, ddof=0)),
        "mean_rsd":      _agg(rsds, np.mean),
        "median_rsd":    _agg(rsds, np.median),
        "std_rsd":       _agg(rsds, lambda x: np.std(x, ddof=0)),
        "num_calib_sets": len(accs),
    }