# processors/rb_processor.py

"""
RB processor — k corrected values per item, multiple batch sizes, transfer support.
"""

import numpy as np
from config import GT_COL, TASK_TO_QTYPE, ANSWER_OPTIONS
from io_utils import (
    load_csv, load_json, save_json, deep_merge,
    results_json_path, iter_rb_files,
    raw_logprob_matrix, avg_corrected_logprob_matrix_rb,
    rb_fold_logprob_matrices, build_rb_transfer_key,
)
from metrics import (
    argmax_predictions, compute_snapshot_metrics,
    compute_rb_run_metrics, gt_distribution,
)


def process_rb() -> None:
    for rec in iter_rb_files():
        df = load_csv(rec["path"])
        if df is None:
            continue

        task_type = rec["task_type"]
        gt_col    = GT_COL[task_type]
        gt_labels = df[gt_col].tolist()
        qtype     = TASK_TO_QTYPE[task_type]
        batch_key = str(rec["batch_size"])

        # ── Raw metrics (before correction) ───────────────────────────────────
        raw_lp    = raw_logprob_matrix(df, task_type)
        raw_preds = argmax_predictions(raw_lp, task_type)
        raw_snap  = compute_snapshot_metrics(raw_preds, gt_labels, task_type)

        # ── Corrected model_dist from avg_corrected logprobs ──────────────────
        avg_lp         = avg_corrected_logprob_matrix_rb(df, task_type)
        avg_preds      = argmax_predictions(avg_lp, task_type)
        avg_corr_snap  = compute_snapshot_metrics(avg_preds, gt_labels, task_type)

        # ── Per-fold run metrics ───────────────────────────────────────────────
        run_matrices, run_gt = rb_fold_logprob_matrices(df, task_type)

        run_predictions = []
        for lp_mat, gts in zip(run_matrices, run_gt):
            preds = argmax_predictions(lp_mat, task_type)
            run_predictions.append(preds)

        agg = compute_rb_run_metrics(run_predictions, run_gt, task_type)

        # ── Ground truth distribution ─────────────────────────────────────────
        gt_dist = gt_distribution(gt_labels, task_type)

        # ── Build metrics block ───────────────────────────────────────────────

        metrics_block = {
        # Primary corrected metrics — from avg_corrected logprobs over full 200 items
        "acc":               avg_corr_snap["acc"],
        "tvd":               avg_corr_snap["tvd"],
        "rsd":               avg_corr_snap["rsd"],
        "model_dist":        avg_corr_snap["model_dist"],
        # Per-fold stability diagnostics
        "mean_acc":          agg["mean_acc"],
        "median_acc":        agg["median_acc"],
        "std_acc":           agg["std_acc"],
        "best_run_acc":      agg["best_run_acc"],
        "worst_run_acc":     agg["worst_run_acc"],
        "mean_tvd":          agg["mean_tvd"],
        "median_tvd":        agg["median_tvd"],
        "std_tvd":           agg["std_tvd"],
        "mean_rsd":          agg["mean_rsd"],
        "median_rsd":        agg["median_rsd"],
        "std_rsd":           agg["std_rsd"],
        # Raw (before correction)
        "raw_acc":           raw_snap["acc"],
        "raw_tvd":           raw_snap["tvd"],
        "raw_rsd":           raw_snap["rsd"],
        "raw_model_dist":    raw_snap["model_dist"],
        "ground_truth_dist": str(gt_dist),
        "num_calib_sets":    agg["num_calib_sets"],
        }

        # ── Build third-level transfer key ────────────────────────────────────
        transfer_key = build_rb_transfer_key(rec)

        # ── Nest into JSON structure ──────────────────────────────────────────
        nested = {
            rec["prompt"]: {
                qtype: {
                    transfer_key: {
                        rec["family"]: {
                            rec["model"]: {
                                batch_key: metrics_block
                            }
                        }
                    }
                }
            }
        }

        out_path = results_json_path(
            "rb", task_type,
            rec["prompt"], rec["family"], rec["dataset"]
        )
        existing = load_json(out_path)
        deep_merge(existing, nested)
        save_json(out_path, existing)