# processors/bc_processor.py

"""
BC processor — one corrected value per item, multiple batch sizes, no transfer.
"""

from config import GT_COL, TASK_TO_QTYPE
from io_utils import (
    load_csv, load_json, save_json, deep_merge,
    results_json_path, iter_bc_files,
    raw_logprob_matrix, corrected_logprob_matrix_cc_bc,
)
from metrics import (
    argmax_predictions, compute_snapshot_metrics,
    gt_distribution,
)


def process_bc() -> None:
    for rec in iter_bc_files():
        df = load_csv(rec["path"])
        if df is None:
            continue

        task_type  = rec["task_type"]
        gt_col     = GT_COL[task_type]
        gt_labels  = df[gt_col].tolist()
        qtype      = TASK_TO_QTYPE[task_type]
        batch_key  = str(rec["batch_size"])

        # ── Raw metrics ───────────────────────────────────────────────────────
        raw_lp    = raw_logprob_matrix(df, task_type)
        raw_preds = argmax_predictions(raw_lp, task_type)
        raw_snap  = compute_snapshot_metrics(raw_preds, gt_labels, task_type)

        # ── Corrected metrics ─────────────────────────────────────────────────
        corr_lp    = corrected_logprob_matrix_cc_bc(df, task_type)
        corr_preds = argmax_predictions(corr_lp, task_type)
        corr_snap  = compute_snapshot_metrics(corr_preds, gt_labels, task_type)

        # ── Ground truth distribution ─────────────────────────────────────────
        gt_dist = gt_distribution(gt_labels, task_type)

        # ── Build metrics block ───────────────────────────────────────────────
        metrics_block = {
            "acc":               corr_snap["acc"],
            "tvd":               corr_snap["tvd"],
            "rsd":               corr_snap["rsd"],
            "model_dist":        corr_snap["model_dist"],
            "raw_acc":           raw_snap["acc"],
            "raw_tvd":           raw_snap["tvd"],
            "raw_rsd":           raw_snap["rsd"],
            "raw_model_dist":    raw_snap["model_dist"],
            "ground_truth_dist": str(gt_dist),
        }

        # ── Nest into JSON structure ──────────────────────────────────────────
        nested = {
            rec["prompt"]: {
                qtype: {
                    rec["dataset"]: {
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
            "bc", task_type,
            rec["prompt"], rec["family"], rec["dataset"]
        )
        existing = load_json(out_path)
        deep_merge(existing, nested)
        save_json(out_path, existing)