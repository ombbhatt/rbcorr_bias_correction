# paired_ttest.py
"""
Paired t-tests comparing debiasing methods across all dataset-model combinations.

Reads the comparison CSV produced by comparison_table.py and runs paired t-tests
for accuracy, TVD, and RSD across the following comparisons:
  - Baseline vs RB
  - Baseline vs BC
  - Baseline vs CC
  - RB vs BC
  - RB vs CC

Usage:
    python paired_ttest.py --csv results/comparison_zeroshot_n40.csv
    python paired_ttest.py --csv results/comparison_zeroshot_n40.csv --out results/ttest_results.csv
"""

import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_delta(series: pd.Series) -> np.ndarray:
    """
    Convert a delta column (e.g. '+0.0432', '-1.7260', 'N/A') to a float array.
    N/A entries become NaN.
    """
    def _parse(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return float("nan")
    return np.array([_parse(v) for v in series])


def parse_val(series: pd.Series) -> np.ndarray:
    """Convert a raw value column to a float array."""
    return series.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def paired_ttest(a: np.ndarray, b: np.ndarray):
    """
    Run a paired t-test on two arrays, dropping rows where either is NaN.
    Returns (mean_diff, t_stat, p_value, n_pairs).
    mean_diff = mean(a - b)
    """
    mask = ~(np.isnan(a) | np.isnan(b))
    a_clean, b_clean = a[mask], b[mask]
    n = len(a_clean)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    diff      = a_clean - b_clean
    mean_diff = float(np.mean(diff))
    t_stat, p_val = stats.ttest_rel(a_clean, b_clean)
    return mean_diff, float(t_stat), float(p_val), n


def sig_stars(p: float) -> str:
    if math.isnan(p):    return ""
    if p < 0.001:        return "***"
    if p < 0.01:         return "**"
    if p < 0.05:         return "*"
    return ""

def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "N/A"
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to comparison CSV from comparison_table.py")

    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # ── Reconstruct absolute metric values from deltas ────────────────────────
    # Raw values
    raw_acc = parse_val(df["raw_acc"])
    raw_tvd = parse_val(df["raw_tvd"])
    raw_rsd = parse_val(df["raw_rsd"])

    # Deltas
    cc_dacc = parse_delta(df["cc_acc"]);  cc_dtvd = parse_delta(df["cc_tvd"]);  cc_drsd = parse_delta(df["cc_rsd"])
    bc_dacc = parse_delta(df["bc_acc"]);  bc_dtvd = parse_delta(df["bc_tvd"]);  bc_drsd = parse_delta(df["bc_rsd"])
    rb_dacc = parse_delta(df["rb_acc"]);  rb_dtvd = parse_delta(df["rb_tvd"]);  rb_drsd = parse_delta(df["rb_rsd"])

    # Absolute corrected values
    cc_acc = raw_acc + cc_dacc;  cc_tvd = raw_tvd + cc_dtvd;  cc_rsd = raw_rsd + cc_drsd
    bc_acc = raw_acc + bc_dacc;  bc_tvd = raw_tvd + bc_dtvd;  bc_rsd = raw_rsd + bc_drsd
    rb_acc = raw_acc + rb_dacc;  rb_tvd = raw_tvd + rb_dtvd;  rb_rsd = raw_rsd + rb_drsd

    # ── Define comparisons ────────────────────────────────────────────────────
    # Each entry: (label, a_values, b_values)
    # mean_diff = mean(a - b), so positive means a > b
    comparisons = [
    ("RB vs Baseline",  rb_acc, raw_acc,  rb_tvd, raw_tvd,  rb_rsd, raw_rsd),
    ("BC vs Baseline",  bc_acc, raw_acc,  bc_tvd, raw_tvd,  bc_rsd, raw_rsd),
    ("CC vs Baseline",  cc_acc, raw_acc,  cc_tvd, raw_tvd,  cc_rsd, raw_rsd),
    ("RB vs BC",        rb_acc, bc_acc,   rb_tvd, bc_tvd,   rb_rsd, bc_rsd),
    ("RB vs CC",        rb_acc, cc_acc,   rb_tvd, cc_tvd,   rb_rsd, cc_rsd),
    ("BC vs CC",        bc_acc, cc_acc,   bc_tvd, cc_tvd,   bc_rsd, cc_rsd),
    ]

    metrics = [
        ("Accuracy",  "higher is better", 0, 1),
        ("TVD",       "lower is better",  2, 3),
        ("RSD",       "lower is better",  4, 5),
    ]

    # ── Run tests and collect results ─────────────────────────────────────────
    result_rows = []

    for metric_name, direction, idx_a, idx_b in metrics:
        for label, a_acc, b_acc, a_tvd, b_tvd, a_rsd, b_rsd in comparisons:
            all_vals = (a_acc, b_acc, a_tvd, b_tvd, a_rsd, b_rsd)
            a = all_vals[idx_a]
            b = all_vals[idx_b]

            mean_diff, t_stat, p_val, n = paired_ttest(a, b)
            result_rows.append({
                "metric":      metric_name,
                "direction":   direction,
                "comparison":  label,
                "mean_diff":   f"{mean_diff:+.4f}" if not math.isnan(mean_diff) else "N/A",
                "t_stat":      f"{t_stat:+.4f}"    if not math.isnan(t_stat)    else "N/A",
                "p_value":     fmt_p(p_val),
                "significance": sig_stars(p_val),
                "n_pairs":     n,
            })

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\nPaired t-tests — zeroshot, n=60, N pairs per test shown in brackets")
    print(f"Significance: * p<0.05  ** p<0.01  *** p<0.001  (two-tailed)")
    print(f"mean_diff = mean(A - B); positive means A > B\n")

    current_metric = None
    for row in result_rows:
        if row["metric"] != current_metric:
            current_metric = row["metric"]
            print(f"{'─'*64}")
            print(f"  {row['metric']}  ({row['direction']})")
            print(f"{'─'*64}")
            print(f"  {'Comparison':<22}  {'mean(A-B)':>10}  {'t-stat':>8}  {'p-value':>10}  {'sig':>4}  {'N':>4}")
            print(f"  {'─'*22}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*4}  {'─'*4}")

        print(f"  {row['comparison']:<22}  {row['mean_diff']:>10}  {row['t_stat']:>8}  {row['p_value']:>10}  {row['significance']:>4}  {row['n_pairs']:>4}")

        # Blank line after last comparison in each metric block
        if row["comparison"] == comparisons[-1][0]:
            print()

    # ── Save CSV ───────────────────────────────────────────────────────────────
    # if args.out:
    out_path = Path(f"{args.csv}").with_name(f"ttest_{Path(args.csv).stem}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result_rows).to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()