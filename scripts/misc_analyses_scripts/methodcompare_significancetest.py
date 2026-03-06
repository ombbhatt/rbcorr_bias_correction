import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ── Config ────────────────────────────────────────────────────────────────────

CSV_PATH = "../../results/table_outputs/methodcompare_table_data_all.csv"
OUTPUT_PATH = "../../results/table_outputs/significance_results.csv"

MODELS = [
    "Falcon3-3B-Base", "Falcon3-3B-Instruct",
    "Falcon3-10B-Base", "Falcon3-10B-Instruct",
    # "gemma-3-12b-pt", "gemma-3-12b-it",
    "gemma-3-27b-pt", "gemma-3-27b-it",
    "Llama-3.1-8B", "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B", "Llama-3.1-70B-Instruct",
]

METHODS = ["CC_Acc", "BC_Acc", "Ours_Acc"]
BASELINE = "Baseline_Acc"

DATASET_TYPES = {
    "Type1": ["ARITH", "BABI", "COMPS", "EWOK"],
    "Type2": ["SNLI", "MNLI"],
    "Type3": ["HUMANITIES", "OTHERS", "SOCIAL_SCI", "STEM"],
}

# ── Load & validate ───────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)

missing_models = set(MODELS) - set(df["Model"].unique())
if missing_models:
    print(f"Warning: these models are missing from the CSV: {missing_models}")

all_datasets = [d for datasets in DATASET_TYPES.values() for d in datasets]
missing_datasets = set(all_datasets) - set(df["Dataset"].unique())
if missing_datasets:
    print(f"Warning: these datasets are missing from the CSV: {missing_datasets}")

# ── Run paired t-tests ────────────────────────────────────────────────────────

records = []

for dtype, datasets in DATASET_TYPES.items():
    for dataset in datasets:
        for method in METHODS:
            subset = df[(df["Dataset"] == dataset) & (df["Model"].isin(MODELS))].copy()

            if subset.empty:
                print(f"Skipping {dataset} / {method}: no data found.")
                continue

            diffs = subset[method].values - subset[BASELINE].values
            n = len(diffs)
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, ddof=1)

            if n < 2 or std_diff == 0:
                print(f"Skipping {dataset} / {method}: insufficient variance or samples.")
                continue

            t_stat, p_val = stats.ttest_1samp(diffs, popmean=0)

            records.append({
                "Dataset_Type": dtype,
                "Dataset": dataset,
                "Method": method,
                "N_Models": n,
                "Mean_Delta": round(mean_diff, 5),
                "Std_Delta": round(std_diff, 5),
                "T_Statistic": round(t_stat, 4),
                "P_Value_Raw": p_val,
            })

results = pd.DataFrame(records)

# ── Apply BH correction within each dataset type ──────────────────────────────

corrected_rows = []

for dtype, group in results.groupby("Dataset_Type"):
    p_vals = group["P_Value_Raw"].values
    reject, p_corrected, _, _ = multipletests(p_vals, method="fdr_bh", alpha=0.05)

    group = group.copy()
    group["P_Value_BH"] = p_corrected
    group["Significant_BH"] = reject
    corrected_rows.append(group)

# results = pd.concat(corrected_rows).reset_index(drop=True)

# ── Direct comparison: Ours vs BC ────────────────────────────────────────────

head2head_records = []

for dtype, datasets in DATASET_TYPES.items():
    for dataset in datasets:
        subset = df[(df["Dataset"] == dataset) & (df["Model"].isin(MODELS))].copy()

        if subset.empty:
            continue

        diffs = subset["Ours_Acc"].values - subset["BC_Acc"].values
        n = len(diffs)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)

        if n < 2 or std_diff == 0:
            continue

        t_stat, p_val = stats.ttest_1samp(diffs, popmean=0)

        head2head_records.append({
            "Dataset_Type": dtype,
            "Dataset": dataset,
            "Comparison": "Ours_Acc vs BC_Acc",
            "N_Models": n,
            "Mean_Delta": round(mean_diff, 5),   # positive = Ours wins
            "Std_Delta": round(std_diff, 5),
            "T_Statistic": round(t_stat, 4),
            "P_Value_Raw": p_val,
        })

h2h = pd.DataFrame(head2head_records)

# BH correction within each dataset type
h2h_corrected = []
for dtype, group in h2h.groupby("Dataset_Type"):
    p_vals = group["P_Value_Raw"].values
    reject, p_corrected, _, _ = multipletests(p_vals, method="fdr_bh", alpha=0.05)
    group = group.copy()
    group["P_Value_BH"] = p_corrected
    group["Significant_BH"] = reject
    h2h_corrected.append(group)

# h2h = pd.concat(h2h_corrected).reset_index(drop=True)
# h2h["P_Value_Raw"] = h2h["P_Value_Raw"].round(5)
# h2h["P_Value_BH"] = h2h["P_Value_BH"].round(5)
h2h = h2h.sort_values(["Dataset_Type", "Dataset"]).reset_index(drop=True)

h2h.to_csv("../../results/table_outputs/ours_vs_bc_results.csv", index=False)
print("\n── Ours vs BC (direct comparison) ──")
print(h2h.to_string(index=False))

# ── Format & save ─────────────────────────────────────────────────────────────

results["P_Value_Raw"] = results["P_Value_Raw"].round(5)
# results["P_Value_BH"] = results["P_Value_BH"].round(5)

col_order = [
    "Dataset_Type", "Dataset", "Method", "N_Models",
    "Mean_Delta", "Std_Delta", "T_Statistic",
    "P_Value_Raw"] # "P_Value_BH", "Significant_BH",]
results = results[col_order].sort_values(["Dataset_Type", "Dataset", "Method"])

results.to_csv(OUTPUT_PATH, index=False)
print(f"\nResults saved to {OUTPUT_PATH}")
print(f"\n{results.to_string(index=False)}")