import csv

# --- Model name mapping ---
MODEL_MAP = {
    "Falcon3-3B-Base":        "F-3B",
    "Falcon3-3B-Instruct":    "F-3B-I",
    "Falcon3-10B-Base":       "F-10B",
    "Falcon3-10B-Instruct":   "F-10B-I",
    "Llama-3.1-8B":           "L-8B",
    "Llama-3.1-8B-Instruct":  "L-8B-I",
    "Llama-3.1-70B":          "L-70B",
    "Llama-3.1-70B-Instruct": "L-70B-I",
    "gemma-3-27b-pt":         "G-27B",
    "gemma-3-27b-it":         "G-27B-I",
    "gemma-3-12b-pt":         "G-12B",
    "gemma-3-12b-it":         "G-12B-I",
}

DATASET_MAP = {
    "ARITH":           "ARITH",
    "BABI":            "BABI",
    "COMPS":           "COMPS",
    "EWOK":            "EWOK",
    "SNLI":            "SNLI",
    "MNLI":            "MNLI",
    "MMLU-HUMANITIES": "HUMANITIES",
    "MMLU-OTHERS":     "OTHERS",
    "MMLU-SOCIAL_SCI": "SOCIAL SCI.",
    "MMLU-STEM":       "STEM",
}

CHOICE_MAP = {
    "ARITH":       "2-Choice",
    "BABI":        "2-Choice",
    "COMPS":       "2-Choice",
    "EWOK":        "2-Choice",
    "SNLI":        "3-Choice",
    "MNLI":        "3-Choice",
    "HUMANITIES":  "4-Choice",
    "OTHERS":      "4-Choice",
    "SOCIAL SCI.": "4-Choice",
    "STEM":        "4-Choice",
}

def fmt(val, is_delta=False):
    try:
        f = float(val)
        return f"${f:+.4f}$" if is_delta else f"${f:.4f}$"
    except ValueError:
        return val

def load_csv(path):
    seen_order, data = [], {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['dataset'], row['model'])
            data[key] = row
            if key not in seen_order:
                seen_order.append(key)
    return seen_order, data

def build_table(path):
    seen_order, data = load_csv(path)

    datasets, ds_models = [], {}
    for (ds, mdl) in seen_order:
        if ds not in ds_models:
            datasets.append(ds)
            ds_models[ds] = []
        ds_models[ds].append(mdl)

    lines = []

    # --- Preamble comment ---
    lines.append(r"% Requires \usepackage{longtable}, \usepackage{multirow} in preamble")
    lines.append(r"% longtable does NOT work inside \resizebox or table float.")
    lines.append(r"% To scale columns, adjust \tabcolsep or use \small/\footnotesize before the env.")
    lines.append(r"{\footnotesize")  # shrink font to help fit width
    lines.append(r"\begin{longtable}{|c|c|ccc|ccc|ccc|}")

    # --- Caption / label (optional, comment out if not needed) ---
    lines.append(r"\caption{Full results across all datasets and models.} \label{tab:full_results} \\")

    # --- Header (first page) ---
    lines.append(r"\hline")
    lines.append(
        r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Model} & "
        r"\multicolumn{3}{c|}{\begin{tabular}[c]{@{}c@{}}CC\\ $\Delta$\end{tabular}} & "
        r"\multicolumn{3}{c|}{\begin{tabular}[c]{@{}c@{}}BC\\ $\Delta$\end{tabular}} & "
        r"\multicolumn{3}{c|}{\begin{tabular}[c]{@{}c@{}}RBCorr (ours)\\ $\Delta$\end{tabular}} \\"
    )
    lines.append(r"\cline{3-11}")
    sub = r"\multicolumn{1}{c|}{Acc.} & \multicolumn{1}{c|}{TVD} & RSD"
    lines.append(f" & & {sub} & {sub} & {sub} \\\\")
    lines.append(r"\hline")
    lines.append(r"\endfirsthead")

    # --- Header (continuation pages) ---
    lines.append(r"\hline")
    lines.append(
        r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Model} & "
        r"\multicolumn{3}{c|}{\begin{tabular}[c]{@{}c@{}}CC\\ $\Delta$\end{tabular}} & "
        r"\multicolumn{3}{c|}{\begin{tabular}[c]{@{}c@{}}BC\\ $\Delta$\end{tabular}} & "
        r"\multicolumn{3}{c|}{\begin{tabular}[c]{@{}c@{}}RBCorr (ours)\\ $\Delta$\end{tabular}} \\"
    )
    lines.append(r"\cline{3-11}")
    lines.append(f" & & {sub} & {sub} & {sub} \\\\")
    lines.append(r"\hline")
    lines.append(r"\endhead")

    # --- Footer on all pages except last ---
    lines.append(r"\hline \multicolumn{11}{|r|}{\textit{Continued on next page}} \\ \hline")
    lines.append(r"\endfoot")

    # --- Footer on last page ---
    lines.append(r"\hline")
    lines.append(r"\endlastfoot")

    # --- Data rows ---
    for ds in datasets:
        models = ds_models[ds]
        n = len(models)
        ds_display = DATASET_MAP.get(ds, ds)
        choice = CHOICE_MAP.get(ds_display, CHOICE_MAP.get(ds, ""))
        ds_cell = r"\begin{tabular}[c]{@{}c@{}}" + ds_display + r"\\ (" + choice + r")\end{tabular}"

        for i, mdl in enumerate(models):
            row = data[(ds, mdl)]
            short = MODEL_MAP.get(mdl, mdl)

            cells = (
                f"{fmt(row['cc_acc'],True)} & {fmt(row['cc_tvd'],True)} & {fmt(row['cc_rsd'],True)} & "
                f"{fmt(row['bc_acc'],True)} & {fmt(row['bc_tvd'],True)} & {fmt(row['bc_rsd'],True)} & "
                f"{fmt(row['rb_acc'],True)} & {fmt(row['rb_tvd'],True)} & {fmt(row['rb_rsd'],True)}"
            )

            ds_col = f"\\multirow{{{n}}}{{*}}{{{ds_cell}}}" if i == 0 else ""
            lines.append(f"{ds_col} & {short} & {cells} \\\\")
            lines.append(r"\hline" if i == n - 1 else r"\cline{2-14}")

    lines.append(r"\end{longtable}")
    lines.append(r"}")  # close \footnotesize

    return "\n".join(lines)

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    print(build_table(path))