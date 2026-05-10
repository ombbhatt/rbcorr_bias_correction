import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker

# ── directory layout ──────────────────────────────────────────────────────────
BASE_DIR  = "../results/Mar-23-2026"
TASK_DIRS = {"yn": "rb_yn", "nli": "rb_nli", "mcq": "rb_mcq"}

# ── model metadata ────────────────────────────────────────────────────────────
MODEL_FAMILIES = {
    "Falcon": ["Falcon3-3B-Base", "Falcon3-3B-Instruct",
               "Falcon3-10B-Base", "Falcon3-10B-Instruct"],
    "Gemma3": ["gemma-3-12b-pt", "gemma-3-12b-it",
               "gemma-3-27b-pt", "gemma-3-27b-it"],
    "Llama3": ["Llama-3.1-8B", "Llama-3.1-8B-Instruct",
               "Llama-3.1-70B", "Llama-3.1-70B-Instruct"],
}

# ── task / dataset metadata ───────────────────────────────────────────────────
TASK_META = {
    "yn":  {"datasets": ["ARITH", "BABI", "COMPS", "EWOK"],
            "type_key": "YESNO"},
    "nli": {"datasets": ["SNLI", "MNLI"],
            "type_key": "NLI"},
    "mcq": {"datasets": ["MMLU-HUMANITIES",
                         "MMLU-SOCIAL_SCI"],
            "type_key": "MCQ"},
}

# CAL_SIZES    = [20, 40, 80, 120]
# CAL_SIZES    = [80, 120, 200, 300]
CAL_SIZES    = [24, 60, 120, 180, 240]
CAL_KEYS     = [str(s) for s in CAL_SIZES]
PROMPT_LEVEL = "fewshot"
SIGMA        = 2.0    # ±2σ variability band

# ── helper ────────────────────────────────────────────────────────────────────
def is_self_pair(key: str, dataset: str) -> bool:
    return key == f"{dataset}-from{dataset}"

# ── data loading ──────────────────────────────────────────────────────────────
def load_rb_data(task: str, family: str, dataset: str) -> dict:
    """
    Returns {model: {cal_key: {"acc": float, "std_acc": float}, ..., "raw_acc": float}}
    """
    folder   = TASK_DIRS[task]
    filename = f"{PROMPT_LEVEL}_{family}_{dataset}.json"
    path     = os.path.join(BASE_DIR, folder, filename)

    with open(path) as f:
        data = json.load(f)

    type_key     = TASK_META[task]["type_key"]
    prompt_block = data[PROMPT_LEVEL][type_key]

    valid_keys = [k for k in prompt_block if is_self_pair(k, dataset)]
    if not valid_keys:
        raise KeyError(f"No self-pair key for '{dataset}' in {path}")
    family_block = prompt_block[valid_keys[0]][family]

    result = {}
    for model, cal_dict in family_block.items():
        entry    = {}
        raw_accs = []
        for ck in CAL_KEYS:
            if ck in cal_dict:
                block     = cal_dict[ck]
                entry[ck] = {
                    "acc":     block["acc"],
                    "std_acc": block["std_acc"],
                }
                raw_accs.append(block["raw_acc"])
        entry["raw_acc"] = float(np.mean(raw_accs)) if raw_accs else np.nan
        result[model]    = entry
    return result


def gather_dataset_stats(task: str, dataset: str):
    """
    Returns:
        cal_means : {cal_key: mean acc across models}
        cal_upper : {cal_key: mean acc + 1.96 * mean(std_acc)}
        cal_lower : {cal_key: mean acc - 1.96 * mean(std_acc)}
        baseline  : mean raw_acc across all models
    """
    all_model_data = {}
    for family in MODEL_FAMILIES:
        try:
            all_model_data.update(load_rb_data(task, family, dataset))
        except (FileNotFoundError, KeyError) as e:
            print(f"  [warn] {e}")

    if not all_model_data:
        return None, None, None, None

    cal_means, cal_upper, cal_lower = {}, {}, {}
    for ck in CAL_KEYS:
        accs     = [v[ck]["acc"]     for v in all_model_data.values() if ck in v]
        std_accs = [v[ck]["std_acc"] for v in all_model_data.values() if ck in v]

        avg_acc = float(np.mean(accs))     if accs     else np.nan
        avg_std = float(np.mean(std_accs)) if std_accs else np.nan

        cal_means[ck] = avg_acc
        cal_upper[ck] = avg_acc + SIGMA * avg_std
        cal_lower[ck] = avg_acc - SIGMA * avg_std

    raw_vals = [v["raw_acc"] for v in all_model_data.values()
                if not np.isnan(v.get("raw_acc", np.nan))]
    baseline = float(np.mean(raw_vals)) if raw_vals else np.nan

    return cal_means, cal_upper, cal_lower, baseline

# ── plotting ──────────────────────────────────────────────────────────────────
BLUE_LINE  = "#1A3A8F"
BAND_COLOR = "#B0D4E3"
GRAY_DASH  = "#808080"
VLINE_X    = 60

def plot_one(ax, cal_means, cal_upper, cal_lower, baseline, title):
    xs     = CAL_SIZES
    means  = np.array([cal_means[ck] for ck in CAL_KEYS])
    uppers = np.array([cal_upper[ck] for ck in CAL_KEYS])
    lowers = np.array([cal_lower[ck] for ck in CAL_KEYS])

    ax.fill_between(xs, lowers, uppers, color=BAND_COLOR, alpha=0.55, zorder=1)
    ax.plot(xs, means, color=BLUE_LINE, linewidth=2.2, zorder=3)
    ax.axhline(baseline, color=GRAY_DASH, linewidth=1.4, linestyle="--", zorder=2)
    ax.axvline(VLINE_X,  color="black",   linewidth=0.9, linestyle=":",  zorder=2)

    y_min = min(lowers.min(), baseline)
    y_max = max(uppers.max(), baseline)
    pad   = (y_max - y_min) * 0.2
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xlim(CAL_SIZES[0], CAL_SIZES[-1])
    ax.set_xticks(CAL_SIZES)
    ax.set_xticklabels([str(s) for s in CAL_SIZES], fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
    ax.set_xlabel("Calibration set size", fontsize=10)
    if title in (f"ARITH ({PROMPT_LEVEL})", f"MMLU-STEM ({PROMPT_LEVEL})"):
        ax.set_ylabel("Accuracy", fontsize=10, labelpad=0)
    else:
        ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)


def main():
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

    all_datasets = [(task, ds)
                    for task in ["yn", "nli", "mcq"]
                    for ds in TASK_META[task]["datasets"]]

    n_cols = 4
    n_rows = int(np.ceil(len(all_datasets) / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(16, 4 * n_rows),
        gridspec_kw={"hspace": 0.35, "wspace": 0.25},
    )
    axes = axes.flatten()

    for idx, (task, dataset) in enumerate(all_datasets):
        ax = axes[idx]
        cal_means, cal_upper, cal_lower, baseline = gather_dataset_stats(task, dataset)
        if cal_means is None:
            ax.set_visible(False)
            continue
        plot_one(ax, cal_means, cal_upper, cal_lower, baseline,
                 f"{dataset}")

    for idx in range(len(all_datasets), len(axes)):
        axes[idx].set_visible(False)

    line_rb   = mlines.Line2D([], [], color=BLUE_LINE, linewidth=2.2,
                              label="RBCorr Method")
    line_base = mlines.Line2D([], [], color=GRAY_DASH, linewidth=1.4,
                              linestyle="--", label="Model baseline")
    fig.legend(handles=[line_rb, line_base], loc="upper center", ncol=2,
               fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.96),
               handlelength=2.5)

    fig.suptitle(
        f"RBCorr On Multiple Calibration Set Sizes (Instruction-only prompt)",
        fontsize=15, fontweight="bold", y=0.98,
    )

    out = f"../results/mar23-rb_calibration_{PROMPT_LEVEL}-95.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()