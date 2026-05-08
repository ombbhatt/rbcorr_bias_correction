import os
import json
import ast
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── directory layout ──────────────────────────────────────────────────────────
BASE_DIR   = "Mar-19-2026"
TASK_DIRS  = {"yn": "cc_yn", "nli": "cc_nli", "mcq": "cc_mcq"}

# ── model metadata ────────────────────────────────────────────────────────────
MODEL_FAMILIES = {
    "Falcon": ["Falcon3-3B-Base", "Falcon3-3B-Instruct",
               "Falcon3-10B-Base", "Falcon3-10B-Instruct"],
    "Gemma3": ["gemma-3-12b-pt", "gemma-3-12b-it",
               "gemma-3-27b-pt", "gemma-3-27b-it"],
    "Llama3": ["Llama-3.1-8B", "Llama-3.1-8B-Instruct",
               "Llama-3.1-70B", "Llama-3.1-70B-Instruct"],
}

SHORTHAND = {
    # Llama
    "Llama-3.1-8B":            "L-8B",
    "Llama-3.1-8B-Instruct":   "L-8B-I",
    "Llama-3.1-70B":           "L-70B",
    "Llama-3.1-70B-Instruct":  "L-70B-I",
    # Gemma
    "gemma-3-12b-pt":  "G-12B",
    "gemma-3-12b-it":  "G-12B-I",
    "gemma-3-27b-pt":  "G-27B",
    "gemma-3-27b-it":  "G-27B-I",
    # Falcon
    "Falcon3-3B-Base":      "F-3B",
    "Falcon3-3B-Instruct":  "F-3B-I",
    "Falcon3-10B-Base":     "F-10B",
    "Falcon3-10B-Instruct": "F-10B-I",
}

# ordered model list (left → right on x-axis: Llama, Gemma, Falcon)
ORDERED_MODELS = (
    MODEL_FAMILIES["Llama3"]
    + MODEL_FAMILIES["Gemma3"]
    + MODEL_FAMILIES["Falcon"]
)
ORDERED_LABELS = [SHORTHAND[m] for m in ORDERED_MODELS]

# ── task / dataset / key metadata ─────────────────────────────────────────────
# Colors for labels
YESNO_COLORS = [
    '#4e79a7',  # Muted blue
    '#f28e2b',  # Muted orange
]

NLI_COLORS = [
    '#4e79a7',  # Muted blue
    '#f28e2b',  # Muted orange
    '#59a14f',  # Muted green
]

MCQ_COLORS = [
    '#4e79a7',  # Muted blue
    '#f28e2b',  # Muted orange
    '#59a14f',  # Muted green
    '#e15759',  # Muted red
]

TASK_META = {
    "yn": {
        "datasets":    ["ARITH", "BABI", "COMPS", "EWOK"],
        "type_key":    "YESNO",
        "label_keys":  ["Yes", "No"],
        "colors":      YESNO_COLORS,
        "hlines":      [50.0],
        "legend_title": None,
    },
    "nli": {
        "datasets":    ["MNLI", "SNLI"],
        "type_key":    "NLI",
        "label_keys":  ["0", "1", "2"],
        "colors":      NLI_COLORS,
        "hlines":      [100/3, 200/3],
        "legend_title": None,
    },
    "mcq": {
        "datasets":    ["MMLU-STEM", "MMLU-HUMANITIES",
                        "MMLU-SOCIAL_SCI", "MMLU-OTHERS"],
        "type_key":    "MCQ",
        "label_keys":  ["A", "B", "C", "D"],
        "colors":      MCQ_COLORS,
        "hlines":      [25.0, 50.0, 75.0],
        "legend_title": None,
    },
}

ROW_TITLES = {
    "yn":  "Baseline Label Preferences - 2-Choice (Yes/No) Questions",
    "nli": "Baseline Label Preferences - 3-Choice (NLI) Questions",
    "mcq": "Baseline Label Preferences - 4-Choice (MCQ) Questions",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_raw_model_dist(task: str, family: str, dataset: str) -> dict:
    """
    Returns {model_name: {label: proportion, ...}, ...}
    for the given task / family / dataset.
    Proportions are in [0, 100] (percentage points).
    """
    folder   = TASK_DIRS[task]
    filename = f"instronly_{family}_{dataset}.json"
    path     = os.path.join(BASE_DIR, folder, filename)

    with open(path) as f:
        data = json.load(f)

    type_key = TASK_META[task]["type_key"]
    family_block = data["instronly"][type_key][dataset][family]

    result = {}
    for model, content in family_block.items():
        raw = content["cc"]["raw_model_dist"]
        # stored as a stringified dict — parse it safely
        if isinstance(raw, str):
            raw = ast.literal_eval(raw)
        result[model] = {k: float(v) * 100 for k, v in raw.items()}
    return result


def build_dataset_table(task: str, dataset: str) -> dict:
    """
    Merges all families into one ordered dict:
      {model_name: {label: pct, ...}}
    in the canonical ORDERED_MODELS order.
    """
    all_dists = {}
    for family in MODEL_FAMILIES:
        try:
            all_dists.update(load_raw_model_dist(task, family, dataset))
        except FileNotFoundError as e:
            print(f"  [warn] {e} — skipping")
    # reorder
    return {m: all_dists[m] for m in ORDERED_MODELS if m in all_dists}


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_stacked_bar(ax, table: dict, meta: dict, dataset_title: str):
    label_keys = meta["label_keys"]
    colors     = meta["colors"]
    hlines     = meta["hlines"]

    models = list(table.keys())
    n      = len(models)
    x      = np.arange(n)
    bar_w  = 0.65

    bottoms = np.zeros(n)
    bars    = []
    for lk, color in zip(label_keys, colors):
        vals = np.array([table[m].get(lk, 0.0) for m in models])
        b = ax.bar(x, vals, bar_w, bottom=bottoms, color=color, zorder=2)
        bars.append(b)
        bottoms += vals

    # red dashed uniform-distribution lines
    for h in hlines:
        ax.axhline(h, color="red", linewidth=0.9, linestyle="--", zorder=3)

    # formatting
    ax.set_title(dataset_title, fontsize=15, fontweight="bold", pad=3)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SHORTHAND.get(m, m) for m in models],
        rotation=40, ha="right", fontsize=11
    )
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 10))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 10)], fontsize=11)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    return bars


def make_legend(ax, meta: dict):
    patches = [
        mpatches.Patch(color=c, label=lk)
        for lk, c in zip(meta["label_keys"], meta["colors"])
    ]
    ax.legend(
        handles=patches,
        fontsize=11,
        loc="upper right",
        framealpha=0.8,
        handlelength=1.2,
        handleheight=0.9,
    )


def main():
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

    tasks      = ["yn", "nli", "mcq"]
    n_cols_row = {"yn": 4, "nli": 2, "mcq": 4}
    max_cols   = 4

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("white")

    # Use a 3-row × 4-col gridspec; NLI spans only 2 (centred) columns
    gs = fig.add_gridspec(
        3, max_cols,
        hspace=0.35,
        wspace=0.15,
        left=0.05, right=0.97,
        top=0.92, bottom=0.07,
    )

    for row_idx, task in enumerate(tasks):
        meta     = TASK_META[task]
        datasets = meta["datasets"]
        n_cols   = n_cols_row[task]
        offset   = 0

        # row super-title (centred over its own subplot band)
        # place it as a Text using figure coordinates
        row_y_top = 0.93 - row_idx * (0.86 / 3)   # approx top of each row band

        for col_idx, dataset in enumerate(datasets):
            actual_col = offset + col_idx
            if n_cols < max_cols:
                # span multiple columns so NLI plots fill the same total width
                span = max_cols // n_cols
                ax = fig.add_subplot(gs[row_idx, actual_col * span:(actual_col + 1) * span])
            else:
                ax = fig.add_subplot(gs[row_idx, actual_col])

            table = build_dataset_table(task, dataset)
            if not table:
                ax.set_visible(False)
                continue

            bars = plot_stacked_bar(ax, table, meta, dataset)  # title set inside

            # y-label only on leftmost subplot of each row
            if col_idx == 0:
                ax.set_ylabel("Label Proportion", fontsize=11)
            else:
                ax.set_ylabel("")

            # legend only on rightmost subplot of each row
            if col_idx == n_cols - 1:
                make_legend(ax, meta)

        # row super-title via fig.text
        fig.text(
            0.5, 0.96 - row_idx * 0.31,
            ROW_TITLES[task],
            ha="center", va="top",
            fontsize=18, fontweight="bold"
        )

    plt.savefig("mar19-instronly_label_distributions.png", dpi=150, bbox_inches="tight")
    print("Saved → mar19-instronly_label_distributions.png")
    plt.show()


if __name__ == "__main__":
    main()