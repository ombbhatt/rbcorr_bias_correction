import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy import stats

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("../results/comparison_mar23_instronly_n60-N60K20.csv")

# ── Model color/style map ──────────────────────────────────────────────────────
MODEL_COLORS = {
    # "gpt2-medium": "#d1c4e9",  # light purple
    # "gpt2-large":  "#7e57c2",  # medium purple
    "Llama-3.1-8B":           "#a8c8e8",   # light blue
    "Llama-3.1-8B-Instruct":  "#5ba3d0",   # medium blue
    "Llama-3.1-70B":          "#1e6fb5",   # blue
    "Llama-3.1-70B-Instruct": "#0d3d7a",   # dark blue
    "gemma-3-12b-pt":         "#a8d8a8",   # light green
    "gemma-3-12b-it":         "#5bb85b",   # medium green
    "gemma-3-27b-pt":         "#2e8b2e",   # green
    "gemma-3-27b-it":         "#1a5c1a",   # dark green
    "Falcon3-3B-Base":        "#f4b8a0",   # light orange/red
    "Falcon3-3B-Instruct":    "#e07850",   # orange
    "Falcon3-10B-Base":       "#c83c1e",   # red
    "Falcon3-10B-Instruct":   "#7a1a0a",   # dark red
}

MODEL_ORDER = list(MODEL_COLORS.keys())

# ── Panels: one representative dataset per question type + average ─────────────
# Pick one dataset per number-of-choices to mirror the figure
PANELS = [
    ("ARITH",   "ARITH (2-choice)"),
    ("MNLI",   "MNLI (3-choice)"),
    ("MMLU-HUMANITIES", "MMLU-HUMANITIES (4-choice)"),
    (None,     "ALL Datasets Avg"),
]

# ── Helper: compute before/after points for one subset ────────────────────────
def get_points(sub):
    """Returns list of (model, raw_acc, raw_tvd, rb_acc, rb_tvd)."""
    rows = []
    for _, r in sub.iterrows():
        raw_acc  = r["raw_acc"] * 100
        raw_tvd  = r["raw_tvd"]
        raw_rsd = r["raw_rsd"]
        rb_acc   = (r["raw_acc"] + r["rb_acc"]) * 100
        rb_tvd   = r["raw_tvd"] + r["rb_tvd"]
        rb_rsd = r["raw_rsd"] + r["rb_rsd"]
        rows.append((r["model"], raw_acc, raw_tvd, raw_rsd, rb_acc, rb_tvd, rb_rsd))
    return rows

def get_avg_points(df):
    """Average across all datasets per model."""
    rows = []
    for model, grp in df.groupby("model"):
        raw_acc = grp["raw_acc"].mean() * 100
        raw_tvd = grp["raw_tvd"].mean()
        raw_rsd = grp["raw_rsd"].mean()
        rb_acc  = (grp["raw_acc"] + grp["rb_acc"]).mean() * 100
        rb_tvd  = (grp["raw_tvd"] + grp["rb_tvd"]).mean()
        rb_rsd  = (grp["raw_rsd"] + grp["rb_rsd"]).mean()
        rows.append((model, raw_acc, raw_tvd, raw_rsd, rb_acc, rb_tvd, rb_rsd))
    return rows


# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 10))  # Reduce width to fit legend
axes = axes.flatten()

for ax, (dataset, title) in zip(axes, PANELS):
    if dataset is None:
        pts = get_avg_points(df)
    else:
        sub = df[df["dataset"] == dataset]
        pts = get_points(sub)

    raw_tvds, raw_accs = [], []
    rb_tvds,  rb_accs  = [], []

    for model, raw_acc, raw_tvd, raw_rsd, rb_acc, rb_tvd, rb_rsd in pts:
        color = MODEL_COLORS.get(model, "gray")

        # Dotted line connecting raw → rb for this model
        ax.plot([raw_tvd, rb_tvd], [raw_acc, rb_acc],
                linestyle="dotted", color="black", lw=0.8, zorder=2)

        # Raw = filled circle (∙)
        ax.scatter(raw_tvd, raw_acc, marker="o", s=60, color=color,
                   edgecolors=color, linewidths=0.5, zorder=3)

        # RBCorr = × marker
        ax.scatter(rb_tvd, rb_acc, marker="x", s=60, color=color,
                   linewidths=1.5, zorder=3)

        raw_tvds.append(raw_tvd); raw_accs.append(raw_acc)
        rb_tvds.append(rb_tvd);   rb_accs.append(rb_acc)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("TVD", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlim(left=-0.02)

    # set y-axis tick fontsize
    ax.tick_params(axis="y", labelsize=12)
    # set x-axis tick fontsize
    ax.tick_params(axis="x", labelsize=12)

    ax.grid(True, linestyle="--", alpha=0.3)

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_handles = []
for model in MODEL_ORDER:
    color = MODEL_COLORS[model]
    patch = mlines.Line2D([], [], color=color, marker="s", linestyle="None",
                          markersize=9, label=model,
                          markerfacecolor=color, markeredgecolor=color)
    legend_handles.append(patch)

# Marker legend (∙ vs ×)
circle_h = mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                          markersize=7, label="Before correction (∙)")
cross_h  = mlines.Line2D([], [], color="gray", marker="x", linestyle="None",
                          markersize=7, markeredgewidth=1.5, label="After RBCorr (×)")
empty_h  = mlines.Line2D([], [], color="none", marker="None", linestyle="None",
                          markersize=0, label="")  # for padding

# make exactly 4 items per row, so add dummy invisible handle if needed

fig.legend(handles=legend_handles + [circle_h, cross_h, empty_h],
           loc="center left", ncol=1, fontsize=11, labelspacing=1.5,
           frameon=True, bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

plt.suptitle("Bias (TVD ↓) vs Accuracy (% ↑): Before (∙) and After (×) RBCorr, Instronly Prompt",
             fontsize=16, fontweight="bold", y=0.98, x=0.63)

# plt.tight_layout(rect=[0, 0, 1, 1])  # Leave space for legend
# set standard tight layout:
plt.tight_layout()
plt.savefig("../results/mar23_bias_accuracy_scatter_instronly_TVD_N60K20.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to ../results/mar23_bias_accuracy_scatter_instronly_TVD_N60K20.png")