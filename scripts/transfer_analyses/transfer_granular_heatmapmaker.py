import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Define the base directory
BASE_DIR = Path("../../results")

# Define datasets and models
YESNO_DATASETS = ["ARITH", "BABI", "COMPS", "EWOK"]
LLAMA3_MODELS = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"]
LLAMA3_MODEL_SHORTHAND = ["L3.1-8B", "L3.1-8B-IT", "L3.1-70B", "L3.1-70B-IT"]
PROMPT_LEVELS = ["zeroshot", "instronly", "fewshot"]

def load_json_file(filepath):
    """Load a JSON file and return its contents."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_metrics(data, prompt_level, qtype, config_key, model_family, model_name, calib_size="500"):
    """
    Extract raw and median metrics from nested JSON structure.
    
    Returns: (raw_acc, raw_tvd, median_acc, median_tvd)
    """
    try:
        nested = data[prompt_level][f"PER_{qtype}"][config_key][model_family][model_name][calib_size]
        return (
            nested["raw_acc"],
            nested["raw_tvd"],
            nested["median_acc"],
            nested["median_tvd"]
        )
    except KeyError as e:
        print(f"Warning: Could not find data for {config_key}, {model_name}")
        return None

def get_cross_dataset_data():
    """Get cross-dataset transfer data for yesno datasets with Llama-3.1-8B, zeroshot."""
    folder = BASE_DIR / "specific_yesno_per_median_TVD"
    n = len(YESNO_DATASETS)
    acc_deltas = np.zeros((n, n))
    tvd_deltas = np.zeros((n, n))
    
    for i, target_dataset in enumerate(YESNO_DATASETS):
        for j, source_dataset in enumerate(YESNO_DATASETS):
            # Construct filename and config key
            filename = f"zeroshot_Llama3_{target_dataset}.json"
            filepath = folder / filename
            
            if source_dataset == target_dataset:
                config_key = f"{target_dataset}-from{source_dataset}"
            else:
                config_key = f"{target_dataset}-from{source_dataset}"
            
            # Load and extract data
            data = load_json_file(filepath)
            metrics = extract_metrics(data, "zeroshot", "YESNO", config_key, "Llama3", "Llama-3.1-8B")
            
            if metrics:
                raw_acc, raw_tvd, median_acc, median_tvd = metrics
                acc_deltas[i, j] = median_acc - raw_acc
                tvd_deltas[i, j] = median_tvd - raw_tvd
    
    return acc_deltas, tvd_deltas

def get_cross_model_data():
    """Get cross-model transfer data for Llama3 models with EWOK, zeroshot."""
    folder = BASE_DIR / "specific_yesno_per_median_TVD"
    filename = "zeroshot_Llama3_EWOK.json"
    filepath = folder / filename
    data = load_json_file(filepath)
    
    n = len(LLAMA3_MODELS)
    acc_deltas = np.zeros((n, n))
    tvd_deltas = np.zeros((n, n))
    
    for i, target_model in enumerate(LLAMA3_MODELS):
        for j, source_model in enumerate(LLAMA3_MODELS):
            # Construct config key
            if source_model == target_model:
                config_key = "EWOK-fromEWOK"
            else:
                config_key = f"EWOK-fromEWOK_{target_model}_from{source_model}"
            
            # Extract data
            metrics = extract_metrics(data, "zeroshot", "YESNO", config_key, "Llama3", target_model)
            
            if metrics:
                raw_acc, raw_tvd, median_acc, median_tvd = metrics
                acc_deltas[i, j] = median_acc - raw_acc
                tvd_deltas[i, j] = median_tvd - raw_tvd
    
    return acc_deltas, tvd_deltas

def get_cross_prompt_data():
    """Get cross-prompt transfer data for Llama-3.1-8B with EWOK."""
    folder = BASE_DIR / "specific_yesno_per_median_TVD"
    n = len(PROMPT_LEVELS)
    acc_deltas = np.zeros((n, n))
    tvd_deltas = np.zeros((n, n))
    
    for i, target_prompt in enumerate(PROMPT_LEVELS):
        # Load the file for the target prompt
        filename = f"{target_prompt}_Llama3_EWOK.json"
        filepath = folder / filename
        data = load_json_file(filepath)
        
        for j, source_prompt in enumerate(PROMPT_LEVELS):
            # Construct config key
            if source_prompt == target_prompt:
                config_key = "EWOK-fromEWOK"
            else:
                config_key = f"EWOK-fromEWOK_{target_prompt}_from{source_prompt}"
            
            # Extract data
            metrics = extract_metrics(data, target_prompt, "YESNO", config_key, "Llama3", "Llama-3.1-8B")
            
            if metrics:
                raw_acc, raw_tvd, median_acc, median_tvd = metrics
                acc_deltas[i, j] = median_acc - raw_acc
                tvd_deltas[i, j] = median_tvd - raw_tvd
    
    return acc_deltas, tvd_deltas

def create_heatmap(ax, acc_deltas, tvd_deltas, row_labels, col_labels, title, cmap_name, fig):
    """Create a single heatmap on the given axis."""
    # Plot heatmap with diverging matplotlib colormaps
    im = ax.imshow(acc_deltas, cmap=cmap_name, aspect='auto', vmin=acc_deltas.min(), vmax=acc_deltas.max())
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=90, ha="center", rotation_mode="anchor")

    # Put xticks at the top instead of bottom
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # add some space between the ticklable and the tick
    ax.tick_params(axis='x', pad=1) 
    ax.tick_params(axis='y', pad=8)
    # set fontsize
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    
    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            acc_delta = acc_deltas[i, j]
            tvd_delta = tvd_deltas[i, j]
            
            # Determine text color based on background intensity
            # Normalize the value to 0-1 range
            normalized_val = (acc_delta - acc_deltas.min()) / (acc_deltas.max() - acc_deltas.min())
            text_color = 'white' if normalized_val < 0.4 or normalized_val > 0.7 else 'black'
            
            # Format the text
            text = f"ΔA: {acc_delta:.3f}\nΔT: {tvd_delta:.3f}"
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=14)
    
    # Set title and labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=18)
    ax.set_xlabel("Source", fontsize=14, labelpad=5)
    ax.set_ylabel("Target", fontsize=14, labelpad=5)
    
    # Add colorbar below the heatmap
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.03, aspect=30, shrink=1.0)
    cbar.set_label('ΔAcc from Baseline Model', fontsize=14)
    if 'Cross-Model' in title:
        cbar.set_label('ΔAcc from Baseline TARGET Model', fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    
    return im

# Main execution
def main():
    # Get data for all three transfer types
    print("Loading cross-dataset data...")
    cd_acc, cd_tvd = get_cross_dataset_data()
    
    print("Loading cross-model data...")
    cm_acc, cm_tvd = get_cross_model_data()
    
    print("Loading cross-prompt data...")
    cp_acc, cp_tvd = get_cross_prompt_data()
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), gridspec_kw={'width_ratios': [4, 4, 3]})
    
    # Cross-Dataset Heatmap
    create_heatmap(
        axes[0], cd_acc, cd_tvd,
        YESNO_DATASETS, YESNO_DATASETS,
        "Cross-Dataset Transfer\n(Llama-3.1-8B, zeroshot)",
        'RdYlBu',
        fig
    )
    
    # Cross-Model Heatmap
    create_heatmap(
        axes[1], cm_acc, cm_tvd,
        LLAMA3_MODEL_SHORTHAND, LLAMA3_MODEL_SHORTHAND,
        "Cross-Model Transfer\n(EWOK, zeroshot)",
        'RdYlBu',
        fig
    )
    
    # Cross-Prompt Heatmap (adjust width)
    create_heatmap(
        axes[2], cp_acc, cp_tvd,
        PROMPT_LEVELS, PROMPT_LEVELS,
        "Cross-Prompt Transfer\n(Llama-3.1-8B, EWOK)",
        'RdYlBu',
        fig
    )
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save figure
    output_path = Path("../../results/plot_outputs/transfer_heatmap_yn_zeroshot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmaps saved to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()