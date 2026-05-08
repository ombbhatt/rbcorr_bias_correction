import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path("../../results/Mar-14-2026")
PROMPT_LEVEL = "zeroshot"
CALIB_SIZES = [20, 40, 80, 120]

# Model families and models
MODELS = {
    "Falcon": ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"],
    "Gemma3": ["gemma-3-27b-pt", "gemma-3-27b-it", "gemma-3-12b-pt", "gemma-3-12b-it"],
    "Llama3": ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"]
}

# Dataset configurations
DATASETS = {
    "yesno": {
        "datasets": ["ARITH", "BABI", "COMPS", "EWOK"],
        "specific_dir": "rb_yn",
        "batchcalib_dir": "bc_yn",
        "qtype": "YESNO",
        "domains": {"ARITH": "arith", "BABI": "babi", "COMPS": "comps", "EWOK": "all_domains"}
    },
    "nli": {
        "datasets": ["MNLI", "SNLI"],
        "specific_dir": "rb_nli",
        "batchcalib_dir": "bc_nli",
        "qtype": "NLI",
        "domains": {"MNLI": "mnli", "SNLI": "snli"}
    },
    "mcq": {
        "datasets": ["MMLU-HUMANITIES", "MMLU-OTHERS", "MMLU-SOCIAL_SCI", "MMLU-STEM"],
        "specific_dir": "rb_mcq",
        "batchcalib_dir": "bc_mcq",
        "qtype": "MMLU",
        "domains": {
            "MMLU-HUMANITIES": "HUMANITIES",
            "MMLU-OTHERS": "OTHERS",
            "MMLU-SOCIAL_SCI": "SOCIAL_SCI",
            "MMLU-STEM": "STEM"
        }
    }
}

def load_specific_data(qtype_config, dataset, model_family, model_name, calib_size):
    """Load data from specific method JSON files."""
    filename = f"{PROMPT_LEVEL}_{model_family}_{dataset}.json"
    filepath = RESULTS_DIR / qtype_config["specific_dir"] / filename
    
    if not filepath.exists():
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    try:
        # Navigate the JSON structure
        prompt_data = data[PROMPT_LEVEL]
        per_qtype = prompt_data[f"PER_{qtype_config['qtype']}"]
        if qtype_config['qtype'] == "MMLU":
            d_key = dataset.replace("MMLU-", "")
            same_condition_key = f"{d_key}-from{d_key}"
        else:
            same_condition_key = f"{dataset}-from{dataset}"
        
        if same_condition_key not in per_qtype:
            return None
        
        family_data = per_qtype[same_condition_key][model_family]
        model_data = family_data[model_name]
        calib_data = model_data[str(calib_size)]
        
        return {
            "acc": calib_data["acc"],
        }
    except (KeyError, TypeError):
        return None

def collect_data_for_dataset(qtype_config, dataset):
    """Collect all data for a specific dataset across all models and calib sizes."""
    specific_data = {size: {"acc": []} for size in CALIB_SIZES}
    
    # Iterate through all models
    for model_family, model_list in MODELS.items():
        for model_name in model_list:
            for calib_size in CALIB_SIZES:
                # Load specific data
                spec_result = load_specific_data(qtype_config, dataset, model_family, model_name, calib_size)
                if spec_result:
                    specific_data[calib_size]["acc"].append(spec_result["acc"])            
    
    # Calculate averages
    specific_avg = {}
    for size in CALIB_SIZES:
        if specific_data[size]["acc"]:
            specific_avg[size] = {
                "acc": np.mean(specific_data[size]["acc"])
            }
    
    baseline_acc = np.mean(raw_accs) if raw_accs else None
    
    return specific_avg, baseline_acc

def plot_dataset(ax, qtype_config, dataset):
    """Plot data for a single dataset."""
    specific_avg, baseline_acc = collect_data_for_dataset(qtype_config, dataset)
    
    # Prepare data for plotting
    x_vals = CALIB_SIZES
    
    # Specific method data
    specific_x = [size for size in CALIB_SIZES if size in specific_avg]
    specific_y = [specific_avg[size]["acc"] for size in specific_x]
    
    # Plot specific method (blue)
    if specific_x:
        ax.plot(specific_x, specific_y, 'b-', linewidth=2, label='RBCorr Method')
    
    # Plot baseline (grey dashed)
    if baseline_acc is not None:
        ax.axhline(y=baseline_acc, color='grey', linestyle='--', linewidth=1.5, label='Model baseline')
    
    # Set x-axis to log scale for proper spacing
    ax.set_xscale('log')
    ax.set_xticks(CALIB_SIZES)
    ax.set_xticklabels([str(x) for x in CALIB_SIZES])

    # Rotate y-ax
    ax.tick_params(axis='y', labelrotation=45)
    
    # Labels and title
    ax.set_xlabel('Calibration set size', fontsize=10)
    if qtype_config['qtype'] != "MMLU":
        ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_title(f'{dataset} (zeroshot)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

def create_comparison_plot():
    """Create the full comparison figure."""
    # fig = plt.figure(figsize=(8, 16))
    fig = plt.figure(figsize=(16, 8))
    
    # Create grid: 2 rows, 4 columns
    gs = fig.add_gridspec(2, 4, hspace=0.29, wspace=0.27)
    
    # Row 1: YesNo datasets (4 plots)
    yesno_config = DATASETS["yesno"]
    for i, dataset in enumerate(yesno_config["datasets"]):
        ax = fig.add_subplot(gs[0, i])
        plot_dataset(ax, yesno_config, dataset)

    
    # Row 2: MCQ datasets (4 plots)
    mcq_config = DATASETS["mcq"]
    for i, dataset in enumerate(mcq_config["datasets"]):
        ax = fig.add_subplot(gs[1, i])
        plot_dataset(ax, mcq_config, dataset)
    
    # Add a single legend at the top of the figure
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), 
               ncol=3, fontsize=11, frameon=True)
    
    plt.suptitle('RBCorr On Multiple Calibration Set Sizes (Zeroshot)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    return fig

if __name__ == "__main__":
    print("Loading data and creating plots...")
    fig = create_comparison_plot()
    
    # Save the figure
    output_path = Path("../../results/rb_calibration_zeroshot.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    
    plt.show()