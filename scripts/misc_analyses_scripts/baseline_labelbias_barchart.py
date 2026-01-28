import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from pathlib import Path

# Define model families and their models
FALCON_MODELS = ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"]
GEMMA3_MODELS = ["gemma-3-12b-pt", "gemma-3-12b-it", "gemma-3-27b-pt", "gemma-3-27b-it"]
LLAMA3_MODELS = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"]

ALL_MODELS = LLAMA3_MODELS + GEMMA3_MODELS + FALCON_MODELS

# Dataset configurations
YESNO_DATASETS = ["ARITH", "BABI", "COMPS", "EWOK"]
NLI_DATASETS = ["MNLI", "SNLI"]
MCQ_DATASETS = ["MMLU-STEM", "MMLU-HUMANITIES", "MMLU-SOCIAL_SCI", "MMLU-OTHERS"]

PROMPT_LEVEL = "fewshot"
CALIB_SIZE = "100"

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

def get_model_family(model):
    """Determine which family a model belongs to."""
    if model in LLAMA3_MODELS:
        return "Llama3"
    elif model in GEMMA3_MODELS:
        return "Gemma3"
    elif model in FALCON_MODELS:
        return "Falcon"
    return None

def load_json_data(results_dir, qtype, modelfamily, dataset):
    """Load JSON data for a specific configuration."""
    qtype_map = {
        'yesno': 'specific_yesno_per_median_TVD',
        'nli': 'specific_nli_per_median_TVD',
        'mcq': 'specific_mcq_per_median_TVD'
    }
    
    subfolder = qtype_map[qtype]
    filename = f"{PROMPT_LEVEL}_{modelfamily}_{dataset}.json"
    filepath = os.path.join(results_dir, subfolder, filename)
    
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_raw_model_dist(data, qtype, dataset, modelfamily, model):
    """Extract raw_model_dist for a specific model."""
    qtype_map = {'yesno': 'YESNO', 'nli': 'NLI', 'mcq': 'MMLU'}
    per_qtype = f"PER_{qtype_map[qtype]}"
    if qtype != 'mcq':
        same_condition = f"{dataset}-from{dataset}"
    else:
        # remove "MMLU-" prefix for config key
        dataset_key = dataset.replace("MMLU-", "")
        same_condition = f"{dataset_key}-from{dataset_key}"
                
    try:
        stats = data[PROMPT_LEVEL][per_qtype][same_condition][modelfamily][model][CALIB_SIZE]
        raw_model_dist_str = stats['raw_model_dist']
        
        # Parse the string representation of dict
        # It's in format: "{'Label1': 0.5, 'Label2': 0.5}"
        raw_model_dist = eval(raw_model_dist_str)
        
        return raw_model_dist
    except KeyError as e:
        print(f"Warning: Could not extract raw_model_dist for {model} on {dataset}: {e}")
        return None
    except Exception as e:
        print(f"Error parsing raw_model_dist for {model} on {dataset}: {e}")
        return None

def collect_label_distributions(results_dir, qtype, datasets):
    """Collect label distributions for all models across given datasets."""
    # Structure: {dataset: {model: {label: proportion}}}
    all_distributions = {}
    
    for dataset in datasets:
        all_distributions[dataset] = {}
        
        for model in ALL_MODELS:
            modelfamily = get_model_family(model)
            data = load_json_data(results_dir, qtype, modelfamily, dataset)
            
            if data is None:
                continue
            
            dist = extract_raw_model_dist(data, qtype, dataset, modelfamily, model)
            if dist is not None:
                all_distributions[dataset][model] = dist
    
    return all_distributions

def plot_stacked_bars(ax, distributions, datasets, colors, uniform_value, ylabel, title):
    """Plot grouped stacked bar chart."""
    n_datasets = len(datasets)
    n_models = len(ALL_MODELS)
    
    # Calculate positions
    group_width = n_models + 1  # Models + gap between groups
    x_positions = []
    dataset_centers = []
    
    for i, dataset in enumerate(datasets):
        group_start = i * group_width
        for j in range(n_models):
            x_positions.append(group_start + j)
        dataset_centers.append(group_start + (n_models - 1) / 2)
    
    # Get label keys from first available distribution
    label_keys = None
    for dataset in datasets:
        for model in ALL_MODELS:
            if model in distributions[dataset]:
                label_keys = list(distributions[dataset][model].keys())
                break
        if label_keys:
            break
    
    if not label_keys:
        print("Warning: No valid distributions found")
        return
    
    # Sort label keys for consistent ordering
    label_keys = sorted(label_keys)
    if 'Yes' in label_keys and 'No' in label_keys:
        label_keys = ['Yes', 'No']
    
    bar_width = 0.8
    
    # Plot stacked bars
    for dataset_idx, dataset in enumerate(datasets):
        for model_idx, model in enumerate(ALL_MODELS):
            if model not in distributions[dataset]:
                continue
            
            pos_idx = dataset_idx * n_models + model_idx
            x_pos = x_positions[pos_idx]
            
            dist = distributions[dataset][model]
            bottom = 0
            
            for label_idx, label in enumerate(label_keys):
                value = dist.get(label, 0)
                ax.bar(x_pos, value, bar_width, bottom=bottom, 
                      color=colors[label_idx], edgecolor='white', linewidth=0.5)
                bottom += value
    
    # Add reference line for uniform distribution
    # ax.axhline(y=uniform_value, color='red', linestyle='--', linewidth=1.5, 
    #           label=f'Uniform ({uniform_value*100:.1f}%)', alpha=0.7)

    # Add additional reference lines at every uniform_value interval
    for i in range(1, int(1 / uniform_value)):
        y = i * uniform_value
        ax.axhline(y=y, color='red', linestyle='--', linewidth=1.25, alpha=0.7)
    
    # Set x-axis labels
    ax.set_xticks(x_positions)
    model_labels = []
    for _ in datasets:
        for model in ALL_MODELS:
            # Abbreviated model names
            if "Llama" in model:
                if "70B" in model:
                    label = "L-70B-I" if "Instruct" in model else "L-70B"
                else:
                    label = "L-8B-I" if "Instruct" in model else "L-8B"
            elif "gemma" in model:
                if "27b" in model:
                    label = "G-27B-I" if "it" in model else "G-27B"
                else:
                    label = "G-12B-I" if "it" in model else "G-12B"
            elif "Falcon" in model:
                if "10B" in model:
                    label = "F-10B-I" if "Instruct" in model else "F-10B"
                else:
                    label = "F-3B-I" if "Instruct" in model else "F-3B"
            else:
                label = model[:8]
            model_labels.append(label)
    
    ax.set_xticklabels(model_labels, rotation=40, ha='right', fontsize=13)
    
    # Add dataset labels
    for i, dataset in enumerate(datasets):
        center = dataset_centers[i]
        # Add vertical separator between groups
        if i > 0:
            separator_x = i * group_width - 0.5
            ax.axvline(x=separator_x, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
        
        # Add dataset label at top
        ax.text(center, 1.02, dataset, ha='center', va='bottom', 
               fontsize=16, fontweight='bold', transform=ax.get_xaxis_transform())
    
    # Styling
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f'{int(y*100)}%' for y in np.arange(0, 1.1, 0.1)], fontsize=14)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=27)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim(-0.5, x_positions[-1] + 1)
    
    # Create legend for labels
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=label) 
                      for i, label in enumerate(label_keys)]
    # legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', 
                                    #  label=f'Uniform ({uniform_value*100:.1f}%)'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=13, framealpha=0.9)

def create_visualization(results_dir, output_file='../../results/plot_outputs/baseline_labelbias_barchart.png'):
    """Create the complete visualization with three stacked bar charts."""
    
    print("Collecting 2-choice (Yes/No) distributions...")
    yesno_dists = collect_label_distributions(results_dir, 'yesno', YESNO_DATASETS)
    
    print("Collecting 3-choice (NLI) distributions...")
    nli_dists = collect_label_distributions(results_dir, 'nli', NLI_DATASETS)
    
    print("Collecting 4-choice (MCQ) distributions...")
    mcq_dists = collect_label_distributions(results_dir, 'mcq', MCQ_DATASETS)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(17, 13))
    fig.subplots_adjust(hspace=0.3, top=1.15, bottom=0.01, left=0.05, right=0.97)
    
    # Plot 2-choice
    plot_stacked_bars(axes[0], yesno_dists, YESNO_DATASETS, YESNO_COLORS, 
                     0.5, 'Label Proportion', 
                     'Baseline Label Preferences - 2-Choice (Yes/No) Questions')
    
    # Plot 3-choice
    plot_stacked_bars(axes[1], nli_dists, NLI_DATASETS, NLI_COLORS, 
                     1/3, 'Label Proportion', 
                     'Baseline Label Preferences - 3-Choice (NLI) Questions')
    
    # Plot 4-choice
    plot_stacked_bars(axes[2], mcq_dists, MCQ_DATASETS, MCQ_COLORS, 
                     0.25, 'Label Proportion', 
                     'Baseline Label Preferences - 4-Choice (MCQ) Questions')
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    # results_dir = os.path.expanduser("~/scratch/yes-bias-in-llms/results")
    results_dir = Path("../../results")
    create_visualization(results_dir)