import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define model families and their models
FALCON_MODELS = ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"]
GEMMA3_MODELS = ["gemma-3-12b-pt", "gemma-3-12b-it", "gemma-3-27b-pt", "gemma-3-27b-it"]
LLAMA3_MODELS = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"]

# Define color gradients for each family
LLAMA_COLORS = ['#a6cee3', '#6baed6', '#3182bd', '#08519c']  # Light to dark blue
GEMMA_COLORS = ['#a1d99b', '#74c476', '#31a354', '#006d2c']  # Light to dark green
FALCON_COLORS = ['#fc9272', '#fb6a4a', '#de2d26', '#a50f15']  # Light to dark red

# Create color mapping
MODEL_COLORS = {}
for i, model in enumerate(LLAMA3_MODELS):
    MODEL_COLORS[model] = LLAMA_COLORS[i]
for i, model in enumerate(GEMMA3_MODELS):
    MODEL_COLORS[model] = GEMMA_COLORS[i]
for i, model in enumerate(FALCON_MODELS):
    MODEL_COLORS[model] = FALCON_COLORS[i]

ALL_MODELS = LLAMA3_MODELS + GEMMA3_MODELS + FALCON_MODELS

# Dataset configurations
YESNO_DATASETS = ["ARITH", "BABI", "COMPS", "EWOK"]
NLI_DATASETS = ["MNLI", "SNLI"]
MCQ_DATASETS = ["MMLU-HUMANITIES", "MMLU-OTHERS", "MMLU-SOCIAL_SCI", "MMLU-STEM"]

def load_json_data(results_dir, qtype, promptlevel, modelfamily, dataset):
    """Load JSON data for a specific configuration."""
    qtype_map = {
        'yesno': 'specific_yesno_per_median_TVD',
        'nli': 'specific_nli_per_median_TVD',
        'mcq': 'specific_mcq_per_median_TVD'
    }
    
    subfolder = qtype_map[qtype]
    filename = f"{promptlevel}_{modelfamily}_{dataset}.json"
    filepath = os.path.join(results_dir, subfolder, filename)
    
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_model_stats(data, promptlevel, qtype, dataset, modelfamily, model):
    """Extract raw_acc, raw_tvd, median_acc, median_tvd for a specific model."""
    qtype_map = {'yesno': 'YESNO', 'nli': 'NLI', 'mcq': 'MMLU'}
    per_qtype = f"PER_{qtype_map[qtype]}"
    if qtype != 'mcq':
        same_condition = f"{dataset}-from{dataset}"
    else:
        # remove the MMLU- prefix for mcq datasets
        dataset_short = dataset.replace("MMLU-", "")
        same_condition = f"{dataset_short}-from{dataset_short}"
    
    try:
        stats = data[promptlevel][per_qtype][same_condition][modelfamily][model]["100"]
        return {
            'raw_acc': stats['raw_acc'] * 100,  # Convert to percentage
            'raw_tvd': stats['raw_tvd'],
            'median_acc': stats['median_acc'] * 100,  # Convert to percentage
            'median_tvd': stats['median_tvd']
        }
    except KeyError as e:
        print(f"Warning: Could not extract stats for {model} on {dataset}: {e}")
        return None

def get_model_family(model):
    """Determine which family a model belongs to."""
    if model in LLAMA3_MODELS:
        return "Llama3"
    elif model in GEMMA3_MODELS:
        return "Gemma3"
    elif model in FALCON_MODELS:
        return "Falcon"
    return None

def collect_dataset_stats(results_dir, promptlevel, qtype, dataset):
    """Collect statistics for all models for a specific dataset."""
    model_stats = {}
    
    for model in ALL_MODELS:
        modelfamily = get_model_family(model)
        data = load_json_data(results_dir, qtype, promptlevel, modelfamily, dataset)
        
        if data is None:
            continue
            
        stats = extract_model_stats(data, promptlevel, qtype, dataset, modelfamily, model)
        if stats is not None:
            model_stats[model] = stats
    
    return model_stats

def collect_averaged_stats(results_dir, promptlevel):
    """Collect averaged statistics across all datasets for all models."""
    all_qtypes = [
        ('yesno', YESNO_DATASETS),
        ('nli', NLI_DATASETS),
        ('mcq', MCQ_DATASETS)
    ]
    
    model_accumulated = {model: {'raw_acc': [], 'raw_tvd': [], 
                                  'median_acc': [], 'median_tvd': []} 
                         for model in ALL_MODELS}
    
    for qtype, datasets in all_qtypes:
        for dataset in datasets:
            for model in ALL_MODELS:
                modelfamily = get_model_family(model)
                data = load_json_data(results_dir, qtype, promptlevel, modelfamily, dataset)
                
                if data is None:
                    continue
                    
                stats = extract_model_stats(data, promptlevel, qtype, dataset, modelfamily, model)
                if stats is not None:
                    model_accumulated[model]['raw_acc'].append(stats['raw_acc'])
                    model_accumulated[model]['raw_tvd'].append(stats['raw_tvd'])
                    model_accumulated[model]['median_acc'].append(stats['median_acc'])
                    model_accumulated[model]['median_tvd'].append(stats['median_tvd'])
    
    # Average the accumulated values
    model_stats = {}
    for model, values in model_accumulated.items():
        if len(values['raw_acc']) > 0:
            model_stats[model] = {
                'raw_acc': np.mean(values['raw_acc']),
                'raw_tvd': np.mean(values['raw_tvd']),
                'median_acc': np.mean(values['median_acc']),
                'median_tvd': np.mean(values['median_tvd'])
            }
    
    return model_stats

def plot_scatterplot(ax, model_stats, title):
    """Plot a single scatterplot with raw and median points."""
    for model in ALL_MODELS:
        if model not in model_stats:
            continue
            
        stats = model_stats[model]
        color = MODEL_COLORS[model]
        
        # Plot raw values as solid circle
        ax.scatter(stats['raw_tvd'], stats['raw_acc'], 
                  c=color, s=100, marker='o', edgecolors='black', linewidths=0.5,
                  zorder=3)
        
        # Plot median values as X mark
        ax.scatter(stats['median_tvd'], stats['median_acc'], 
                  c=color, s=100, marker='x', linewidths=2,
                  zorder=3)
        
        # Draw dotted line connecting them
        ax.plot([stats['raw_tvd'], stats['median_tvd']], 
               [stats['raw_acc'], stats['median_acc']], 
               'k:', linewidth=1, zorder=2)
    
    # Calculate dynamic axis limits
    if model_stats:
        all_tvd = [stats['raw_tvd'] for stats in model_stats.values()] + \
                [stats['median_tvd'] for stats in model_stats.values()]
        all_acc = [stats['raw_acc'] for stats in model_stats.values()] + \
                [stats['median_acc'] for stats in model_stats.values()]
        
        max_tvd = max(all_tvd)
        min_acc = min(all_acc)
        
        # Round up to nearest 0.1 for TVD
        tvd_upper = np.ceil(max_tvd * 10) / 10
        # Round down to nearest 10 for accuracy
        acc_lower = np.floor(min_acc / 10) * 10
        
        ax.set_xlim(0, tvd_upper)
        ax.set_ylim(acc_lower, 90)
        ax.set_xticks(np.arange(0, tvd_upper + 0.1, 0.1))
        ax.set_yticks(np.arange(acc_lower, 100, 10))
    else:
        # Fallback if no data
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_yticks(np.arange(0, 110, 10))

    # set xticks and yticks font size
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    ax.set_xlabel('TVD', fontsize=15)
    ax.set_ylabel('Accuracy (%)', fontsize=15)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

def create_visualization(results_dir, output_file='../../results/plot_outputs/specific_correction_scatterplot.png'):
    """Create the complete 2x2 scatterplot visualization."""
    fig = plt.figure(figsize=(16, 14))
    
    # Create 2x2 grid for scatterplots
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.13], hspace=0.25, wspace=0.15,
                         left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    axes = [
        fig.add_subplot(gs[0, 0]),  # Top-left
        fig.add_subplot(gs[0, 1]),  # Top-right
        fig.add_subplot(gs[1, 0]),  # Bottom-left
        fig.add_subplot(gs[1, 1])   # Bottom-right
    ]
    
    promptlevel = "fewshot"
    
    # Top-left: EWOK
    print("Processing EWOK...")
    ewok_stats = collect_dataset_stats(results_dir, promptlevel, 'yesno', 'EWOK')
    plot_scatterplot(axes[0], ewok_stats, f'{promptlevel.capitalize()} EWOK (2-choice)')
    
    # Top-right: SNLI
    print("Processing SNLI...")
    snli_stats = collect_dataset_stats(results_dir, promptlevel, 'nli', 'SNLI')
    plot_scatterplot(axes[1], snli_stats, f'{promptlevel.capitalize()} SNLI (3-choice)')

    # Bottom-left: MMLU-STEM
    print("Processing MMLU-STEM...")
    mmlu_stats = collect_dataset_stats(results_dir, promptlevel, 'mcq', 'MMLU-OTHERS')
    plot_scatterplot(axes[2], mmlu_stats, f'{promptlevel.capitalize()} MMLU-OTHERS (4-choice)')
    
    # Bottom-right: ALL Datasets Average
    print("Processing ALL datasets average...")
    avg_stats = collect_averaged_stats(results_dir, promptlevel)
    plot_scatterplot(axes[3], avg_stats, f'{promptlevel.capitalize()} ALL Datasets Avg')
    
    # Create legend at the bottom
    legend_ax = fig.add_subplot(gs[2, :])
    legend_ax.axis('off')
    
    # Create legend handles
    from matplotlib.patches import Patch
    legend_elements = []
    for model in ALL_MODELS:
        legend_elements.append(Patch(facecolor=MODEL_COLORS[model], 
                                    edgecolor='black', label=model))
    
    # Split into two rows
    # legend1 should have elements 0,1,4,5,8,9 from the legend_elements
    legend1_elements = [legend_elements[i] for i in [0,1,4,5,8,9]]
    legend1 = legend_ax.legend(handles=legend1_elements, 
                              loc='upper left', ncol=6, frameon=True,
                              fontsize=13, bbox_to_anchor=(-0.01, 1.2))
    legend_ax.add_artist(legend1)
    # legend2 should have elements 2,3,6,7,10,11 from the legend_elements
    legend2_elements = [legend_elements[i] for i in [2,3,6,7,10,11]]
    legend2 = legend_ax.legend(handles=legend2_elements, 
                              loc='upper left', ncol=6, frameon=True,
                              fontsize=13, bbox_to_anchor=(-0.025, 0.7))
        
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    # Set the results directory
    results_dir = Path("../../results")
    
    # Create the visualization
    create_visualization(results_dir)