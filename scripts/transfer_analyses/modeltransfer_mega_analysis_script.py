# this is modeltransfer_mega_analysis_script.py:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def load_cross_model_data(results_dir='../../results'):
    """Load specific method results including cross-model transfers"""
    results_dir = Path(results_dir)
    data = {}
    
    # Look for specific method directories
    for subdir in results_dir.iterdir():
        if subdir.is_dir() and 'specific' in subdir.name:
            for json_file in subdir.glob("*.json"):
                filename = json_file.stem
                parts = filename.split('_')
                
                if len(parts) >= 3:
                    prompt_type = parts[0]
                    model_family = parts[1]
                    dataset = '_'.join(parts[2:])
                    
                    # Handle MMLU naming
                    if dataset.startswith('MMLU_SOCIAL'):
                        dataset = 'MMLU-SOCIAL_SCI'
                    elif dataset.startswith('MMLU_'):
                        dataset = dataset.replace('MMLU_', 'MMLU-', 1)
                    
                    dir_parts = subdir.name.split('_')
                    if len(dir_parts) == 5:
                        method, question_type, aggregation = dir_parts[0], dir_parts[1], dir_parts[2]

                        key = (method, question_type, aggregation, prompt_type, model_family, dataset)
                        
                        try:
                            with open(json_file, 'r') as f:
                                data[key] = json.load(f)
                        except Exception as e:
                            print(f"Error loading {json_file}: {e}")
    
    return data

def extract_cross_model_metrics(data, question_type, prompt_type, model_family, 
                                dataset, target_model, source_model, batch_size):
    """Extract metrics for cross-model transfer
    
    Args:
        source_model: If None, extracts same-model results
    """
    key = ('specific', question_type, 'per', prompt_type, model_family, dataset)
    
    if key not in data:
        return None
    
    try:
        json_data = data[key]
        
        if question_type == 'yesno':
            dataset_key = 'PER_YESNO'
            # Check for cross-model domain key
            if source_model:
                domain_key = f"{dataset}-from{dataset}_{target_model}_from{source_model}"
            else:
                domain_key = f"{dataset}-from{dataset}"
        elif question_type == 'nli':
            dataset_key = 'PER_NLI'
            if source_model:
                domain_key = f"{dataset}-from{dataset}_{target_model}_from{source_model}"
            else:
                domain_key = f"{dataset}-from{dataset}"
        else:  # mcq
            dataset_key = 'PER_MMLU'
            domain = dataset.split('-')[1] if '-' in dataset else dataset
            if source_model:
                domain_key = f"{domain}-from{domain}_{target_model}_from{source_model}"
            else:
                domain_key = f"{domain}-from{domain}"
        
        if domain_key not in json_data[prompt_type][dataset_key]:
            return None
        
        model_data = json_data[prompt_type][dataset_key][domain_key][model_family]
        
        # Get metrics for specific target model
        if target_model not in model_data:
            return None
        
        batch_size_str = str(batch_size)
        if batch_size_str not in model_data[target_model]:
            return None
        
        batch_data = model_data[target_model][batch_size_str]
        
        return {
            'median_acc': batch_data.get('median_acc', 0),
            'q25_acc': batch_data.get('q25_acc', 0),
            'q75_acc': batch_data.get('q75_acc', 0),
            'bias_median': batch_data.get('median_tvd', 0),
            'acc_mean': batch_data.get('mean_acc', 0),
            'best_acc': batch_data.get('best_run_acc', 0),
            'worst_acc': batch_data.get('worst_run_acc', 0)
        }
    
    except Exception as e:
        print(f"Error extracting cross-model metrics: {e}")
        return None

def get_model_pairs_from_family(model_family):
    """Get all model pairs within a family for cross-model analysis"""
    model_configs = {
        "Falcon": ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"],
        "Gemma3": ["gemma-3-27b-pt", "gemma-3-27b-it", "gemma-3-12b-pt", "gemma-3-12b-it"],
        "Llama3": ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"]
    }
    
    models = model_configs.get(model_family, [])
    pairs = []
    
    # Generate all pairs (target, source) where target != source
    for target in models:
        for source in models:
            if target != source:
                pairs.append((target, source))
    
    return models, pairs

def create_cross_model_transfer_table():
    """Create comprehensive table showing cross-model transfer performance"""
    data = load_cross_model_data()
    
    yn_datasets = ['ARITH', 'BABI', 'COMPS', 'EWOK']
    nli_datasets = ['MNLI', 'SNLI']
    mcq_datasets = ['MMLU-STEM', 'MMLU-HUMANITIES', 'MMLU-SOCIAL_SCI', 'MMLU-OTHERS']
    model_families = ['Falcon', 'Gemma3', 'Llama3']
    prompt_types = ['zeroshot', 'instronly', 'fewshot']
    
    results = []
    
    # Process Yes-No
    batch_size_yn = 500
    for dataset in yn_datasets:
        for prompt in prompt_types:
            for family in model_families:
                models, pairs = get_model_pairs_from_family(family)
                
                # Same-model results
                for model in models:
                    metrics = extract_cross_model_metrics(
                        data, 'yesno', prompt, family, dataset, model, None, batch_size_yn
                    )
                    
                    if metrics:
                        results.append({
                            'Question_Type': 'YN',
                            'Dataset': dataset,
                            'Model_Family': family,
                            'Target_Model': model,
                            'Source_Model': model,
                            'Transfer_Type': 'Same',
                            'Prompt': prompt,
                            'Batch_Size': batch_size_yn,
                            **metrics
                        })
                
                # Cross-model results
                for target, source in pairs:
                    metrics = extract_cross_model_metrics(
                        data, 'yesno', prompt, family, dataset, target, source, batch_size_yn
                    )
                    
                    if metrics:
                        results.append({
                            'Question_Type': 'YN',
                            'Dataset': dataset,
                            'Model_Family': family,
                            'Target_Model': target,
                            'Source_Model': source,
                            'Transfer_Type': 'Cross',
                            'Prompt': prompt,
                            'Batch_Size': batch_size_yn,
                            **metrics
                        })

    # Process NLI
    batch_size_nli = 500
    for dataset in nli_datasets:
        for prompt in prompt_types:
            for family in model_families:
                models, pairs = get_model_pairs_from_family(family)
                
                # Same-model results
                for model in models:
                    metrics = extract_cross_model_metrics(
                        data, 'nli', prompt, family, dataset, model, None, batch_size_nli
                    )
                    
                    if metrics:
                        results.append({
                            'Question_Type': 'NLI',
                            'Dataset': dataset,
                            'Model_Family': family,
                            'Target_Model': model,
                            'Source_Model': model,
                            'Transfer_Type': 'Same',
                            'Prompt': prompt,
                            'Batch_Size': batch_size_nli,
                            **metrics
                        })
                
                # Cross-model results
                for target, source in pairs:
                    metrics = extract_cross_model_metrics(
                        data, 'nli', prompt, family, dataset, target, source, batch_size_nli
                    )
                    
                    if metrics:
                        results.append({
                            'Question_Type': 'NLI',
                            'Dataset': dataset,
                            'Model_Family': family,
                            'Target_Model': target,
                            'Source_Model': source,
                            'Transfer_Type': 'Cross',
                            'Prompt': prompt,
                            'Batch_Size': batch_size_nli,
                            **metrics
                        })
    
    # Process MCQ
    batch_size_mcq = 500
    for dataset in mcq_datasets:
        for prompt in prompt_types:
            for family in model_families:
                models, pairs = get_model_pairs_from_family(family)
                
                # Same-model results
                for model in models:
                    metrics = extract_cross_model_metrics(
                        data, 'mcq', prompt, family, dataset, model, None, batch_size_mcq
                    )
                    
                    if metrics:
                        results.append({
                            'Question_Type': 'MCQ',
                            'Dataset': dataset,
                            'Model_Family': family,
                            'Target_Model': model,
                            'Source_Model': model,
                            'Transfer_Type': 'Same',
                            'Prompt': prompt,
                            'Batch_Size': batch_size_mcq,
                            **metrics
                        })
                
                # Cross-model results
                for target, source in pairs:
                    metrics = extract_cross_model_metrics(
                        data, 'mcq', prompt, family, dataset, target, source, batch_size_mcq
                    )
                    
                    if metrics:
                        results.append({
                            'Question_Type': 'MCQ',
                            'Dataset': dataset,
                            'Model_Family': family,
                            'Target_Model': target,
                            'Source_Model': source,
                            'Transfer_Type': 'Cross',
                            'Prompt': prompt,
                            'Batch_Size': batch_size_mcq,
                            **metrics
                        })
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        output_dir = Path('../../results/table_outputs/transfer_model')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_dir / 'cross_model_transfer_full.csv', index=False)
        print(f"Created full cross-model transfer table with {len(df)} rows")
        
        # Create summary aggregated by target model (average over sources and prompts)
        summary = df.groupby([
            'Question_Type', 'Dataset', 'Model_Family', 'Target_Model', 'Transfer_Type'
        ]).agg({
            'median_acc': 'mean',
            'q25_acc': 'mean',
            'q75_acc': 'mean',
            'bias_median': 'mean',
            'best_acc': 'mean',
            'worst_acc': 'mean'
        }).round(4).reset_index()
        
        summary.to_csv(output_dir / 'cross_model_transfer_summary.csv', index=False)
        print(f"Created summary table with {len(summary)} rows")
        
        return df, summary
    
    return pd.DataFrame(), pd.DataFrame()

def plot_cross_model_heatmap(df, question_type='YN', model_family='Falcon', 
                            dataset='EWOK', prompt_type='fewshot'):
    """Create heatmap showing cross-model transfer performance for a specific dataset and prompt"""
    
    df_subset = df[
        (df['Question_Type'] == question_type) & 
        (df['Model_Family'] == model_family) &
        (df['Dataset'] == dataset) &
        (df['Prompt'] == prompt_type)  # ADDED: Filter by prompt
    ]
    
    if df_subset.empty:
        print(f"No data for {question_type}, {model_family}, {dataset}, {prompt_type}")
        return
    
    # Get unique models
    models = sorted(df_subset['Target_Model'].unique())
    
    # Extract model size information
    def get_model_size(model_name):
        """Extract size from model name (e.g., '3B', '10B', '8B', '70B')"""
        import re
        match = re.search(r'(\d+)B', model_name, re.IGNORECASE)
        if match:
            return f"{match.group(1)}B"
        return ""
    
    def get_short_name(model_name):
        """Get short name (Base/Instruct) with size"""
        size = get_model_size(model_name)
        if 'Instruct' in model_name or '-it' in model_name or 'chat' in model_name:
            return f"Instruct ({size})" if size else "Instruct"
        else:
            return f"Base ({size})" if size else "Base"
    
    # Create pivot table and store same-model baseline values
    pivot_data = []
    same_model_baselines = {}
    
    for target in models:
        row = []
        # Get same-model baseline for this target
        same_model_row = df_subset[
            (df_subset['Target_Model'] == target) & 
            (df_subset['Transfer_Type'] == 'Same')
        ]
        if not same_model_row.empty:
            same_model_baselines[target] = same_model_row['median_acc'].mean()
        else:
            same_model_baselines[target] = np.nan
        
        for source in models:
            if target == source:
                # Same-model
                if not same_model_row.empty:
                    row.append(same_model_row['median_acc'].mean())
                else:
                    row.append(np.nan)
            else:
                # Cross-model
                cross_model_row = df_subset[
                    (df_subset['Target_Model'] == target) & 
                    (df_subset['Source_Model'] == source)
                ]
                if not cross_model_row.empty:
                    row.append(cross_model_row['median_acc'].mean())
                else:
                    row.append(np.nan)
        pivot_data.append(row)
    
    pivot = pd.DataFrame(pivot_data, index=models, columns=models)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(pivot.values, cmap='Blues', aspect='auto',
                   vmin=np.nanmin(pivot.values), vmax=np.nanmax(pivot.values))
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Median Accuracy', fontsize=12)
    
    # Set ticks with size-enhanced names
    display_names = [get_short_name(m) for m in models]
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(display_names, fontsize=10)
    ax.set_yticklabels(display_names, fontsize=10)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(models)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                # Determine text color based on background
                # text_color = "black" if val > 0.5 else "white"
                text_color = "white" if val > (np.nanmax(pivot.values) - np.nanmin(pivot.values)) / 2 + np.nanmin(pivot.values) else "black"
                
                if i == j:
                    # Diagonal (same-model): bold, just show accuracy
                    ax.text(j, i, f'{val:.4f}',
                           ha="center", va="center",
                           color=text_color,
                           fontsize=10, fontweight='bold')
                else:
                    # Off-diagonal (cross-model): show accuracy and delta
                    baseline = same_model_baselines[models[i]]
                    if not np.isnan(baseline):
                        delta = val - baseline
                        delta_str = f"{delta:+.4f}"  # + or - prefix
                        
                        # Main accuracy value
                        ax.text(j, i - 0.15, f'{val:.4f}',
                               ha="center", va="center",
                               color=text_color,
                               fontsize=9, fontweight='normal')
                        
                        # Delta value (smaller, below)
                        delta_color = "green" if delta > 0 else "red" if delta < 0 else text_color
                        ax.text(j, i + 0.15, delta_str,
                               ha="center", va="center",
                               color=delta_color,
                               fontsize=7, fontweight='bold', style='italic')
                    else:
                        # No baseline, just show accuracy
                        ax.text(j, i, f'{val:.4f}',
                               ha="center", va="center",
                               color=text_color,
                               fontsize=9, fontweight='normal')
    
    ax.set_xlabel('Source Model (Calibration)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Target Model (Test)', fontsize=13, fontweight='bold')
    
    question_name = "Yes-No" if question_type == "YN" else "NLI" if question_type == "NLI" else "MCQ"
    ax.set_title(
        f'{question_name} Cross-Model Transfer: {model_family} on {dataset} ({prompt_type})\n'
        f'(Diagonal = Same-Model, Off-Diagonal = Cross-Model with Δ from Same-Model)',
        fontsize=14, fontweight='bold', pad=20
    )
    
    plt.tight_layout()
    output_dir = Path('../../results/plot_outputs/transfer_model')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / f'crossmodel_heatmap_{question_type}_{model_family}_{dataset}_{prompt_type}.png',
        dpi=300, bbox_inches='tight'
    )
    print(f"Saved heatmap for {question_type}, {model_family}, {dataset}, {prompt_type}")
    plt.close()


def plot_model_size_transfer_analysis(df, question_type='YN', model_family='Llama3', prompt_type='fewshot'):
    """Analyze whether larger models can calibrate smaller models better"""
    
    df_subset = df[
        (df['Question_Type'] == question_type) & 
        (df['Model_Family'] == model_family) &
        (df['Transfer_Type'] == 'Cross') &
        (df['Prompt'] == prompt_type)  # ADDED: Filter by prompt
    ]
    
    if df_subset.empty:
        print(f"No cross-model data for {question_type}, {model_family}, {prompt_type}")
        return
    
    # Define model size ordering
    size_map = {
        'Llama-3.1-8B': 8,
        'Llama-3.1-8B-Instruct': 8,
        'Llama-3.1-70B': 70,
        'Llama-3.1-70B-Instruct': 70,
        'Falcon3-3B-Base': 3,
        'Falcon3-3B-Instruct': 3,
        'Falcon3-10B-Base': 10,
        'Falcon3-10B-Instruct': 10,
        'Qwen2.5-14B': 14,
        'Qwen2.5-14B-Instruct': 14,
        'Qwen2.5-32B': 32,
        'Qwen2.5-32B-Instruct': 32,
        'Llama-2-7b-hf': 7,
        'Llama-2-7b-chat-hf': 7,
        'Llama-2-13b-hf': 13,
        'Llama-2-13b-chat-hf': 13,
        'gemma-3-27b-pt': 27,
        'gemma-3-27b-it': 27,
        'gemma-3-12b-pt': 12,
        'gemma-3-12b-it': 12,
    }
    
    # Add size columns
    df_subset = df_subset.copy()
    df_subset['target_size'] = df_subset['Target_Model'].map(size_map)
    df_subset['source_size'] = df_subset['Source_Model'].map(size_map)
    df_subset = df_subset.dropna(subset=['target_size', 'source_size'])
    
    if df_subset.empty:
        print(f"No size information available for {model_family}, {prompt_type}")
        return
    
    # Calculate relative performance
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Source size vs accuracy (for each target size)
    ax = axes[0]
    for target_size in sorted(df_subset['target_size'].unique()):
        subset = df_subset[df_subset['target_size'] == target_size]
        ax.scatter(subset['source_size'], subset['median_acc'], 
                  label=f'Target: {int(target_size)}B', alpha=0.6, s=50)
    
    ax.set_xlabel('Source Model Size (B parameters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Median Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Does Larger Source Help?', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Size ratio vs accuracy
    ax = axes[1]
    df_subset['size_ratio'] = df_subset['source_size'] / df_subset['target_size']
    ax.scatter(df_subset['size_ratio'], df_subset['median_acc'], alpha=0.6, s=50)
    ax.axvline(x=1.0, color='red', linestyle='--', label='Same Size')
    
    ax.set_xlabel('Source/Target Size Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Median Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Size Ratio vs Performance', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{question_type} Cross-Model Transfer: {model_family} ({prompt_type})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_dir = Path('../../results/plot_outputs/transfer_model')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / f'model_size_analysis_{question_type}_{model_family}_{prompt_type}.png',
        dpi=300, bbox_inches='tight'
    )
    print(f"Saved size analysis for {question_type}, {model_family}, {prompt_type}")
    plt.close()


def plot_same_vs_cross_model_comparison(df_summary):
    """Compare same-model vs cross-model calibration performance"""

    print(df_summary)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    for idx, (qtype, datasets) in enumerate([
        ('YN', ['ARITH', 'BABI', 'COMPS', 'EWOK']),
        ('NLI', ['MNLI', 'SNLI']),
        ('MCQ', ['MMLU-STEM', 'MMLU-HUMANITIES', 'MMLU-SOCIAL_SCI', 'MMLU-OTHERS'])
    ]):
        df_subset = df_summary[df_summary['Question_Type'] == qtype]
        
        if df_subset.empty:
            continue
        
        # Plot 1: Accuracy comparison
        ax = axes[idx][0]
        
        for dataset in datasets:
            dataset_data = df_subset[df_subset['Dataset'] == dataset]
            
            same_model = dataset_data[dataset_data['Transfer_Type'] == 'Same']['median_acc']
            cross_model = dataset_data[dataset_data['Transfer_Type'] == 'Cross']['median_acc']
            
            if not same_model.empty and not cross_model.empty:
                short_name = dataset.replace('MMLU-', '')
                ax.scatter([same_model.mean()], [cross_model.mean()], 
                          label=short_name, s=100, alpha=0.7)
        
        # Diagonal line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, label='x=y')
        
        ax.set_xlabel('Same-Model Accuracy', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cross-Model Accuracy (avg)', fontsize=11, fontweight='bold')
        ax.set_title(f'{"Yes-No" if qtype == "YN" else "NLI" if qtype == "NLI" else "MCQ"} Questions',
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Degradation by model family
        ax = axes[idx][1]
        
        families = df_subset['Model_Family'].unique()
        degradations = []
        family_labels = []
        
        for family in families:
            family_data = df_subset[df_subset['Model_Family'] == family]
            same = family_data[family_data['Transfer_Type'] == 'Same']['median_acc'].mean()
            cross = family_data[family_data['Transfer_Type'] == 'Cross']['median_acc'].mean()
            
            if not np.isnan(same) and not np.isnan(cross):
                degradations.append((same - cross) * 100)
                family_labels.append(family)
        
        ax.bar(family_labels, degradations, alpha=0.7)
        ax.set_xlabel('Model Family', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy Drop (% points)', fontsize=11, fontweight='bold')
        ax.set_title(f'{"Yes-No" if qtype == "YN" else "NLI" if qtype == "NLI" else "MCQ"}: Cross-Model Degradation',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.suptitle('Same-Model vs Cross-Model Calibration Performance',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_dir = Path('../../results/plot_outputs/transfer_model')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / 'same_vs_cross_model_comparison.png',
        dpi=300, bbox_inches='tight'
    )
    print("Saved same vs cross-model comparison")
    plt.close()


if __name__ == "__main__":
    print("Creating cross-model transfer tables...")
    df_full, df_summary = create_cross_model_transfer_table()
    
    if not df_full.empty:
        # Get available combinations from the data
        available_combinations = df_full.groupby([
            'Question_Type', 'Dataset', 'Model_Family', 'Prompt'
        ]).size().reset_index()[['Question_Type', 'Dataset', 'Model_Family', 'Prompt']]
        
        print("\nGenerating heatmaps for available data combinations...")
        for _, row in available_combinations.iterrows():
            qtype = row['Question_Type']
            dataset = row['Dataset']
            family = row['Model_Family']
            prompt = row['Prompt']
            
            print(f"  Creating heatmap: {qtype}, {family}, {dataset}, {prompt}")
            plot_cross_model_heatmap(df_full, qtype, family, dataset, prompt)
        
        print("\nGenerating model size transfer analysis for available data...")
        size_combinations = df_full[df_full['Transfer_Type'] == 'Cross'].groupby([
            'Question_Type', 'Model_Family', 'Prompt'
        ]).size().reset_index()[['Question_Type', 'Model_Family', 'Prompt']]
        
        for _, row in size_combinations.iterrows():
            qtype = row['Question_Type']
            family = row['Model_Family']
            prompt = row['Prompt']
            
            print(f"  Creating size analysis: {qtype}, {family}, {prompt}")
            plot_model_size_transfer_analysis(df_full, qtype, family, prompt)
        
        print("\nCreating same vs cross-model comparison...")
        plot_same_vs_cross_model_comparison(df_summary)
        
        print("\nDone! Check table_outputs/transfer_model/ and plot_outputs/transfer_model/")
    else:
        print("No cross-model data found. Make sure you have processed cross-model results.")