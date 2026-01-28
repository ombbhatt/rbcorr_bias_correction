# this is datasettransfer_mega_analysis_script.py:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def load_cross_dataset_data(results_dir='../../results'):
    """Load specific method results including cross-dataset transfers"""
    results_dir = Path(results_dir)
    data = {}
    
    # Look for specific method directories
    for subdir in results_dir.iterdir():
        if subdir.is_dir() and 'specific' in subdir.name and 'TVD' in subdir.name:
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
    # print head of loaded data for verification
    print(data.keys())
    return data

def extract_cross_dataset_metrics(data, question_type, prompt_type, model_family, 
                                  target_dataset, source_dataset, model_name, batch_size):
    """Extract metrics for cross-dataset transfer
    
    Args:
        source_dataset: If None, extracts same-dataset results
        target_dataset: The dataset being tested on
    """
    # The JSON is organized by TARGET dataset
    key = ('specific', question_type, 'per', prompt_type, model_family, target_dataset)
    
    if key not in data:
        return None
    
    try:
        json_data = data[key]
        
        if question_type == 'yesno':
            dataset_key = 'PER_YESNO'
            # Check for cross-dataset domain key
            if source_dataset and source_dataset != target_dataset:
                domain_key = f"{target_dataset}-from{source_dataset}"
            else:
                domain_key = f"{target_dataset}-from{target_dataset}"
        elif question_type == 'nli':
            dataset_key = 'PER_NLI'
            # Check for cross-dataset domain key
            if source_dataset and source_dataset != target_dataset:
                domain_key = f"{target_dataset}-from{source_dataset}"
            else:
                domain_key = f"{target_dataset}-from{target_dataset}"
        else:  # mcq
            dataset_key = 'PER_MMLU'
            target_domain = target_dataset.split('-')[1] if '-' in target_dataset else target_dataset
            if source_dataset and source_dataset != target_dataset:
                source_domain = source_dataset.split('-')[1] if '-' in source_dataset else source_dataset
                domain_key = f"{target_domain}-from{source_domain}"
            else:
                domain_key = f"{target_domain}-from{target_domain}"
        
        if domain_key not in json_data[prompt_type][dataset_key]:
            return None
        
        model_data = json_data[prompt_type][dataset_key][domain_key][model_family]
        
        # Get metrics for specific model
        if model_name not in model_data:
            return None
        
        batch_size_str = str(batch_size)
        if batch_size_str not in model_data[model_name]:
            return None
        
        batch_data = model_data[model_name][batch_size_str]
        
        return {
            'median_acc': batch_data.get('median_acc', 0),
            'q25_acc': batch_data.get('q25_acc', 0),
            'q75_acc': batch_data.get('q75_acc', 0),
            'bias_median': batch_data.get('median_tvd', 0),
            'acc_mean': batch_data.get('mean_acc', 0),
            'best_run_acc': batch_data.get('best_run_acc', 0),
            'worst_run_acc': batch_data.get('worst_run_acc', 0)
        }
    
    except Exception as e:
        print(f"Error extracting cross-dataset metrics: {e}")
        return None

def get_dataset_pairs(question_type):
    """Get all dataset pairs for cross-dataset analysis"""
    if question_type == 'yesno':
        datasets = ['ARITH', 'BABI', 'COMPS', 'EWOK']
    elif question_type == 'nli':
        datasets = ['MNLI', 'SNLI']
    else:  # mcq
        datasets = ['MMLU-STEM', 'MMLU-HUMANITIES', 'MMLU-SOCIAL_SCI', 'MMLU-OTHERS']
    
    pairs = []
    # Generate all pairs (target, source) where target != source
    for target in datasets:
        for source in datasets:
            if target != source:
                pairs.append((target, source))
    
    return datasets, pairs

def get_models_from_family(model_family):
    """Get all models within a family"""
    model_configs = {
        "Falcon": ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"],
        "Gemma3": ["gemma-3-27b-pt", "gemma-3-27b-it", "gemma-3-12b-pt", "gemma-3-12b-it"],
        "Llama3": ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"]
    }
    
    return model_configs.get(model_family, [])

def create_cross_dataset_transfer_table():
    """Create comprehensive table showing cross-dataset transfer performance"""
    data = load_cross_dataset_data()
    if not data:
        print("No cross-dataset data found.")
    
    model_families = ['Falcon', 'Gemma3', 'Llama3']
    prompt_types = ['zeroshot', 'instronly', 'fewshot']
    
    results = []
    
    # Process Yes-No
    batch_size_yn = 500
    yn_datasets, yn_pairs = get_dataset_pairs('yesno')
    
    for target_dataset in yn_datasets:
        for prompt in prompt_types:
            for family in model_families:
                models = get_models_from_family(family)
                
                for model in models:
                    # Same-dataset results
                    metrics = extract_cross_dataset_metrics(
                        data, 'yesno', prompt, family, 
                        target_dataset, None, model, batch_size_yn
                    )
                    
                    if metrics:
                        results.append({
                            'Question_Type': 'YN',
                            'Target_Dataset': target_dataset,
                            'Source_Dataset': target_dataset,
                            'Transfer_Type': 'Same',
                            'Model_Family': family,
                            'Model': model,
                            'Prompt': prompt,
                            'Batch_Size': batch_size_yn,
                            **metrics
                        })
                    
                    # Cross-dataset results
                    for source_dataset in yn_datasets:
                        if source_dataset != target_dataset:
                            metrics = extract_cross_dataset_metrics(
                                data, 'yesno', prompt, family,
                                target_dataset, source_dataset, model, batch_size_yn
                            )
                            
                            if metrics:
                                results.append({
                                    'Question_Type': 'YN',
                                    'Target_Dataset': target_dataset,
                                    'Source_Dataset': source_dataset,
                                    'Transfer_Type': 'Cross',
                                    'Model_Family': family,
                                    'Model': model,
                                    'Prompt': prompt,
                                    'Batch_Size': batch_size_yn,
                                    **metrics
                                })

    # Process NLI
    batch_size_nli = 500
    nli_datasets, nli_pairs = get_dataset_pairs('nli')

    for target_dataset in nli_datasets:
        for prompt in ['instronly', 'fewshot']:
            for family in model_families:
                models = get_models_from_family(family)

                for model in models:
                    # Same-dataset results
                    metrics = extract_cross_dataset_metrics(
                        data, 'nli', prompt, family,
                        target_dataset, None, model, batch_size_nli
                    )

                    if metrics:
                        results.append({
                            'Question_Type': 'NLI',
                            'Target_Dataset': target_dataset,
                            'Source_Dataset': target_dataset,
                            'Transfer_Type': 'Same',
                            'Model_Family': family,
                            'Model': model,
                            'Prompt': prompt,
                            'Batch_Size': batch_size_nli,
                            **metrics
                        })

                    # Cross-dataset results
                    for source_dataset in nli_datasets:
                        if source_dataset != target_dataset:
                            metrics = extract_cross_dataset_metrics(
                                data, 'nli', prompt, family,
                                target_dataset, source_dataset, model, batch_size_nli
                            )

                            if metrics:
                                results.append({
                                    'Question_Type': 'NLI',
                                    'Target_Dataset': target_dataset,
                                    'Source_Dataset': source_dataset,
                                    'Transfer_Type': 'Cross',
                                    'Model_Family': family,
                                    'Model': model,
                                    'Prompt': prompt,
                                    'Batch_Size': batch_size_nli,
                                    **metrics
                                })
    
    # Process MCQ
    batch_size_mcq = 500
    mcq_datasets, mcq_pairs = get_dataset_pairs('mcq')
    
    for target_dataset in mcq_datasets:
        for prompt in prompt_types:
            for family in model_families:
                models = get_models_from_family(family)
                
                for model in models:
                    # Same-dataset results
                    metrics = extract_cross_dataset_metrics(
                        data, 'mcq', prompt, family,
                        target_dataset, None, model, batch_size_mcq
                    )
                    
                    if metrics:
                        results.append({
                            'Question_Type': 'MCQ',
                            'Target_Dataset': target_dataset,
                            'Source_Dataset': target_dataset,
                            'Transfer_Type': 'Same',
                            'Model_Family': family,
                            'Model': model,
                            'Prompt': prompt,
                            'Batch_Size': batch_size_mcq,
                            **metrics
                        })
                    
                    # Cross-dataset results
                    for source_dataset in mcq_datasets:
                        if source_dataset != target_dataset:
                            metrics = extract_cross_dataset_metrics(
                                data, 'mcq', prompt, family,
                                target_dataset, source_dataset, model, batch_size_mcq
                            )
                            
                            if metrics:
                                results.append({
                                    'Question_Type': 'MCQ',
                                    'Target_Dataset': target_dataset,
                                    'Source_Dataset': source_dataset,
                                    'Transfer_Type': 'Cross',
                                    'Model_Family': family,
                                    'Model': model,
                                    'Prompt': prompt,
                                    'Batch_Size': batch_size_mcq,
                                    **metrics
                                })
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        output_dir = Path('../../results/table_outputs/transfer_dataset')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_dir / 'cross_dataset_transfer_full.csv', index=False)
        print(f"Created full cross-dataset transfer table with {len(df)} rows")
        
        # Create summary aggregated by target dataset (average over models and prompts)
        summary = df.groupby([
            'Question_Type', 'Target_Dataset', 'Source_Dataset', 'Transfer_Type'
        ]).agg({
            'median_acc': 'mean',
            'q25_acc': 'mean',
            'q75_acc': 'mean',
            'bias_median': 'mean',
            'best_run_acc': 'mean',
            'worst_run_acc': 'mean'
        }).round(4).reset_index()
        
        summary.to_csv(output_dir / 'cross_dataset_transfer_summary.csv', index=False)
        print(f"Created summary table with {len(summary)} rows")
        
        return df, summary
    
    return pd.DataFrame(), pd.DataFrame()

def plot_cross_dataset_heatmap(df, question_type='YN', model_family='Falcon',
                               model_name='Falcon3-10B-Base', prompt_type='fewshot'):
    """Create heatmap showing cross-dataset transfer performance"""
    
    df_subset = df[
        (df['Question_Type'] == question_type) & 
        (df['Model_Family'] == model_family) &
        (df['Model'] == model_name) &
        (df['Prompt'] == prompt_type)
    ]
    
    if df_subset.empty:
        print(f"No data for {question_type}, {model_family}, {model_name}, {prompt_type}")
        return
    
    # Get unique datasets
    datasets = sorted(df_subset['Target_Dataset'].unique())
    
    # Create pivot table and store same-dataset baseline values
    pivot_data = []
    same_dataset_baselines = {}
    
    for target in datasets:
        row = []
        # Get same-dataset baseline for this target
        same_dataset_row = df_subset[
            (df_subset['Target_Dataset'] == target) & 
            (df_subset['Transfer_Type'] == 'Same')
        ]
        if not same_dataset_row.empty:
            same_dataset_baselines[target] = same_dataset_row['median_acc'].mean()
        else:
            same_dataset_baselines[target] = np.nan
        
        for source in datasets:
            if target == source:
                # Same-dataset
                if not same_dataset_row.empty:
                    row.append(same_dataset_row['median_acc'].mean())
                else:
                    row.append(np.nan)
            else:
                # Cross-dataset
                cross_dataset_row = df_subset[
                    (df_subset['Target_Dataset'] == target) & 
                    (df_subset['Source_Dataset'] == source)
                ]
                if not cross_dataset_row.empty:
                    row.append(cross_dataset_row['median_acc'].mean())
                else:
                    row.append(np.nan)
        pivot_data.append(row)
    
    pivot = pd.DataFrame(pivot_data, index=datasets, columns=datasets)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(pivot.values, cmap='Greens', aspect='auto',
                   vmin=np.nanmin(pivot.values), vmax=np.nanmax(pivot.values))
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Median Accuracy', fontsize=12)
    
    # Set ticks with short names
    def get_short_name(dataset):
        """Get short dataset name"""
        return dataset.replace('MMLU-', '')
    
    display_names = [get_short_name(d) for d in datasets]
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(display_names, fontsize=11)
    ax.set_yticklabels(display_names, fontsize=11)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                # Determine text color
                text_color = "white" if val > (np.nanmax(pivot.values) - np.nanmin(pivot.values)) / 2 + np.nanmin(pivot.values) else "black"
                
                if i == j:
                    # Diagonal (same-dataset): bold
                    ax.text(j, i, f'{val:.4f}',
                           ha="center", va="center",
                           color=text_color,
                           fontsize=10, fontweight='bold')
                else:
                    # Off-diagonal (cross-dataset): show accuracy and delta
                    baseline = same_dataset_baselines[datasets[i]]
                    if not np.isnan(baseline):
                        delta = val - baseline
                        delta_str = f"{delta:+.4f}"
                        
                        # Main accuracy value
                        ax.text(j, i - 0.15, f'{val:.4f}',
                               ha="center", va="center",
                               color=text_color,
                               fontsize=9)
                        
                        # Delta value (smaller, below)
                        delta_color = "green" if delta > 0 else "red" if delta < 0 else text_color
                        ax.text(j, i + 0.15, delta_str,
                               ha="center", va="center",
                               color=delta_color,
                               fontsize=7, fontweight='bold', style='italic')
                    else:
                        ax.text(j, i, f'{val:.4f}',
                               ha="center", va="center",
                               color=text_color,
                               fontsize=9)
    
    ax.set_xlabel('Source Dataset (Calibration)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Target Dataset (Test)', fontsize=13, fontweight='bold')
    
    # question_name = "Yes-No" if question_type == "YN" else "MCQ"
    if question_type == 'YN':
        question_name = "Yes-No"
    elif question_type == 'NLI':
        question_name = "NLI"
    else:
        question_name = "MCQ"
    model_short = model_name.replace('Falcon3-', '').replace('Llama-3.1-', '').replace('Qwen2.5-', '').replace('gemma-3-', '')
    ax.set_title(
        f'{question_name} Cross-Dataset Transfer: {model_short} ({prompt_type})\n'
        f'(Diagonal = Same-Dataset, Off-Diagonal = Cross-Dataset with Δ)',
        fontsize=14, fontweight='bold', pad=20
    )
    
    plt.tight_layout()
    output_dir = Path('../../results/plot_outputs/transfer_dataset')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / f'crossdataset_heatmap_{question_type}_{model_family}_{model_name}_{prompt_type}.png',
        dpi=300, bbox_inches='tight'
    )
    print(f"Saved heatmap for {question_type}, {model_family}, {model_name}, {prompt_type}")
    plt.close()


def plot_same_vs_cross_dataset_comparison(df_summary):
    """Compare same-dataset vs cross-dataset calibration performance"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, qtype in enumerate(['YN', 'NLI', 'MCQ']):
        df_subset = df_summary[df_summary['Question_Type'] == qtype]
        
        if df_subset.empty:
            continue
        
        ax = axes[idx]
        
        datasets = sorted(df_subset['Target_Dataset'].unique())
        
        for dataset in datasets:
            dataset_data = df_subset[df_subset['Target_Dataset'] == dataset]
            
            same_dataset = dataset_data[dataset_data['Transfer_Type'] == 'Same']['median_acc']
            cross_dataset = dataset_data[dataset_data['Transfer_Type'] == 'Cross']['median_acc']
            
            if not same_dataset.empty and not cross_dataset.empty:
                short_name = dataset.replace('MMLU-', '')
                ax.scatter([same_dataset.mean()], [cross_dataset.mean()], 
                          label=short_name, s=150, alpha=0.7)
        
        # Diagonal line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, label='x=y')
        
        ax.set_xlabel('Same-Dataset Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cross-Dataset Accuracy (avg)', fontsize=12, fontweight='bold')
        ax.set_title(f'{"Yes-No" if qtype == "YN" else "NLI" if qtype == "NLI" else "MCQ"} Questions',
                    fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Same-Dataset vs Cross-Dataset Calibration Performance',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_dir = Path('../../results/plot_outputs/transfer_dataset')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / 'same_vs_cross_dataset_comparison.png',
        dpi=300, bbox_inches='tight'
    )
    print("Saved same vs cross-dataset comparison")
    plt.close()


def plot_dataset_transfer_degradation(df_summary):
    """Plot degradation for each target dataset when using cross-dataset calibration"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, qtype in enumerate(['YN', 'NLI', 'MCQ']):
        df_subset = df_summary[df_summary['Question_Type'] == qtype]
        
        if df_subset.empty:
            continue
        
        ax = axes[idx]
        
        datasets = sorted(df_subset['Target_Dataset'].unique())
        degradations = []
        dataset_labels = []
        
        for dataset in datasets:
            dataset_data = df_subset[df_subset['Target_Dataset'] == dataset]
            same = dataset_data[dataset_data['Transfer_Type'] == 'Same']['median_acc'].mean()
            cross = dataset_data[dataset_data['Transfer_Type'] == 'Cross']['median_acc'].mean()
            
            if not np.isnan(same) and not np.isnan(cross):
                degradations.append((same - cross) * 100)
                dataset_labels.append(dataset.replace('MMLU-', ''))
        
        ax.bar(dataset_labels, degradations, alpha=0.7, color='coral')
        ax.set_xlabel('Target Dataset', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy Drop (% points)', fontsize=12, fontweight='bold')
        ax.set_title(f'{"Yes-No" if qtype == "YN" else "NLI" if qtype == "NLI" else "MCQ"}: Cross-Dataset Degradation',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    plt.suptitle('Cross-Dataset Transfer Degradation by Target Dataset',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_dir = Path('../../results/plot_outputs/transfer_dataset')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / 'dataset_transfer_degradation.png',
        dpi=300, bbox_inches='tight'
    )
    print("Saved dataset transfer degradation plot")
    plt.close()


if __name__ == "__main__":
    print("Creating cross-dataset transfer tables...")
    df_full, df_summary = create_cross_dataset_transfer_table()
    
    if not df_full.empty:
        # Get available combinations
        available_combinations = df_full.groupby([
            'Question_Type', 'Model_Family', 'Model', 'Prompt'
        ]).size().reset_index()[['Question_Type', 'Model_Family', 'Model', 'Prompt']]
        
        print("\nGenerating heatmaps for available data combinations...")
        # Sample: Generate for one model per family to avoid too many plots
        sample_models = {
            'Falcon': 'Falcon3-10B-Base',
            'Gemma3': 'gemma-3-27b-pt',
            'Llama3': 'Llama-3.1-70B'
        }

        for qtype in ['YN', 'NLI', 'MCQ']:
            for family, model in sample_models.items():
                for prompt in ['zeroshot', 'instronly', 'fewshot']:
                    # Check if this combination exists
                    exists = not df_full[
                        (df_full['Question_Type'] == qtype) &
                        (df_full['Model_Family'] == family) &
                        (df_full['Model'] == model) &
                        (df_full['Prompt'] == prompt)
                    ].empty
                    
                    if exists:
                        print(f"  Creating heatmap: {qtype}, {family}, {model}, {prompt}")
                        plot_cross_dataset_heatmap(df_full, qtype, family, model, prompt)
        
        print("\nCreating comparison plots...")
        plot_same_vs_cross_dataset_comparison(df_summary)
        plot_dataset_transfer_degradation(df_summary)
        
        print("\nDone! Check table_outputs/transfer_dataset/ and plot_outputs/transfer_dataset/")
    else:
        print("No cross-dataset data found. Make sure you have processed cross-dataset results.")