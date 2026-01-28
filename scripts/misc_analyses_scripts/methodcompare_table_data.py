import json
import os
from pathlib import Path

# Model configuration
MODELS = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct"]
MODEL_FAMILY = "Llama3"
PROMPT_LEVEL = "fewshot"

# Dataset configurations with their properties
DATASETS = {
    'ARITH': {'qtype': 'yesno', 'domain': 'arith', 'display': 'ARITH'},
    'BABI': {'qtype': 'yesno', 'domain': 'babi', 'display': 'BABI'},
    'COMPS': {'qtype': 'yesno', 'domain': 'comps', 'display': 'COMPS'},
    'EWOK': {'qtype': 'yesno', 'domain': 'all_domains', 'display': 'EWOK'},
    'SNLI': {'qtype': 'nli', 'domain': 'snli', 'display': 'SNLI'},
    'MNLI': {'qtype': 'nli', 'domain': 'mnli', 'display': 'MNLI'},
    'MMLU-HUMANITIES': {'qtype': 'mcq', 'domain': 'HUMANITIES', 'display': 'HUMANITIES'},
    'MMLU-OTHERS': {'qtype': 'mcq', 'domain': 'OTHERS', 'display': 'OTHERS'},
    'MMLU-SOCIAL_SCI': {'qtype': 'mcq', 'domain': 'SOCIAL_SCI', 'display': 'SOCIAL_SCI'},
    'MMLU-STEM': {'qtype': 'mcq', 'domain': 'STEM', 'display': 'STEM'},
}

def load_json_file(filepath):
    """Load JSON data from file."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_specific_stats(results_dir, dataset, model):
    """Extract statistics from specific method."""
    qtype = DATASETS[dataset]['qtype']
    qtype_folder = f"specific_{qtype}_per_median_TVD"
    filename = f"{PROMPT_LEVEL}_{MODEL_FAMILY}_{dataset}.json"
    filepath = os.path.join(results_dir, qtype_folder, filename)
    
    data = load_json_file(filepath)
    if data is None:
        return None
    
    try:
        # Navigate: promptlevel -> PER_QTYPE -> DATASET-fromDATASET -> modelfamily -> model -> "100"
        qtype_map = {'yesno': 'YESNO', 'nli': 'NLI', 'mcq': 'MMLU'}
        per_qtype = f"PER_{qtype_map[qtype]}"
        if qtype != 'mcq':
            same_condition = f"{dataset}-from{dataset}"
        else:
            # remove the MMLU- prefix for mcq datasets
            dataset_short = dataset.replace("MMLU-", "")
            same_condition = f"{dataset_short}-from{dataset_short}"
        
        stats = data[PROMPT_LEVEL][per_qtype][same_condition][MODEL_FAMILY][model]["100"]
        
        return {
            'raw_acc': stats['raw_acc'],
            'raw_tvd': stats['raw_tvd'],
            'median_acc': stats['median_acc'],
            'median_tvd': stats['median_tvd']
        }
    except KeyError as e:
        print(f"Error extracting specific stats for {dataset}/{model}: {e}")
        return None

def extract_contextcalib_stats(results_dir, dataset, model):
    """Extract statistics from contextcalib method."""
    qtype = DATASETS[dataset]['qtype']
    domain = DATASETS[dataset]['domain']
    qtype_folder = f"contextcalib_{qtype}_per_TVD"
    filename = f"{PROMPT_LEVEL}_{MODEL_FAMILY}_{dataset}.json"
    filepath = os.path.join(results_dir, qtype_folder, filename)
    
    data = load_json_file(filepath)
    if data is None:
        return None
    
    try:
        # Navigate: promptlevel -> DATASET -> domain -> modelfamily -> model
        stats = data[PROMPT_LEVEL][dataset][domain][MODEL_FAMILY][model]
        
        return {
            'raw_acc': stats['raw_acc'],
            'raw_tvd': stats['raw_tvd'],
            'corrected_acc': stats['corrected_acc'],
            'corrected_tvd': stats['corrected_tvd']
        }
    except KeyError as e:
        print(f"Error extracting contextcalib stats for {dataset}/{model}: {e}")
        return None

def extract_batchcalib_stats(results_dir, dataset, model):
    """Extract statistics from batchcalib method."""
    qtype = DATASETS[dataset]['qtype']
    domain = DATASETS[dataset]['domain']
    qtype_folder = f"batchcalib_{qtype}_per_TVD"
    filename = f"{PROMPT_LEVEL}_{MODEL_FAMILY}_{dataset}.json"
    filepath = os.path.join(results_dir, qtype_folder, filename)
    
    data = load_json_file(filepath)
    if data is None:
        return None
    
    try:
        # Navigate: promptlevel -> DATASET -> domain -> modelfamily -> model -> "100"
        stats = data[PROMPT_LEVEL][dataset][domain][MODEL_FAMILY][model]["100"]
        
        return {
            'raw_acc': stats['raw_acc'],
            'raw_tvd': stats['raw_tvd'],
            'corrected_acc': stats['corrected_acc'],
            'corrected_tvd': stats['corrected_tvd']
        }
    except KeyError as e:
        print(f"Error extracting batchcalib stats for {dataset}/{model}: {e}")
        return None

def format_number(value, is_percentage=False, decimal_places=2):
    """Format number for display."""
    if value is None:
        return "N/A"
    
    if is_percentage:
        # Convert to percentage if not already
        if value <= 1.0:
            value *= 100
        return f"{value:.{decimal_places}f}"
    else:
        return f"{value:.{decimal_places}f}"

def format_delta(value, is_percentage=False, decimal_places=2):
    """Format delta value with sign."""
    if value is None:
        return "N/A"
    
    sign = "+" if value > 0 else ""
    
    if is_percentage:
        # Convert to percentage if needed
        if abs(value) <= 1.0:
            value *= 100
        return f"{sign}{value:.{decimal_places}f}"
    else:
        return f"{sign}{value:.{decimal_places}f}"

def create_table_data(results_dir, output_file='../../results/table_outputs/methodcompare_table_data.txt'):
    """Create table data for all datasets and models."""
    
    output_lines = []
    output_lines.append("=" * 120)
    output_lines.append("TABLE DATA FOR LATEX")
    output_lines.append("=" * 120)
    output_lines.append("")
    output_lines.append("Format: Dataset | Model | Baseline(Acc|Bias) | CC Δ(Acc|Bias) | BC Δ(Acc|Bias) | Ours Δ(Acc|Bias)")
    output_lines.append("-" * 120)
    output_lines.append("")
    
    # Process each dataset
    for dataset_name, dataset_info in DATASETS.items():
        display_name = dataset_info['display']
        output_lines.append(f"\n{display_name}")
        output_lines.append("-" * 60)
        
        # Process each model
        for model in MODELS:
            model_short = "8B" if model == "Llama-3.1-8B" else "8B-Instruct"
            
            # Get baseline from any method (they should all have same raw values)
            # We'll use specific method for baseline
            specific_stats = extract_specific_stats(results_dir, dataset_name, model)
            
            if specific_stats is None:
                output_lines.append(f"{model_short}: Data not found")
                continue
            
            baseline_acc = specific_stats['raw_acc']
            baseline_tvd = specific_stats['raw_tvd']
            
            # Get contextcalib stats
            cc_stats = extract_contextcalib_stats(results_dir, dataset_name, model)
            if cc_stats:
                cc_delta_acc = cc_stats['corrected_acc'] - baseline_acc
                cc_delta_tvd = cc_stats['corrected_tvd'] - baseline_tvd
            else:
                cc_delta_acc = None
                cc_delta_tvd = None
            
            # Get batchcalib stats
            bc_stats = extract_batchcalib_stats(results_dir, dataset_name, model)
            if bc_stats:
                bc_delta_acc = bc_stats['corrected_acc'] - baseline_acc
                bc_delta_tvd = bc_stats['corrected_tvd'] - baseline_tvd
            else:
                bc_delta_acc = None
                bc_delta_tvd = None
            
            # Get specific (Ours) stats - already have it
            ours_delta_acc = specific_stats['median_acc'] - baseline_acc
            ours_delta_tvd = specific_stats['median_tvd'] - baseline_tvd
            
            # Format output line
            line = f"{model_short:12} | "
            line += f"{format_number(baseline_acc, True, 1):>5} | {format_number(baseline_tvd, False, 3):>5} | "
            line += f"{format_delta(cc_delta_acc, True, 1):>6} | {format_delta(cc_delta_tvd, False, 3):>6} | "
            line += f"{format_delta(bc_delta_acc, True, 1):>6} | {format_delta(bc_delta_tvd, False, 3):>6} | "
            line += f"{format_delta(ours_delta_acc, True, 1):>6} | {format_delta(ours_delta_tvd, False, 3):>6}"
            
            output_lines.append(line)
    
    output_lines.append("\n" + "=" * 120)
    output_lines.append("\nFORMATTED FOR LATEX TABLE (copy values in order)")
    output_lines.append("=" * 120)
    output_lines.append("")
    
    # Create latex-friendly format
    for dataset_name, dataset_info in DATASETS.items():
        display_name = dataset_info['display']
        
        for model in MODELS:
            model_short = "8B" if model == "Llama-3.1-8B" else "8B-Instruct"
            
            specific_stats = extract_specific_stats(results_dir, dataset_name, model)
            if specific_stats is None:
                continue
            
            baseline_acc = specific_stats['raw_acc']
            baseline_tvd = specific_stats['raw_tvd']
            
            cc_stats = extract_contextcalib_stats(results_dir, dataset_name, model)
            if cc_stats:
                cc_delta_acc = cc_stats['corrected_acc'] - baseline_acc
                cc_delta_tvd = cc_stats['corrected_tvd'] - baseline_tvd
            else:
                cc_delta_acc = None
                cc_delta_tvd = None
            
            bc_stats = extract_batchcalib_stats(results_dir, dataset_name, model)
            if bc_stats:
                bc_delta_acc = bc_stats['corrected_acc'] - baseline_acc
                bc_delta_tvd = bc_stats['corrected_tvd'] - baseline_tvd
            else:
                bc_delta_acc = None
                bc_delta_tvd = None
            
            ours_delta_acc = specific_stats['median_acc'] - baseline_acc
            ours_delta_tvd = specific_stats['median_tvd'] - baseline_tvd
            
            # LaTeX row format
            latex_line = f"{display_name:12} {model_short:12} & "
            latex_line += f"{format_number(baseline_acc, True, 1)} & {format_number(baseline_tvd, False, 3)} & "
            latex_line += f"{format_delta(cc_delta_acc, True, 1)} & {format_delta(cc_delta_tvd, False, 3)} & "
            latex_line += f"{format_delta(bc_delta_acc, True, 1)} & {format_delta(bc_delta_tvd, False, 3)} & "
            latex_line += f"{format_delta(ours_delta_acc, True, 1)} & {format_delta(ours_delta_tvd, False, 3)} \\\\"
            
            output_lines.append(latex_line)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\nTable data written to {output_file}")
    print("\n".join(output_lines))

if __name__ == "__main__":
    results_dir = Path("../../results")
    create_table_data(results_dir)