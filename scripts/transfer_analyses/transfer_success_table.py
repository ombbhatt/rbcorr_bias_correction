import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

# Define the base directory
# BASE_DIR = Path.home() / "scratch/yes-bias-in-llms/results"
BASE_DIR = Path("../../results")

# Define all configurations
YESNO_DATASETS = ["ARITH", "BABI", "COMPS", "EWOK"]
NLI_DATASETS = ["MNLI", "SNLI"]
MCQ_DATASETS = ["MMLU-HUMANITIES", "MMLU-OTHERS", "MMLU-SOCIAL_SCI", "MMLU-STEM"]

FALCON_MODELS = ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"]
GEMMA3_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it", "gemma-3-12b-pt", "gemma-3-12b-it"]
LLAMA3_MODELS = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"]

MODEL_FAMILIES = {
    "Falcon": FALCON_MODELS,
    "Gemma3": GEMMA3_MODELS,
    "Llama3": LLAMA3_MODELS
}

PROMPT_LEVELS = ["zeroshot", "instronly", "fewshot"]

QTYPE_MAPPING = {
    "yesno": ("YESNO", YESNO_DATASETS, "specific_yesno_per_median_TVD"),
    "nli": ("NLI", NLI_DATASETS, "specific_nli_per_median_TVD"),
    "mcq": ("MMLU", MCQ_DATASETS, "specific_mcq_per_median_TVD")
}

def load_json_file(filepath):
    """Load a JSON file and return its contents."""
    if not filepath.exists():
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_metrics(data, prompt_level, qtype_key, config_key, model_family, model_name, calib_size="500"):
    """
    Extract raw and median metrics from nested JSON structure.
    
    Returns: (raw_acc, raw_tvd, median_acc, median_tvd) or None if not found
    """
    try:
        nested = data[prompt_level][f"PER_{qtype_key}"][config_key][model_family][model_name][calib_size]
        return (
            nested["raw_acc"],
            nested["raw_tvd"],
            nested["median_acc"],
            nested["median_tvd"]
        )
    except:
        # print exception for debugging
        import traceback
        traceback.print_exc()
        return None

def get_same_condition_key(dataset):
    """Get the same-condition config key for a dataset."""
    return f"{dataset}-from{dataset}"

def get_cross_dataset_pairs():
    """
    Generate all valid cross-dataset transfer pairs.
    Returns list of tuples: (qtype, source_dataset, target_dataset, model_family, model_name, prompt_level)
    """
    pairs = []
    
    for qtype, (qtype_key, datasets, folder) in QTYPE_MAPPING.items():
        for source_dataset in datasets:
            for target_dataset in datasets:
                if source_dataset == target_dataset:
                    continue  # Skip same-condition
                
                for family_name, models in MODEL_FAMILIES.items():
                    for model in models:
                        for prompt in PROMPT_LEVELS:
                            # Skip zeroshot for NLI datasets
                            if qtype == "nli" and prompt == "zeroshot":
                                continue
                            
                            pairs.append((qtype, source_dataset, target_dataset, family_name, model, prompt))
    
    return pairs

def get_cross_model_pairs():
    """
    Generate all valid cross-model transfer pairs.
    Returns list of tuples: (qtype, dataset, source_model, target_model, model_family, prompt_level)
    """
    pairs = []
    
    for qtype, (qtype_key, datasets, folder) in QTYPE_MAPPING.items():
        for dataset in datasets:
            for family_name, models in MODEL_FAMILIES.items():
                for source_model in models:
                    for target_model in models:
                        if source_model == target_model:
                            continue  # Skip same-condition
                        
                        for prompt in PROMPT_LEVELS:
                            # Skip zeroshot for NLI datasets
                            if qtype == "nli" and prompt == "zeroshot":
                                continue
                            
                            pairs.append((qtype, dataset, source_model, target_model, family_name, prompt))
    
    return pairs

def get_cross_prompt_pairs():
    """
    Generate all valid cross-prompt transfer pairs.
    Returns list of tuples: (qtype, dataset, model_family, model_name, source_prompt, target_prompt)
    """
    pairs = []
    
    for qtype, (qtype_key, datasets, folder) in QTYPE_MAPPING.items():
        for dataset in datasets:
            for family_name, models in MODEL_FAMILIES.items():
                for model in models:
                    for source_prompt in PROMPT_LEVELS:
                        for target_prompt in PROMPT_LEVELS:
                            if source_prompt == target_prompt:
                                continue  # Skip same-condition
                            
                            # Skip if either source or target is zeroshot for NLI
                            if qtype == "nli" and (source_prompt == "zeroshot" or target_prompt == "zeroshot"):
                                continue
                            
                            pairs.append((qtype, dataset, family_name, model, source_prompt, target_prompt))
    
    return pairs

def analyze_cross_dataset():
    """Analyze cross-dataset transfer efficacy."""
    pairs = get_cross_dataset_pairs()
    total_count = len(pairs)
    successful_count = 0
    successful_acc_changes = []
    successful_tvd_changes = []
    all_acc_changes = []
    all_tvd_changes = []
    failed_bias_only = 0
    failed_acc_only = 0
    failed_both = 0

    missing_data_count = 0
    missing_pairs = []
    
    for qtype, source_dataset, target_dataset, family_name, model, prompt in pairs:
        qtype_key, datasets, folder_name = QTYPE_MAPPING[qtype]
        folder = BASE_DIR / folder_name
        
        # Load target dataset file
        filename = f"{prompt}_{family_name}_{target_dataset}.json"
        filepath = folder / filename
        data = load_json_file(filepath)
        
        if data is None:
            missing_data_count += 1
            missing_pairs.append(f"Missing data: {filepath} | {qtype} | {source_dataset}->{target_dataset} | {family_name}/{model} | {prompt}")
            continue
        
        # Get transfer results
        if "MMLU-" in source_dataset or "MMLU-" in target_dataset:
            source_dataset1 = source_dataset.replace("MMLU-", "")
            target_dataset1 = target_dataset.replace("MMLU-", "")
            transfer_config_key = f"{target_dataset1}-from{source_dataset1}"
        else:
            transfer_config_key = f"{target_dataset}-from{source_dataset}"
        transfer_metrics = extract_metrics(data, prompt, qtype_key, transfer_config_key, family_name, model)
        
        if transfer_metrics is None:
            missing_data_count += 1
            missing_pairs.append(f"Missing transfer metrics: {transfer_config_key} | {qtype} | {source_dataset}->{target_dataset} | {family_name}/{model} | {prompt}")
            continue
        
        transfer_raw_acc, transfer_raw_tvd, transfer_median_acc, transfer_median_tvd = transfer_metrics
        transfer_acc_gain = transfer_median_acc - transfer_raw_acc
        transfer_bias_reduction = transfer_raw_tvd - transfer_median_tvd

        all_acc_changes.append(transfer_acc_gain)
        all_tvd_changes.append(transfer_bias_reduction)
        
        # Get same-condition results for source dataset
        source_filename = f"{prompt}_{family_name}_{source_dataset}.json"
        source_filepath = folder / source_filename
        source_data = load_json_file(source_filepath)

        # Get same-condition metrics for target dataset
        target_filename = f"{prompt}_{family_name}_{target_dataset}.json"
        target_filepath = folder / target_filename
        target_data = load_json_file(target_filepath)
        
        if source_data is None:
            raise ValueError(f"Source data file not found: {source_filepath}")
        if target_data is None:
            raise ValueError(f"Target data file not found: {target_filepath}")
        
        # same_condition_key = get_same_condition_key(source_dataset)
        # if "MMLU-" in source_dataset:
        #     source_dataset1 = source_dataset.replace("MMLU-", "")
        #     same_condition_key = get_same_condition_key(source_dataset1)
        # same_condition_metrics = extract_metrics(source_data, prompt, qtype_key, same_condition_key, family_name, model)

        same_condition_key = get_same_condition_key(target_dataset)
        if "MMLU-" in target_dataset:
            target_dataset1 = target_dataset.replace("MMLU-", "")
            same_condition_key = get_same_condition_key(target_dataset1)
        else:
            same_condition_key = get_same_condition_key(target_dataset)
        same_condition_metrics = extract_metrics(target_data, prompt, qtype_key, same_condition_key, family_name, model)
        
        if same_condition_metrics is None:
            missing_data_count += 1
            missing_pairs.append(f"Missing file: {filepath} | {qtype} | {source_dataset}->{target_dataset} | {family_name}/{model} | {prompt}")
            continue
        
        sc_raw_acc, sc_raw_tvd, sc_median_acc, sc_median_tvd = same_condition_metrics
        sc_acc_gain = sc_median_acc - sc_raw_acc
        sc_bias_reduction = sc_raw_tvd - sc_median_tvd
        
        # Check success criteria
        bias_preserved = transfer_bias_reduction >= 0.8 * sc_bias_reduction
        acc_preserved = transfer_acc_gain >= 0.8 * sc_acc_gain
        
        if bias_preserved and acc_preserved:
            successful_count += 1
            successful_acc_changes.append(transfer_acc_gain)
            successful_tvd_changes.append(transfer_bias_reduction)
        else:
            # Track failure reasons
            if not bias_preserved and not acc_preserved:
                failed_both += 1
            elif not bias_preserved:
                failed_bias_only += 1
            else:  # not acc_preserved
                failed_acc_only += 1
    
    avg_acc_succ = np.mean(successful_acc_changes) if successful_acc_changes else 0.0
    avg_tvd_succ = np.mean(successful_tvd_changes) if successful_tvd_changes else 0.0
    avg_acc_all = np.mean(all_acc_changes) if all_acc_changes else 0.0
    avg_tvd_all = np.mean(all_tvd_changes) if all_tvd_changes else 0.0
    
    return {
        "total_pairs": total_count,
        "successful_pairs": successful_count,
        "failed_bias_only": failed_bias_only,
        "failed_acc_only": failed_acc_only,
        "failed_both": failed_both,
        "missing_data": missing_data_count,
        "missing_pairs_list": missing_pairs,
        "avg_succ_acc_change": avg_acc_succ,
        "avg_succ_tvd_change": avg_tvd_succ,
        "avg_all_acc_change": avg_acc_all,
        "avg_all_tvd_change": avg_tvd_all
    }

def analyze_cross_model():
    """Analyze cross-model transfer efficacy."""
    pairs = get_cross_model_pairs()
    total_count = len(pairs)
    successful_count = 0
    successful_acc_changes = []
    successful_tvd_changes = []
    all_acc_changes = []
    all_tvd_changes = []
    failed_bias_only = 0
    failed_acc_only = 0
    failed_both = 0

    missing_data_count = 0
    missing_pairs = []
    
    for qtype, dataset, source_model, target_model, family_name, prompt in pairs:
        qtype_key, datasets, folder_name = QTYPE_MAPPING[qtype]
        folder = BASE_DIR / folder_name
        
        # Load dataset file
        filename = f"{prompt}_{family_name}_{dataset}.json"
        filepath = folder / filename
        data = load_json_file(filepath)
        
        if data is None:
            missing_data_count += 1
            missing_pairs.append(f"Missing file: {filepath} | {qtype}/{dataset} | {source_model}->{target_model} | {family_name} | {prompt}")
            continue
        
        # Get transfer results
        if qtype == "mcq":
            dataset1 = dataset.replace("MMLU-", "")
            transfer_config_key = f"{dataset1}-from{dataset1}_{target_model}_from{source_model}"
        else:
            transfer_config_key = f"{dataset}-from{dataset}_{target_model}_from{source_model}"
        transfer_metrics = extract_metrics(data, prompt, qtype_key, transfer_config_key, family_name, target_model)
        
        if transfer_metrics is None:
            missing_data_count += 1
            missing_pairs.append(f"Missing transfer metrics: {transfer_config_key} | {qtype}/{dataset} | {source_model}->{target_model} | {family_name} | {prompt}")
            continue
        
        transfer_raw_acc, transfer_raw_tvd, transfer_median_acc, transfer_median_tvd = transfer_metrics
        transfer_acc_gain = transfer_median_acc - transfer_raw_acc
        transfer_bias_reduction = transfer_raw_tvd - transfer_median_tvd

        all_acc_changes.append(transfer_acc_gain)
        all_tvd_changes.append(transfer_bias_reduction)

        # Get same-condition results for source model
        if qtype == "mcq":
            dataset1 = dataset.replace("MMLU-", "")
            same_condition_key = get_same_condition_key(dataset1)
        else:
            same_condition_key = get_same_condition_key(dataset)
        # same_condition_metrics = extract_metrics(data, prompt, qtype_key, same_condition_key, family_name, source_model)

        # Get same-condition results for target model
        same_condition_metrics = extract_metrics(data, prompt, qtype_key, same_condition_key, family_name, target_model)
        
        if same_condition_metrics is None:
            raise ValueError(f"Same-condition data not found for {dataset}, {target_model}")
        
        sc_raw_acc, sc_raw_tvd, sc_median_acc, sc_median_tvd = same_condition_metrics
        sc_acc_gain = sc_median_acc - sc_raw_acc
        sc_bias_reduction = sc_raw_tvd - sc_median_tvd
        
        # Check success criteria
        bias_preserved = transfer_bias_reduction >= 0.8 * sc_bias_reduction
        acc_preserved = transfer_acc_gain >= 0.8 * sc_acc_gain
        
        if bias_preserved and acc_preserved:
            successful_count += 1
            successful_acc_changes.append(transfer_acc_gain)
            successful_tvd_changes.append(transfer_bias_reduction)
        else:
            # Track failure reasons
            if not bias_preserved and not acc_preserved:
                failed_both += 1
            elif not bias_preserved:
                failed_bias_only += 1
            else:  # not acc_preserved
                failed_acc_only += 1

    avg_acc_succ = np.mean(successful_acc_changes) if successful_acc_changes else 0.0
    avg_tvd_succ = np.mean(successful_tvd_changes) if successful_tvd_changes else 0.0
    avg_acc_all = np.mean(all_acc_changes) if all_acc_changes else 0.0
    avg_tvd_all = np.mean(all_tvd_changes) if all_tvd_changes else 0.0

    return {
        "total_pairs": total_count,
        "successful_pairs": successful_count,
        "failed_bias_only": failed_bias_only,
        "failed_acc_only": failed_acc_only,
        "failed_both": failed_both,
        "missing_data": missing_data_count,
        "missing_pairs_list": missing_pairs,
        "avg_succ_acc_change": avg_acc_succ,
        "avg_succ_tvd_change": avg_tvd_succ,
        "avg_all_acc_change": avg_acc_all,
        "avg_all_tvd_change": avg_tvd_all
    }

def analyze_cross_prompt():
    """Analyze cross-prompt transfer efficacy."""
    pairs = get_cross_prompt_pairs()
    total_count = len(pairs)
    successful_count = 0
    successful_acc_changes = []
    successful_tvd_changes = []
    all_acc_changes = []
    all_tvd_changes = []
    failed_bias_only = 0
    failed_acc_only = 0
    failed_both = 0

    missing_data_count = 0
    missing_pairs = []

    for qtype, dataset, family_name, model, source_prompt, target_prompt in pairs:
        qtype_key, datasets, folder_name = QTYPE_MAPPING[qtype]
        folder = BASE_DIR / folder_name
        
        # Load target prompt file
        filename = f"{target_prompt}_{family_name}_{dataset}.json"
        filepath = folder / filename
        data = load_json_file(filepath)
        
        if data is None:
            missing_data_count += 1
            missing_pairs.append(f"Missing file: {filepath} | {qtype} | {source_prompt}->{target_prompt} | {family_name}/{model} | {dataset}")
            continue
        
        # Get transfer results
        if qtype == "mcq":
            dataset1 = dataset.replace("MMLU-", "")
            transfer_config_key = f"{dataset1}-from{dataset1}_{target_prompt}_from{source_prompt}"
        else:
            transfer_config_key = f"{dataset}-from{dataset}_{target_prompt}_from{source_prompt}"
        transfer_metrics = extract_metrics(data, target_prompt, qtype_key, transfer_config_key, family_name, model)
        
        if transfer_metrics is None:
            missing_data_count += 1
            missing_pairs.append(f"Missing transfer metrics: {transfer_config_key} | {qtype} | {source_prompt}->{target_prompt} | {family_name}/{model} | {dataset}")
            continue
        
        transfer_raw_acc, transfer_raw_tvd, transfer_median_acc, transfer_median_tvd = transfer_metrics
        transfer_acc_gain = transfer_median_acc - transfer_raw_acc
        transfer_bias_reduction = transfer_raw_tvd - transfer_median_tvd

        all_acc_changes.append(transfer_acc_gain)
        all_tvd_changes.append(transfer_bias_reduction)
        
        # Get same-condition results for source prompt
        source_filename = f"{source_prompt}_{family_name}_{dataset}.json"
        source_filepath = folder / source_filename
        source_data = load_json_file(source_filepath)

        # Get same-condition metrics for target prompt
        target_filename = f"{target_prompt}_{family_name}_{dataset}.json"
        target_filepath = folder / target_filename
        target_data = load_json_file(target_filepath)
        
        if source_data is None:
            raise ValueError(f"Source data file not found: {source_filepath}")
        if target_data is None:
            raise ValueError(f"Target data file not found: {target_filepath}")
        
        same_condition_key = get_same_condition_key(dataset)
        if qtype == "mcq":
            dataset1 = dataset.replace("MMLU-", "")
            same_condition_key = get_same_condition_key(dataset1)
        same_condition_metrics = extract_metrics(target_data, target_prompt, qtype_key, same_condition_key, family_name, model)

        # same_condition_key = get_same_condition_key(dataset)
        # same_condition_metrics = extract_metrics(source_data, target_prompt, qtype_key, same_condition_key, family_name, model)
        
        if same_condition_metrics is None:
            missing_data_count += 1
            missing_pairs.append(f"Missing file: {filepath} | {qtype} | {source_prompt}->{target_prompt} | {family_name}/{model} | {dataset}")
            continue
        
        sc_raw_acc, sc_raw_tvd, sc_median_acc, sc_median_tvd = same_condition_metrics
        sc_acc_gain = sc_median_acc - sc_raw_acc
        sc_bias_reduction = sc_raw_tvd - sc_median_tvd
        
        # Check success criteria
        bias_preserved = transfer_bias_reduction >= 0.8 * sc_bias_reduction
        acc_preserved = transfer_acc_gain >= 0.8 * sc_acc_gain

        if bias_preserved and acc_preserved:
            successful_count += 1
            successful_acc_changes.append(transfer_acc_gain)
            successful_tvd_changes.append(transfer_bias_reduction)
        else:
            # Track failure reasons
            if not bias_preserved and not acc_preserved:
                failed_both += 1
            elif not bias_preserved:
                failed_bias_only += 1
            else:  # not acc_preserved
                failed_acc_only += 1
    
    avg_acc_succ = np.mean(successful_acc_changes) if successful_acc_changes else 0.0
    avg_tvd_succ = np.mean(successful_tvd_changes) if successful_tvd_changes else 0.0
    avg_acc_all = np.mean(all_acc_changes) if all_acc_changes else 0.0
    avg_tvd_all = np.mean(all_tvd_changes) if all_tvd_changes else 0.0

    return {
        "total_pairs": total_count,
        "successful_pairs": successful_count,
        "failed_bias_only": failed_bias_only,
        "failed_acc_only": failed_acc_only,
        "failed_both": failed_both,
        "missing_data": missing_data_count,
        "missing_pairs_list": missing_pairs,
        "avg_succ_acc_change": avg_acc_succ,
        "avg_succ_tvd_change": avg_tvd_succ,
        "avg_all_acc_change": avg_acc_all,
        "avg_all_tvd_change": avg_tvd_all
    }

def main():

    print("Analyzing Cross-Dataset Transfer Efficacy...")
    cross_dataset_results = analyze_cross_dataset()
    
    print("Analyzing Cross-Model Transfer Efficacy...")
    cross_model_results = analyze_cross_model()
    
    print("Analyzing Cross-Prompt Transfer Efficacy...")
    cross_prompt_results = analyze_cross_prompt()

    # Save all pairs to text files for verification
    print("\nSaving all transfer pairs to text files...")
    
    # Cross-Dataset pairs
    cd_pairs = get_cross_dataset_pairs()
    with open(Path.home() / "scratch/yes-bias-in-llms" / "cross_dataset_pairs.txt", 'w') as f:
        f.write(f"Total Cross-Dataset Pairs: {len(cd_pairs)}\n")
        f.write("="*80 + "\n\n")
        for qtype, src_ds, tgt_ds, family, model, prompt in cd_pairs:
            f.write(f"{qtype} | {src_ds} -> {tgt_ds} | {family}/{model} | {prompt}\n")
    
    # Cross-Model pairs
    cm_pairs = get_cross_model_pairs()
    with open(Path.home() / "scratch/yes-bias-in-llms" / "cross_model_pairs.txt", 'w') as f:
        f.write(f"Total Cross-Model Pairs: {len(cm_pairs)}\n")
        f.write("="*80 + "\n\n")
        for qtype, ds, src_mdl, tgt_mdl, family, prompt in cm_pairs:
            f.write(f"{qtype}/{ds} | {src_mdl} -> {tgt_mdl} | {family} | {prompt}\n")
    
    # Cross-Prompt pairs
    cp_pairs = get_cross_prompt_pairs()
    with open(Path.home() / "scratch/yes-bias-in-llms" / "cross_prompt_pairs.txt", 'w') as f:
        f.write(f"Total Cross-Prompt Pairs: {len(cp_pairs)}\n")
        f.write("="*80 + "\n\n")
        for qtype, ds, family, model, src_pr, tgt_pr in cp_pairs:
            f.write(f"{qtype}/{ds} | {src_pr} -> {tgt_pr} | {family}/{model}\n")
    
    print("Pair lists saved to cross_dataset_pairs.txt, cross_model_pairs.txt, cross_prompt_pairs.txt")

    # Write missing pairs to files
    with open(Path.home() / "scratch/yes-bias-in-llms" / "missing_cross_dataset_pairs.txt", 'w') as f:
        f.write(f"Missing Cross-Dataset Pairs: {cross_dataset_results['missing_data']}\n")
        f.write("="*80 + "\n\n")
        for pair_info in cross_dataset_results['missing_pairs_list']:
            f.write(f"{pair_info}\n")

    with open(Path.home() / "scratch/yes-bias-in-llms" / "missing_cross_model_pairs.txt", 'w') as f:
        f.write(f"Missing Cross-Model Pairs: {cross_model_results['missing_data']}\n")
        f.write("="*80 + "\n\n")
        for pair_info in cross_model_results['missing_pairs_list']:
            f.write(f"{pair_info}\n")

    with open(Path.home() / "scratch/yes-bias-in-llms" / "missing_cross_prompt_pairs.txt", 'w') as f:
        f.write(f"Missing Cross-Prompt Pairs: {cross_prompt_results['missing_data']}\n")
        f.write("="*80 + "\n\n")
        for pair_info in cross_prompt_results['missing_pairs_list']:
            f.write(f"{pair_info}\n")

    print("Missing pairs saved to missing_cross_*_pairs.txt files")
    
    # Print results
    print("\n" + "="*80)
    print("TRANSFER EFFICACY ANALYSIS RESULTS")
    print("="*80)
    
    print("\nCROSS-DATASET TRANSFER:")
    print(f"  Total Transfer Pairs: {cross_dataset_results['total_pairs']}")
    print(f"  Successful Pairs: {cross_dataset_results['successful_pairs']}")
    print(f"  Success Rate: {cross_dataset_results['successful_pairs']/cross_dataset_results['total_pairs']*100:.2f}%")
    print(f"  Failed (bias only): {cross_dataset_results['failed_bias_only']}")
    print(f"  Failed (accuracy only): {cross_dataset_results['failed_acc_only']}")
    print(f"  Failed (both): {cross_dataset_results['failed_both']}")
    print(f"  Avg Accuracy Change (successful): {cross_dataset_results['avg_succ_acc_change']:.6f}")
    print(f"  Avg TVD Change (successful): {cross_dataset_results['avg_succ_tvd_change']:.6f}")
    print(f"  Avg Accuracy Change (all): {cross_dataset_results['avg_all_acc_change']:.6f}")
    print(f"  Avg TVD Change (all): {cross_dataset_results['avg_all_tvd_change']:.6f}")

    print("\nCROSS-MODEL TRANSFER:")
    print(f"  Total Transfer Pairs: {cross_model_results['total_pairs']}")
    print(f"  Successful Pairs: {cross_model_results['successful_pairs']}")
    print(f"  Success Rate: {cross_model_results['successful_pairs']/cross_model_results['total_pairs']*100:.2f}%")
    print(f"  Failed (bias only): {cross_model_results['failed_bias_only']}")
    print(f"  Failed (accuracy only): {cross_model_results['failed_acc_only']}")
    print(f"  Failed (both): {cross_model_results['failed_both']}")
    print(f"  Avg Accuracy Change (successful): {cross_model_results['avg_succ_acc_change']:.6f}")
    print(f"  Avg TVD Change (successful): {cross_model_results['avg_succ_tvd_change']:.6f}")
    print(f"  Avg Accuracy Change (all): {cross_model_results['avg_all_acc_change']:.6f}")
    print(f"  Avg TVD Change (all): {cross_model_results['avg_all_tvd_change']:.6f}")
    
    print("\nCROSS-PROMPT TRANSFER:")
    print(f"  Total Transfer Pairs: {cross_prompt_results['total_pairs']}")
    print(f"  Successful Pairs: {cross_prompt_results['successful_pairs']}")
    print(f"  Success Rate: {cross_prompt_results['successful_pairs']/cross_prompt_results['total_pairs']*100:.2f}%")
    print(f"  Failed (bias only): {cross_prompt_results['failed_bias_only']}")
    print(f"  Failed (accuracy only): {cross_prompt_results['failed_acc_only']}")
    print(f"  Failed (both): {cross_prompt_results['failed_both']}")
    print(f"  Avg Accuracy Change (successful): {cross_prompt_results['avg_succ_acc_change']:.6f}")
    print(f"  Avg TVD Change (successful): {cross_prompt_results['avg_succ_tvd_change']:.6f}")
    print(f"  Avg Accuracy Change (all): {cross_prompt_results['avg_all_acc_change']:.6f}")
    print(f"  Avg TVD Change (all): {cross_prompt_results['avg_all_tvd_change']:.6f}")
    
    print("\n" + "="*80)
    
    # Save to CSV
    # output_file = Path.home() / "scratch/yes-bias-in-llms" / "transfer_efficacy_summary.csv"
    output_file = Path("../../results/table_outputs/transfer_success_table.csv")
    with open(output_file, 'w') as f:
        f.write("Transfer_Type,Total_Pairs,Successful_Pairs,Success_Rate, Failed_Bias_Only, Failed_Accuracy_Only, Failed_Both, Avg_Acc_Change,Avg_TVD_Change\n")
        f.write(f"Cross-Dataset,{cross_dataset_results['total_pairs']},{cross_dataset_results['successful_pairs']},"
                f"{cross_dataset_results['successful_pairs']/cross_dataset_results['total_pairs']*100:.2f},"
                f"{cross_dataset_results['failed_bias_only']},{cross_dataset_results['failed_acc_only']},{cross_dataset_results['failed_both']},"
                f"{cross_dataset_results['avg_succ_acc_change']:.6f},{cross_dataset_results['avg_succ_tvd_change']:.6f},"
                f"{cross_dataset_results['avg_all_acc_change']:.6f},{cross_dataset_results['avg_all_tvd_change']:.6f}\n")
        f.write(f"Cross-Model,{cross_model_results['total_pairs']},{cross_model_results['successful_pairs']},"
                f"{cross_model_results['successful_pairs']/cross_model_results['total_pairs']*100:.2f},"
                f"{cross_model_results['failed_bias_only']},{cross_model_results['failed_acc_only']},{cross_model_results['failed_both']},"
                f"{cross_model_results['avg_succ_acc_change']:.6f},{cross_model_results['avg_succ_tvd_change']:.6f},"
                f"{cross_model_results['avg_all_acc_change']:.6f},{cross_model_results['avg_all_tvd_change']:.6f}\n")
        f.write(f"Cross-Prompt,{cross_prompt_results['total_pairs']},{cross_prompt_results['successful_pairs']},"
                f"{cross_prompt_results['successful_pairs']/cross_prompt_results['total_pairs']*100:.2f},"
                f"{cross_prompt_results['failed_bias_only']},{cross_prompt_results['failed_acc_only']},{cross_prompt_results['failed_both']},"
                f"{cross_prompt_results['avg_succ_acc_change']:.6f},{cross_prompt_results['avg_succ_tvd_change']:.6f},"
                f"{cross_prompt_results['avg_all_acc_change']:.6f},{cross_prompt_results['avg_all_tvd_change']:.6f}\n")
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()