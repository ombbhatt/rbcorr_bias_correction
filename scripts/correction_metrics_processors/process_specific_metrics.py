#!/usr/bin/env python3
"""
Specific Correction Metrics Processor - Processes specific correction results with 100 calibration runs.

This script handles three transfer modalities:
1. Same-dataset, same-model (baseline)
2. Cross-dataset transfer (different calibration dataset, same model)
3. Cross-model transfer (same dataset, different calibration model)
4. Cross-prompt transfer (same dataset, same model, different calibration prompt)
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import sys
import ast

# def calculate_yesno_bias(predictions: List[str]) -> float:
#     """Calculate yes-no bias metric from a list of predictions"""
#     yes_count = sum(1 for pred in predictions if pred == 'Yes')
#     no_count = sum(1 for pred in predictions if pred == 'No')
#     total = len(predictions)
#     if total == 0:
#         return 0.0
#     return (yes_count - no_count) / total

# def calculate_rstd(predictions: List[str], answers: List[str]) -> float:
#     """Calculate Recall Standard Deviation for MCQ position bias"""
#     if len(predictions) != len(answers):
#         # Handle length mismatch gracefully
#         min_len = min(len(predictions), len(answers))
#         predictions = predictions[:min_len]
#         answers = answers[:min_len]
    
#     recalls = []
#     for option in ['A', 'B', 'C', 'D']:
#         # Get questions where this option is correct
#         correct_indices = [i for i, ans in enumerate(answers) if ans == option]
#         if len(correct_indices) == 0:
#             recalls.append(0.0)
#             continue
        
#         # Calculate recall for this position
#         correct_predictions = sum(1 for i in correct_indices if predictions[i] == option)
#         recall = correct_predictions / len(correct_indices)
#         recalls.append(recall)
    
#     return np.std(recalls)

from typing import List, Dict, Tuple, Optional

def calculate_tvd(
    predictions: List[str], 
    answers: List[str],
    question_type: str = "mcq"
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Calculate Total Variation Distance for different question types
    
    Args:
        predictions: List of model predictions
        answers: List of ground truth answers
        question_type: One of "yn" (yes/no), "mcq" (multiple choice), "nli" (entailment)
    
    Returns:
        tvd: Total Variation Distance (0 to 1)
        model_dist: Dictionary of model's answer distribution
        ground_truth_dist: Dictionary of ground truth answer distribution
    """
    if len(predictions) != len(answers):
        # Handle length mismatch gracefully
        min_len = min(len(predictions), len(answers))
        predictions = predictions[:min_len]
        answers = answers[:min_len]
    
    # Define options based on question type
    if question_type == "yn":
        options = ['Yes', 'No']
    elif question_type == "mcq":
        options = ['A', 'B', 'C', 'D']
    elif question_type == "nli":
        options = ['0', '1', '2']
    else:
        raise ValueError(f"Unknown question_type: {question_type}. Must be 'yn', 'mcq', or 'nli'")
    
    total = len(predictions)
    
    # Calculate model distribution (what the model actually chose)
    model_dist = {}
    for option in options:
        count = sum(1 for pred in predictions if str(pred) == option)
        model_dist[str(option)] = count / total
    
    # Calculate ground truth distribution (what the correct answers are)
    ground_truth_dist = {}
    for option in options:
        count = sum(1 for ans in answers if str(ans) == option)
        ground_truth_dist[str(option)] = count / total
    
    # Calculate TVD
    tvd = 0.5 * sum(abs(model_dist[option] - ground_truth_dist[option]) for option in options)
    
    return tvd, model_dist, ground_truth_dist

def parse_list_column(series: pd.Series) -> List[List]:
    """Parse string representations of lists back to actual lists"""
    result = []
    for item in series:
        if pd.isna(item):
            result.append([])
        elif isinstance(item, str):
            try:
                # Handle string representation of lists
                parsed = ast.literal_eval(item)
                if isinstance(parsed, list):
                    result.append(parsed)
                else:
                    result.append([parsed])
            except (ValueError, SyntaxError):
                result.append([])
        elif isinstance(item, list):
            result.append(item)
        else:
            result.append([item])
    return result

def get_specific_csv_paths(date: str, prompt: str, dataset: str, model_family: str,
                          domain: str, model_name: str, question_type: str,
                          cross_dataset: Optional[str] = None, 
                          cross_model: Optional[str] = None,
                          cross_prompt: Optional[str] = None) -> Dict[int, Path]:
    """Get paths to all specific correction CSV files for different calibration counts
    
    Args:
        date: Date folder
        prompt: Target prompt type
        dataset: Target dataset
        model_family: Model family
        domain: Domain within dataset
        model_name: Target model name
        question_type: 'yesno' or 'mcq'
        cross_dataset: Source dataset for cross-dataset correction (None for same-dataset)
        cross_model: Source model for cross-model correction (None for same-model)
        cross_prompt: Source prompt for cross-prompt correction (None for same-prompt)
    """

    calib_counts = [20, 50, 100, 500, 1000]
    
    # Build path components based on transfer modality
    # Priority: cross_prompt > cross_dataset > cross_model (can combine)
    
    # 1. Determine prompt folder (target prompt with optional cross-prompt suffix)
    if cross_prompt:
        prompt_folder = f"{prompt}_from{cross_prompt}"
    else:
        prompt_folder = prompt
    
    # 2. Determine method folder (with optional cross-dataset suffix)
    if cross_dataset:
        method = f"{question_type}specific_fixedcounts_from{cross_dataset}_median"
    else:
        method = f"{question_type}specific_fixedcounts_median"
    
    base_path = Path(__file__).parent.parent / "outputs" / date / prompt_folder / dataset / method / model_family / domain
    
    paths = {}
    for count in calib_counts:
        # 3. Build filename (with optional cross-model suffix)
        if count != 0:
            if cross_model:
                csv_path = base_path / f"{model_name}_from{cross_model}_calib{count}_results.csv"
            else:
                csv_path = base_path / f"{model_name}_calib{count}_results.csv"
        else:
            if cross_model:
                csv_path = base_path / f"{model_name}_from{cross_model}_fullbatch_results.csv"
            else:
                csv_path = base_path / f"{model_name}_fullbatch_results.csv"
        
        if csv_path.exists():
            paths[count] = csv_path
    
    return paths


def process_single_calib_count(csv_path: Path, question_type: str) -> Dict[str, Union[float, int]]:
    """Process a single calibration count CSV file and extract statistics across 100 runs"""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        return {}
    
    # Parse list columns
    specific_predictions = parse_list_column(df['specific_predicted_answer'])
    specific_correct = parse_list_column(df['specific_is_correct'])
    
    # Determine baseline column names and answer column based on question type
    if question_type == 'yesno':
        baseline_pred_col = 'raw_predicted_answer'
        baseline_corr_col = 'raw_is_correct'
        answer_col = 'Correct Answer'
        tvd_question_type = 'yn'
    elif question_type == 'mcq':  # mcq
        baseline_pred_col = 'plain_predicted_answer'
        baseline_corr_col = 'plain_is_correct'
        answer_col = 'answer'
        tvd_question_type = 'mcq'
    elif question_type == 'nli':  # nli
        baseline_pred_col = 'plain_predicted_answer'
        baseline_corr_col = 'plain_is_correct'
        answer_col = 'Correct Answer'
        tvd_question_type = 'nli'
    
    # Handle baseline predictions with fallback
    if baseline_pred_col in df.columns:
        baseline_predictions = parse_list_column(df[baseline_pred_col])
        baseline_correct = parse_list_column(df[baseline_corr_col])
    else:
        raise ValueError(f"Baseline prediction column '{baseline_pred_col}' not found in CSV file: {csv_path}")
    
    answers = df[answer_col].tolist()
    bias_metric_name = "tvd"
    
    # Process calibration runs
    bias_values = {"tvd": [], "model_dist": []}
    acc_values = []
    
    # Determine number of runs (should be 100)
    num_runs = len(specific_predictions[0]) if len(specific_predictions) > 0 and len(specific_predictions[0]) > 0 else 0
    
    for run_idx in range(num_runs):
        # Extract predictions for this run across all questions
        run_predictions = []
        run_correct = []
        
        for question_idx in range(len(specific_predictions)):
            if run_idx < len(specific_predictions[question_idx]):
                run_predictions.append(specific_predictions[question_idx][run_idx])
                run_correct.append(specific_correct[question_idx][run_idx])
        
        if len(run_predictions) > 0:
            # Calculate bias for this run
            tvd, model_dist, ground_truth_dist = calculate_tvd(run_predictions, answers, question_type=tvd_question_type)
            bias_values["tvd"].append(tvd)
            bias_values["model_dist"].append(model_dist)
            
            # Calculate accuracy for this run
            acc = sum(run_correct) / len(run_correct) if len(run_correct) > 0 else 0.0
            acc_values.append(acc)
    
    # Calculate baseline metrics for comparison
    baseline_pred_first = [preds[0] for preds in baseline_predictions]
    raw_bias, raw_model_dist, ground_truth_dist = calculate_tvd(baseline_pred_first, answers, question_type=tvd_question_type)
    bias_values["ground_truth_dist"] = ground_truth_dist
    
    # Calculate baseline accuracy
    baseline_corr_first = [corr[0] if len(corr) > 0 else False for corr in baseline_correct]
    raw_acc = sum(baseline_corr_first) / len(baseline_corr_first) if len(baseline_corr_first) > 0 else 0.0
    
    if len(bias_values["tvd"]) == 0:
        return {}
    
    # Calculate statistics across 100 runs
    stats = {
        # Bias statistics (best = lowest)
        f"best_{bias_metric_name}": float(np.min(bias_values["tvd"])),
        f"best_{bias_metric_name}_model_dist": str(bias_values["model_dist"][np.argmin(bias_values["tvd"])]),
        f"worst_{bias_metric_name}": float(np.max(bias_values["tvd"])),
        f"worst_{bias_metric_name}_model_dist": str(bias_values["model_dist"][np.argmax(bias_values["tvd"])]),
        f"mean_{bias_metric_name}": float(np.mean(bias_values["tvd"])),
        f"mean_{bias_metric_name}_model_dist": str(bias_values["model_dist"][np.argmin(np.abs(bias_values["tvd"] - np.mean(bias_values["tvd"])))]),
        f"median_{bias_metric_name}": float(np.median(bias_values["tvd"])),
        f"median_{bias_metric_name}_model_dist": str(bias_values["model_dist"][np.argmin(np.abs(bias_values["tvd"] - np.median(bias_values["tvd"])))]),
        f"std_{bias_metric_name}": float(np.std(bias_values["tvd"])),
        f"q25_{bias_metric_name}": float(np.percentile(bias_values["tvd"], 25)),
        f"q25_{bias_metric_name}_model_dist": str(bias_values["model_dist"][np.argsort(bias_values["tvd"])[int(0.25 * len(bias_values["tvd"]))]]),
        f"q75_{bias_metric_name}": float(np.percentile(bias_values["tvd"], 75)),
        f"q75_{bias_metric_name}_model_dist": str(bias_values["model_dist"][np.argsort(bias_values["tvd"])[int(0.75 * len(bias_values["tvd"]))]]),

        # Accuracy statistics (best = highest)
        "best_run_acc": float(np.max(acc_values)),
        "worst_run_acc": float(np.min(acc_values)),
        "mean_acc": float(np.mean(acc_values)),
        "median_acc": float(np.median(acc_values)),
        "std_acc": float(np.std(acc_values)),
        "q25_acc": float(np.percentile(acc_values, 25)),
        "q75_acc": float(np.percentile(acc_values, 75)),

        # Raw baseline for comparison
        f"raw_{bias_metric_name}": float(raw_bias),
        "raw_acc": float(raw_acc),
        "raw_model_dist": str(raw_model_dist),
        "ground_truth_dist": str(bias_values["ground_truth_dist"]),
        
        # Metadata
        "num_calib_sets": len(bias_values["tvd"]),
    }
    
    return stats

def get_dataset_info(dataset: str, aggregation_level: str) -> Tuple[str, List[str]]:
    """Get dataset and domain information based on aggregation level"""
    
    if aggregation_level == 'all':
        if dataset == 'YESNO':
            return 'ALL_YESNO', ['EWOK', 'COMPS', 'BABI', 'ARITH']
        elif dataset == 'MMLU':
            return 'ALL_MMLU', ['MMLU-STEM', 'MMLU-HUMANITIES', 'MMLU-SOCIAL_SCI', 'MMLU-OTHERS']
        elif dataset == "NLI":
            return 'ALL_NLI', ['SNLI', 'MNLI']
    elif aggregation_level == 'per':  # per
        if dataset in ['EWOK', 'COMPS', 'BABI', 'ARITH']:
            domain_map = {
                'EWOK': 'all_domains',
                'COMPS': 'comps',
                'BABI': 'babi',
                'ARITH': 'arith'
            }
            return dataset, [domain_map[dataset]]
        elif dataset in ['SNLI', 'MNLI']:
            domain_map = {
                'SNLI': 'snli', 
                'MNLI': 'mnli'
            }
            return dataset, [domain_map[dataset]]
        elif dataset.startswith('MMLU-'):
            domain = dataset.split('-')[1]
            return dataset, [domain]
    
    raise ValueError(f"Invalid dataset: {dataset}")

def get_model_configs() -> Dict[str, List[str]]:
    """Get model family configurations"""
    return {
        "Falcon": ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"],
        "Gemma3": ["gemma-3-27b-pt", "gemma-3-27b-it", "gemma-3-12b-pt", "gemma-3-12b-it"],
        # "Qwen2": ["Qwen2.5-14B", "Qwen2.5-14B-Instruct", "Qwen2.5-32B", "Qwen2.5-32B-Instruct"],
        # "Llama": ["Llama-2-7b-hf", "Llama-2-7b-chat-hf", "Llama-2-13b-hf", "Llama-2-13b-chat-hf"],
        "Llama3": ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"]
    }

def get_cross_dataset_sources(question_type: str, target_dataset: str) -> List[str]:
    """Get list of possible cross-dataset sources for a target dataset"""
    if question_type == 'yesno':
        all_datasets = ['EWOK', 'COMPS', 'BABI', 'ARITH']
        return [d for d in all_datasets if d != target_dataset]
    elif question_type == 'mcq':  # mcq
        if target_dataset.startswith('MMLU-'):
            all_domains = ['MMLU-STEM', 'MMLU-HUMANITIES', 'MMLU-SOCIAL_SCI', 'MMLU-OTHERS']
            return [d for d in all_domains if d != target_dataset]
    elif question_type == 'nli':  # nli
        all_datasets = ['SNLI', 'MNLI']
        return [d for d in all_datasets if d != target_dataset]
    return []

def get_cross_model_sources(model_family: str, target_model: str) -> List[str]:
    """Get list of possible cross-model sources within the same family"""
    model_configs = get_model_configs()
    all_models = model_configs.get(model_family, [])
    return [m for m in all_models if m != target_model]

def get_cross_prompt_sources(target_prompt: str, per_type: str) -> List[str]:
    """Get list of possible cross-prompt sources"""
    all_prompts = ['zeroshot', 'instronly', 'fewshot'] if per_type != "PER_NLI" else ['instronly', 'fewshot']
    return [p for p in all_prompts if p != target_prompt]

def build_domain_identifier(target_dataset: str, cross_dataset: Optional[str] = None,
                           target_model: Optional[str] = None, cross_model: Optional[str] = None,
                           target_prompt: Optional[str] = None, cross_prompt: Optional[str] = None) -> str:
    """Build domain identifier for JSON structure
    
    Handles all combinations of transfer modalities:
    - Same-dataset, same-model, same-prompt: "BABI-fromBABI"
    - Cross-dataset: "BABI-fromCOMPS"  
    - Cross-model: "BABI-fromBABI_Llama-2-7b-hf_fromLlama-2-13b-hf"
    - Cross-prompt: "BABI-fromBABI_fewshot_frominstronly"
    - Combinations: "BABI-fromCOMPS_Llama-2-7b-hf_fromLlama-2-13b-hf"
    """
    # Build dataset part
    if target_dataset.startswith('MMLU-'):
        target_domain = target_dataset.split('-')[1]
        if cross_dataset:
            source_domain = cross_dataset.split('-')[1] if cross_dataset.startswith('MMLU-') else cross_dataset
            dataset_part = f"{target_domain}-from{source_domain}"
        else:
            dataset_part = f"{target_domain}-from{target_domain}"
    else:
        if cross_dataset:
            dataset_part = f"{target_dataset}-from{cross_dataset}"
        else:
            dataset_part = f"{target_dataset}-from{target_dataset}"
    
    # Add model part if cross-model
    if cross_model and target_model:
        dataset_part = f"{dataset_part}_{target_model}_from{cross_model}"
    
    # Add prompt part if cross-prompt
    if cross_prompt and target_prompt:
        dataset_part = f"{dataset_part}_{target_prompt}_from{cross_prompt}"
    
    return dataset_part



def deep_merge(dict1, dict2):
    """Recursively merge dict2 into dict1"""
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            deep_merge(dict1[key], value)
        else:
            dict1[key] = value
    return dict1


def main():
    parser = argparse.ArgumentParser(description='Process specific correction metrics with transfer modalities')
    parser.add_argument('--question_type', choices=['yesno', 'mcq', 'nli'], required=True,
                      help='Question type: yesno, mcq, or nli')
    parser.add_argument('--aggregation_level', choices=['per', 'all'], required=True,
                      help='Aggregation level: per (individual) or all (combined)')
    parser.add_argument('--model_family',
                      choices=['Falcon', 'Gemma3', 'Llama3'],
                      help='Model family to process (default: all)')
    parser.add_argument('--prompt_type',
                      choices=['zeroshot', 'instronly', 'fewshot'],
                      help='Target prompt type (default: all)')
    parser.add_argument('--dataset', required=True,
                      help='Target dataset (e.g., BABI, MMLU-HUMANITIES)')
    parser.add_argument('--include_cross_dataset', action='store_true',
                      help='Include cross-dataset transfer results')
    parser.add_argument('--include_cross_model', action='store_true',
                      help='Include cross-model transfer results')
    parser.add_argument('--include_cross_prompt', action='store_true',
                      help='Include cross-prompt transfer results')
    parser.add_argument('--date', default='Sep-16-2025',
                      help='Date folder (default: Sep-16-2025)')
    
    args = parser.parse_args()
    
    # Set defaults
    model_families = [args.model_family] if args.model_family else list(get_model_configs().keys())
    # prompt_types = [args.prompt_type] if args.prompt_type else ['zeroshot', 'instronly', 'fewshot']
    if args.prompt_type:
        prompt_types = [args.prompt_type]
    else:
        if args.question_type == 'nli':
            prompt_types = ['instronly', 'fewshot']
        else:
            prompt_types = ['zeroshot', 'instronly', 'fewshot']
    
    try:
        dataset_name, domains = get_dataset_info(args.dataset, args.aggregation_level)
        
        # Process each combination
        for prompt_type in prompt_types:
            for model_family in model_families:
                print(f"\nProcessing {prompt_type} - {model_family}")
                
                model_configs = get_model_configs()
                
                # Create output directory
                output_dir = Path(__file__).parent.parent / "results" / f"specific_{args.question_type}_{args.aggregation_level}_median_TVD"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Initialize results structure
                if args.aggregation_level == 'all':
                    print("  Aggregation across datasets not yet implemented for specific correction")
                    continue
                
                if args.question_type == 'yesno': 
                    per_type = "PER_YESNO"
                elif args.question_type == 'mcq': 
                    per_type = "PER_MMLU"
                elif args.question_type == 'nli': 
                    per_type = "PER_NLI"
                    
                # Per-dataset processing
                results = {
                    prompt_type: {
                        per_type: {}
                    }
                }
                
                for domain in domains:
                    # 1. Process same-dataset, same-model, same-prompt (baseline)
                    print(f"  Processing baseline: {args.dataset} (same-dataset, same-model, same-prompt)")
                    domain_identifier = build_domain_identifier(args.dataset)
                    
                    if domain_identifier not in results[prompt_type][per_type]:
                        results[prompt_type][per_type][domain_identifier] = {
                            model_family: {}
                        }
                    
                    for model_name in model_configs[model_family]:
                        csv_paths = get_specific_csv_paths(
                            args.date, prompt_type, args.dataset,
                            model_family, domain, model_name, args.question_type
                        )
                        
                        model_results = {}
                        for calib_count, csv_path in csv_paths.items():
                            try:
                                stats = process_single_calib_count(csv_path, args.question_type)
                                if stats:
                                    model_results[str(calib_count)] = stats
                            except Exception as e:
                                print(f"    Error processing {model_name} calib{calib_count}: {e}")
                        
                        if model_results:
                            results[prompt_type][per_type][domain_identifier][model_family][model_name] = model_results
                            print(f"    Processed {model_name}: {len(model_results)} calibration counts")
                    
                    # 2. Process cross-dataset transfer
                    if args.include_cross_dataset:
                        cross_datasets = get_cross_dataset_sources(args.question_type, args.dataset)
                        
                        for cross_dataset in cross_datasets:
                            print(f"  Processing cross-dataset: {cross_dataset} → {args.dataset}")
                            cross_identifier = build_domain_identifier(args.dataset, cross_dataset)
                            
                            if cross_identifier not in results[prompt_type][per_type]:
                                results[prompt_type][per_type][cross_identifier] = {
                                    model_family: {}
                                }
                            
                            for model_name in model_configs[model_family]:
                                csv_paths = get_specific_csv_paths(
                                    args.date, prompt_type, args.dataset,
                                    model_family, domain, model_name, args.question_type,
                                    cross_dataset=cross_dataset
                                )
                                
                                model_results = {}
                                for calib_count, csv_path in csv_paths.items():
                                    try:
                                        stats = process_single_calib_count(csv_path, args.question_type)
                                        if stats:
                                            model_results[str(calib_count)] = stats
                                    except Exception as e:
                                        print(f"    Error: {e}")
                                
                                if model_results:
                                    results[prompt_type][per_type][cross_identifier][model_family][model_name] = model_results
                    
                    # 3. Process cross-model transfer
                    if args.include_cross_model:
                        print(f"  Processing cross-model transfers")
                        
                        for model_name in model_configs[model_family]:
                            cross_models = get_cross_model_sources(model_family, model_name)
                            
                            for cross_model in cross_models:
                                cross_identifier = build_domain_identifier(
                                    args.dataset, None, model_name, cross_model
                                )
                                
                                if cross_identifier not in results[prompt_type][per_type]:
                                    results[prompt_type][per_type][cross_identifier] = {
                                        model_family: {}
                                    }
                                
                                csv_paths = get_specific_csv_paths(
                                    args.date, prompt_type, args.dataset,
                                    model_family, domain, model_name, args.question_type,
                                    cross_model=cross_model
                                )
                                
                                model_results = {}
                                for calib_count, csv_path in csv_paths.items():
                                    try:
                                        stats = process_single_calib_count(csv_path, args.question_type)
                                        if stats:
                                            model_results[str(calib_count)] = stats
                                    except Exception as e:
                                        pass
                                
                                if model_results:
                                    results[prompt_type][per_type][cross_identifier][model_family][model_name] = model_results
                    
                    # 4. Process cross-prompt transfer
                    if args.include_cross_prompt:
                        cross_prompts = get_cross_prompt_sources(prompt_type, per_type)
                        
                        for cross_prompt in cross_prompts:
                            print(f"  Processing cross-prompt: {cross_prompt} → {prompt_type}")
                            cross_identifier = build_domain_identifier(
                                args.dataset, None, None, None, prompt_type, cross_prompt
                            )
                            
                            if cross_identifier not in results[prompt_type][per_type]:
                                results[prompt_type][per_type][cross_identifier] = {
                                    model_family: {}
                                }
                            
                            for model_name in model_configs[model_family]:
                                csv_paths = get_specific_csv_paths(
                                    args.date, prompt_type, args.dataset,
                                    model_family, domain, model_name, args.question_type,
                                    cross_prompt=cross_prompt
                                )
                                
                                model_results = {}
                                for calib_count, csv_path in csv_paths.items():
                                    try:
                                        stats = process_single_calib_count(csv_path, args.question_type)
                                        if stats:
                                            model_results[str(calib_count)] = stats
                                    except Exception as e:
                                        pass
                                
                                if model_results:
                                    results[prompt_type][per_type][cross_identifier][model_family][model_name] = model_results
                 
                # Write results to JSON
                # output_file = output_dir / f"{prompt_type}_{model_family}_{args.dataset}.json"
                # with open(output_file, 'w') as f:
                #     json.dump(results, f, indent=2)
                
                # print(f"  Results written to: {output_file}")
                output_file = output_dir / f"{prompt_type}_{model_family}_{args.dataset}.json"

                # Read existing JSON if it exists and merge
                if output_file.exists():
                    print(f"  Found existing JSON, merging results...")
                    with open(output_file, 'r') as f:
                        existing_results = json.load(f)
                    results = deep_merge(existing_results, results)

                # Write merged results
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)

                print(f"  Results written to: {output_file}")
        
        print(f"\nAll processing complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()