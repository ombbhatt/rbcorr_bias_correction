#!/usr/bin/env python3
"""
Batch Calibration Metrics Processor - Processes batch calibration results with multiple batch sizes.

This script reads batch calibration CSV files for different batch sizes and calculates TVD 
and accuracy metrics, comparing raw (uncorrected) vs batch-calibrated performance.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
import sys

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

def get_batchcalib_csv_paths(date: str, prompt: str, dataset: str, model_family: str, 
                             domain: str, model_name: str, question_type: str) -> Dict[str, Path]:
    """Get paths to all batch calibration CSV files for different batch sizes
    
    Returns:
        Dictionary mapping batch_size (as string) to Path
    """
    batch_sizes = [10, 20, 50, 100, 500, 1000]
    
    method = f"{question_type}batchcalib_fixedcounts"
    base_path = Path(__file__).parent.parent / "outputs" / date / prompt / dataset / method / model_family / domain
    
    paths = {}
    for batch_size in batch_sizes:
        csv_path = base_path / f"{model_name}_calib{batch_size}_results.csv"
        if csv_path.exists():
            paths[str(batch_size)] = csv_path

    return paths

def process_single_batch_size(csv_path: Path, question_type: str) -> Dict[str, Union[float, Dict]]:
    """Process a single batch size CSV file and extract metrics"""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if question_type == 'yesno':
        # Convert to format expected by calculate_tvd
        tvd_question_type = 'yn'
        
        # Calculate TVD for raw predictions
        raw_tvd, raw_model_dist, raw_gt_dist = calculate_tvd(
            df['raw_predicted_answer'].tolist(),
            df['Correct Answer'].tolist(),
            question_type=tvd_question_type
        )
        
        # Calculate TVD for corrected predictions
        corrected_tvd, corrected_model_dist, corrected_gt_dist = calculate_tvd(
            df['batchcalib_predicted_answer'].tolist(),
            df['Correct Answer'].tolist(),
            question_type=tvd_question_type
        )
        
        raw_acc = df['raw_is_correct'].mean()
        corrected_acc = df['batchcalib_is_correct'].mean()
        
        # Get batch bias estimates - note these may vary across batches
        # We'll take the mean across all batches for summary
        batch_yes_mean = df['batch_yes_mean'].mean() if len(df) > 0 else 0.0
        batch_no_mean = df['batch_no_mean'].mean() if len(df) > 0 else 0.0
        
        batch_bias_estimate = {
            "avg_batch_yes_mean": batch_yes_mean,
            "avg_batch_no_mean": batch_no_mean
        }
        
    elif question_type == 'mcq':  # mcq
        # Calculate TVD for raw predictions
        raw_tvd, raw_model_dist, raw_gt_dist = calculate_tvd(
            df['raw_predicted_answer'].tolist(),
            df['answer'].tolist(),
            question_type='mcq'
        )
        
        # Calculate TVD for corrected predictions
        corrected_tvd, corrected_model_dist, corrected_gt_dist = calculate_tvd(
            df['batchcalib_predicted_answer'].tolist(),
            df['answer'].tolist(),
            question_type='mcq'
        )
        
        raw_acc = df['raw_is_correct'].mean()
        corrected_acc = df['batchcalib_is_correct'].mean()
        
        # Get batch bias estimates - average across batches
        batch_oa_mean = df['batch_oa_mean'].mean() if len(df) > 0 else 0.0
        batch_ob_mean = df['batch_ob_mean'].mean() if len(df) > 0 else 0.0
        batch_oc_mean = df['batch_oc_mean'].mean() if len(df) > 0 else 0.0
        batch_od_mean = df['batch_od_mean'].mean() if len(df) > 0 else 0.0
        
        batch_bias_estimate = {
            "avg_batch_oa_mean": batch_oa_mean,
            "avg_batch_ob_mean": batch_ob_mean,
            "avg_batch_oc_mean": batch_oc_mean,
            "avg_batch_od_mean": batch_od_mean
        }

    elif question_type == 'nli':  # nli
        # Calculate TVD for raw predictions
        raw_tvd, raw_model_dist, raw_gt_dist = calculate_tvd(
            df['raw_predicted_answer'].tolist(),
            df['Correct Answer'].tolist(),
            question_type='nli'
        )
        
        # Calculate TVD for corrected predictions
        corrected_tvd, corrected_model_dist, corrected_gt_dist = calculate_tvd(
            df['batchcalib_predicted_answer'].tolist(),
            df['Correct Answer'].tolist(),
            question_type='nli'
        )
        
        raw_acc = df['raw_is_correct'].mean()
        corrected_acc = df['batchcalib_is_correct'].mean()
        
        # Get batch bias estimates - average across batches
        batch_o0_mean = df['batch_o0_mean'].mean() if len(df) > 0 else 0.0
        batch_o1_mean = df['batch_o1_mean'].mean() if len(df) > 0 else 0.0
        batch_o2_mean = df['batch_o2_mean'].mean() if len(df) > 0 else 0.0
        
        batch_bias_estimate = {
            "avg_batch_o0_mean": batch_o0_mean,
            "avg_batch_o1_mean": batch_o1_mean,
            "avg_batch_o2_mean": batch_o2_mean
        }
    
    return {
        "raw_tvd": raw_tvd,
        "raw_model_dist": raw_model_dist,
        "raw_acc": raw_acc,
        "corrected_tvd": corrected_tvd,
        "corrected_model_dist": corrected_model_dist,
        "corrected_acc": corrected_acc,
        "batch_bias_estimate": batch_bias_estimate,
        "ground_truth_dist": raw_gt_dist
    }

def get_dataset_info(dataset: str, aggregation_level: str) -> Tuple[str, List[str]]:
    """Get dataset and domain information based on aggregation level"""
    
    if aggregation_level == 'all':
        if dataset == 'YESNO':
            return 'ALL_YESNO', ['EWOK', 'COMPS', 'BABI', 'ARITH']
        elif dataset == 'MMLU':
            return 'ALL_MMLU', ['MMLU-STEM', 'MMLU-HUMANITIES', 'MMLU-SOCIAL_SCI', 'MMLU-OTHERS']
        elif dataset == 'NLI':
            return 'ALL_NLI', ['SNLI', 'MNLI']
    else:  # per
        if dataset in ['EWOK', 'COMPS', 'BABI', 'ARITH', 'SNLI', 'MNLI']:
            domain_map = {
                'EWOK': 'all_domains',
                'COMPS': 'comps', 
                'BABI': 'babi',
                'ARITH': 'arith',
                'SNLI': 'snli',
                'MNLI': 'mnli'
            }
            return dataset, [domain_map[dataset]]
        elif dataset.startswith('MMLU-'):
            domain = dataset.split('-')[1]  # Extract STEM, HUMANITIES, etc.
            return dataset, [domain]
    
    raise ValueError(f"Invalid dataset: {dataset}")

def get_model_configs() -> Dict[str, List[str]]:
    """Get model family configurations"""
    return {
        "Falcon": ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"],
        "Gemma3": ["gemma-3-27b-pt", "gemma-3-27b-it", "gemma-3-12b-pt", "gemma-3-12b-it"],
        "Llama3": ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"]
    }

def aggregate_across_datasets_for_batch_size(question_type: str, prompt: str, model_family: str,
                                             datasets: List[str], batch_size: str, 
                                             date: str = "Sep-16-2025") -> Dict[str, Dict[str, Union[float, Dict]]]:
    """Aggregate metrics across multiple datasets for a specific batch size"""
    model_configs = get_model_configs()
    aggregated_results = {}
    
    for model_name in model_configs[model_family]:
        all_dfs = []
        
        # Collect data from all datasets for this specific batch size
        for dataset in datasets:
            _, domains = get_dataset_info(dataset, 'per')
            domain = domains[0]
            
            # Get path for this specific batch size
            csv_paths = get_batchcalib_csv_paths(date, prompt, dataset, model_family, 
                                                domain, model_name, question_type)
            
            if batch_size in csv_paths:
                df = pd.read_csv(csv_paths[batch_size])
                all_dfs.append(df)
        
        if all_dfs:
            # Combine all dataframes for this batch size
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # Calculate metrics on combined data
            if question_type == 'yesno':
                tvd_question_type = 'yn'
                
                # Calculate TVD for raw predictions
                raw_tvd, raw_model_dist, raw_gt_dist = calculate_tvd(
                    combined_df['raw_predicted_answer'].tolist(),
                    combined_df['Correct Answer'].tolist(),
                    question_type=tvd_question_type
                )
                
                # Calculate TVD for corrected predictions
                corrected_tvd, corrected_model_dist, corrected_gt_dist = calculate_tvd(
                    combined_df['batchcalib_predicted_answer'].tolist(),
                    combined_df['Correct Answer'].tolist(),
                    question_type=tvd_question_type
                )
                
                raw_acc = combined_df['raw_is_correct'].mean()
                corrected_acc = combined_df['batchcalib_is_correct'].mean()
                
                # Average batch means across all datasets
                batch_yes_mean = combined_df['batch_yes_mean'].mean() if len(combined_df) > 0 else 0.0
                batch_no_mean = combined_df['batch_no_mean'].mean() if len(combined_df) > 0 else 0.0
                
                batch_bias_estimate = {
                    "avg_batch_yes_mean": batch_yes_mean,
                    "avg_batch_no_mean": batch_no_mean
                }
                
            elif args.question_type == 'mcq':  # mcq
                # Calculate TVD for raw predictions
                raw_tvd, raw_model_dist, raw_gt_dist = calculate_tvd(
                    combined_df['raw_predicted_answer'].tolist(),
                    combined_df['answer'].tolist(),
                    question_type='mcq'
                )
                
                # Calculate TVD for corrected predictions
                corrected_tvd, corrected_model_dist, corrected_gt_dist = calculate_tvd(
                    combined_df['batchcalib_predicted_answer'].tolist(),
                    combined_df['answer'].tolist(),
                    question_type='mcq'
                )
                
                raw_acc = combined_df['raw_is_correct'].mean()
                corrected_acc = combined_df['batchcalib_is_correct'].mean()
                
                # Average batch means across all datasets
                batch_oa_mean = combined_df['batch_oa_mean'].mean() if len(combined_df) > 0 else 0.0
                batch_ob_mean = combined_df['batch_ob_mean'].mean() if len(combined_df) > 0 else 0.0
                batch_oc_mean = combined_df['batch_oc_mean'].mean() if len(combined_df) > 0 else 0.0
                batch_od_mean = combined_df['batch_od_mean'].mean() if len(combined_df) > 0 else 0.0
                
                batch_bias_estimate = {
                    "avg_batch_oa_mean": batch_oa_mean,
                    "avg_batch_ob_mean": batch_ob_mean,
                    "avg_batch_oc_mean": batch_oc_mean,
                    "avg_batch_od_mean": batch_od_mean
                }

            elif question_type == 'nli':  # nli
                # Calculate TVD for raw predictions
                raw_tvd, raw_model_dist, raw_gt_dist = calculate_tvd(
                    combined_df['raw_predicted_answer'].tolist(),
                    combined_df['Correct Answer'].tolist(),
                    question_type='nli'
                )
                
                # Calculate TVD for corrected predictions
                corrected_tvd, corrected_model_dist, corrected_gt_dist = calculate_tvd(
                    combined_df['batchcalib_predicted_answer'].tolist(),
                    combined_df['Correct Answer'].tolist(),
                    question_type='nli'
                )
                
                raw_acc = combined_df['raw_is_correct'].mean()
                corrected_acc = combined_df['batchcalib_is_correct'].mean()
                
                # Average batch means across all datasets
                batch_o0_mean = combined_df['batch_o0_mean'].mean() if len(combined_df) > 0 else 0.0
                batch_o1_mean = combined_df['batch_o1_mean'].mean() if len(combined_df) > 0 else 0.0
                batch_o2_mean = combined_df['batch_o2_mean'].mean() if len(combined_df) > 0 else 0.0
                
                batch_bias_estimate = {
                    "avg_batch_o0_mean": batch_o0_mean,
                    "avg_batch_o1_mean": batch_o1_mean,
                    "avg_batch_o2_mean": batch_o2_mean
                }
            
            aggregated_results[model_name] = {
                "raw_tvd": raw_tvd,
                "raw_model_dist": raw_model_dist,
                "raw_acc": raw_acc,
                "corrected_tvd": corrected_tvd,
                "corrected_model_dist": corrected_model_dist,
                "corrected_acc": corrected_acc,
                "batch_bias_estimate": batch_bias_estimate,
                "ground_truth_dist": raw_gt_dist
            }
    
    return aggregated_results

def main():
    parser = argparse.ArgumentParser(description='Process batch calibration metrics with multiple batch sizes')
    parser.add_argument('--question_type', choices=['yesno', 'nli', 'mcq'], required=True,
                      help='Question type: yesno (yes-no) or mcq (multiple choice) or nli (entailment)')
    parser.add_argument('--aggregation_level', choices=['per', 'all'], required=True,
                      help='Aggregation level: per (individual datasets) or all (combined)')
    parser.add_argument('--model_family', 
                      choices=['Falcon', 'Gemma3', 'Llama3'],
                      help='Model family to process (default: all)')
    parser.add_argument('--prompt_type', 
                      choices=['zeroshot', 'instronly', 'fewshot'],
                      help='Prompt type (default: all)')
    parser.add_argument('--dataset', required=True,
                      help='Dataset to process (e.g., BABI, MMLU-HUMANITIES, YESNO, MMLU, SNLI, MNLI)')
    parser.add_argument('--date', default='Sep-16-2025',
                      help='Date folder (default: Sep-16-2025)')
    
    args = parser.parse_args()
    
    # Set defaults if not provided
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
        
        # Process each combination of prompt_type and model_family
        for prompt_type in prompt_types:
            for model_family in model_families:
                print(f"\nProcessing {prompt_type} - {model_family}")
                
                model_configs = get_model_configs()
                
                # Create output directory
                output_dir = Path(__file__).parent.parent / "results" / f"batchcalib_{args.question_type}_{args.aggregation_level}_TVD"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Initialize results structure
                results = {
                    prompt_type: {
                        dataset_name: {}
                    }
                }
                
                if args.aggregation_level == 'all':
                    # Handle aggregated case - loop through batch sizes
                    if args.question_type == 'yesno':
                        datasets_to_aggregate = ['EWOK', 'COMPS', 'BABI', 'ARITH']
                        domain_key = 'all_yesno_domains'
                    elif args.question_type == 'nli':
                        datasets_to_aggregate = ['SNLI', 'MNLI']
                        domain_key = 'all_nli_domains'
                    else:  # mcq
                        datasets_to_aggregate = ['MMLU-STEM', 'MMLU-HUMANITIES', 'MMLU-SOCIAL_SCI', 'MMLU-OTHERS']
                        domain_key = 'all_mcq_domains'

                    batch_sizes = [10, 20, 50, 100, 500, 1000]
                    
                    # Initialize nested structure for batch sizes
                    results[prompt_type][dataset_name][domain_key] = {
                        model_family: {}
                    }
                    
                    # Get all models for this family
                    for model_name in model_configs[model_family]:
                        results[prompt_type][dataset_name][domain_key][model_family][model_name] = {}
                    
                    # Process each batch size
                    for batch_size in batch_sizes:
                        print(f"  Aggregating batch_size: {batch_size}")
                        
                        aggregated_results = aggregate_across_datasets_for_batch_size(
                            args.question_type, prompt_type, model_family,
                            datasets_to_aggregate, str(batch_size), args.date
                        )
                        
                        # Store results for each model under this batch size
                        for model_name, metrics in aggregated_results.items():
                            batch_size_key = str(batch_size)
                            results[prompt_type][dataset_name][domain_key][model_family][model_name][batch_size_key] = metrics
                    
                else:
                    # Handle per-dataset case
                    for domain in domains:
                        results[prompt_type][dataset_name][domain] = {
                            model_family: {}
                        }
                        
                        for model_name in model_configs[model_family]:
                            print(f"  Processing {model_name}")
                            
                            # Get paths for all batch sizes
                            csv_paths = get_batchcalib_csv_paths(args.date, prompt_type, args.dataset,
                                                                model_family, domain, model_name, args.question_type)
                            
                            if not csv_paths:
                                print(f"    No batch calibration files found for {model_name}")
                                continue
                            
                            model_results = {}
                            for batch_size, csv_path in csv_paths.items():
                                try:
                                    metrics = process_single_batch_size(csv_path, args.question_type)
                                    model_results[batch_size] = metrics
                                    print(f"    Processed batch_size: {batch_size}")
                                except Exception as e:
                                    print(f"    Error processing batch_size {batch_size}: {e}")
                            
                            if model_results:
                                results[prompt_type][dataset_name][domain][model_family][model_name] = model_results
                
                # Write results to JSON
                output_file = output_dir / f"{prompt_type}_{model_family}_{args.dataset}.json"
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"  Results written to: {output_file}")
        
        print(f"\nAll processing complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()