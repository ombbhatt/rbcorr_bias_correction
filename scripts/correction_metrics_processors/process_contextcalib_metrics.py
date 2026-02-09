#!/usr/bin/env python3
"""
Contextual Calibration Metrics Processor - Processes contextual calibration results and generates JSON metrics files.

This script reads contextual calibration CSV files and calculates TVD and accuracy metrics,
comparing raw (uncorrected) vs contextually-calibrated performance.
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

def get_contextcalib_csv_path(date: str, prompt: str, dataset: str, model_family: str, 
                             domain: str, model_name: str, question_type: str) -> Path:
    """Construct path to contextual calibration CSV file"""
    method = f"{question_type}contextcalib"
    # Assume script is in scripts/ folder, so go up one level to reach outputs/
    base_path = Path(__file__).parent.parent / "outputs" / date / prompt / dataset / method / model_family / domain
    return base_path / f"{model_name}_results.csv"

def process_single_model(csv_path: Path, question_type: str) -> Dict[str, Union[float, Dict]]:
    """Process a single contextual calibration CSV file and extract metrics"""
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
            df['contextcalib_predicted_answer'].tolist(),
            df['Correct Answer'].tolist(),
            question_type=tvd_question_type
        )
        
        raw_acc = df['raw_is_correct'].mean()
        corrected_acc = df['contextcalib_is_correct'].mean()
        
        # Get content-free bias estimates
        cf_yes_prob = df['cf_yes_prob'].iloc[0] if len(df) > 0 else 0.0
        cf_no_prob = df['cf_no_prob'].iloc[0] if len(df) > 0 else 0.0
        
        cf_bias_estimate = {
            "cf_yes_prob": cf_yes_prob,
            "cf_no_prob": cf_no_prob
        }
        
    elif question_type == "mcq":  # mcq
        # Calculate TVD for raw predictions
        raw_tvd, raw_model_dist, raw_gt_dist = calculate_tvd(
            df['raw_predicted_answer'].tolist(),
            df['answer'].tolist(),
            question_type='mcq'
        )
        
        # Calculate TVD for corrected predictions
        corrected_tvd, corrected_model_dist, corrected_gt_dist = calculate_tvd(
            df['contextcalib_predicted_answer'].tolist(),
            df['answer'].tolist(),
            question_type='mcq'
        )
        
        raw_acc = df['raw_is_correct'].mean()
        corrected_acc = df['contextcalib_is_correct'].mean()
        
        # Get content-free bias estimates
        cf_oa_prob = df['cf_oa_prob'].iloc[0] if len(df) > 0 else 0.0
        cf_ob_prob = df['cf_ob_prob'].iloc[0] if len(df) > 0 else 0.0
        cf_oc_prob = df['cf_oc_prob'].iloc[0] if len(df) > 0 else 0.0
        cf_od_prob = df['cf_od_prob'].iloc[0] if len(df) > 0 else 0.0
        
        cf_bias_estimate = {
            "cf_oa_prob": cf_oa_prob,
            "cf_ob_prob": cf_ob_prob,
            "cf_oc_prob": cf_oc_prob,
            "cf_od_prob": cf_od_prob
        }

    elif question_type == 'nli':
        tvd_question_type = 'nli'
        
        # Calculate TVD for raw predictions
        raw_tvd, raw_model_dist, raw_gt_dist = calculate_tvd(
            df['raw_predicted_answer'].tolist(),
            df['Correct Answer'].tolist(),
            question_type=tvd_question_type
        )
        
        # Calculate TVD for corrected predictions
        corrected_tvd, corrected_model_dist, corrected_gt_dist = calculate_tvd(
            df['contextcalib_predicted_answer'].tolist(),
            df['Correct Answer'].tolist(),
            question_type=tvd_question_type
        )
        
        raw_acc = df['raw_is_correct'].mean()
        corrected_acc = df['contextcalib_is_correct'].mean()
        
        # Get content-free bias estimates
        cf_o0_prob = df['cf_o0_prob'].iloc[0] if len(df) > 0 else 0.0
        cf_o1_prob = df['cf_o1_prob'].iloc[0] if len(df) > 0 else 0.0
        cf_o2_prob = df['cf_o2_prob'].iloc[0] if len(df) > 0 else 0.0

        cf_bias_estimate = {
            "cf_o0_prob": cf_o0_prob,
            "cf_o1_prob": cf_o1_prob,
            "cf_o2_prob": cf_o2_prob
        }
    
    return {
        "raw_tvd": raw_tvd,
        "raw_model_dist": raw_model_dist,
        "raw_acc": raw_acc,
        "corrected_tvd": corrected_tvd,
        "corrected_model_dist": corrected_model_dist,
        "corrected_acc": corrected_acc,
        "cf_bias_estimate": cf_bias_estimate,
        "ground_truth_dist": raw_gt_dist
    }

def get_dataset_info(dataset: str, aggregation_level: str) -> Tuple[str, List[str]]:
    """Get dataset and domain information based on aggregation level"""
    
    if aggregation_level == 'all':
        if dataset == 'YESNO':
            return 'ALL_YESNO', ['EWOK', 'COMPS', 'BABI', 'ARITH']
        elif dataset == 'MMLU':
            return 'ALL_MMLU', ['MMLU-STEM', 'MMLU-HUMANITIES', 'MMLU-SOCIAL_SCI', 'MMLU-OTHERS']
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

def aggregate_across_datasets(question_type: str, prompt: str, model_family: str, 
                             datasets: List[str], date: str = "Sep-16-2025") -> Dict[str, Dict[str, Union[float, Dict]]]:
    """Aggregate metrics across multiple datasets for 'all' aggregation level"""
    model_configs = get_model_configs()
    aggregated_results = {}
    
    for model_name in model_configs[model_family]:
        all_dfs = []
        
        # Collect data from all datasets
        for dataset in datasets:
            _, domains = get_dataset_info(dataset, 'per')
            domain = domains[0]  # Each dataset has one domain in this context
            
            csv_path = get_contextcalib_csv_path(date, prompt, dataset, model_family, domain, model_name, question_type)
            
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                all_dfs.append(df)
        
        if all_dfs:
            # Combine all dataframes
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
                    combined_df['contextcalib_predicted_answer'].tolist(),
                    combined_df['Correct Answer'].tolist(),
                    question_type=tvd_question_type
                )
                
                raw_acc = combined_df['raw_is_correct'].mean()
                corrected_acc = combined_df['contextcalib_is_correct'].mean()
                
                # Content-free probabilities should be the same across all datasets for a given model
                cf_yes_prob = combined_df['cf_yes_prob'].iloc[0] if len(combined_df) > 0 else 0.0
                cf_no_prob = combined_df['cf_no_prob'].iloc[0] if len(combined_df) > 0 else 0.0
                
                cf_bias_estimate = {
                    "cf_yes_prob": cf_yes_prob,
                    "cf_no_prob": cf_no_prob
                }
                
            elif question_type == 'mcq':  # mcq
                question_type = 'mcq'
                # Calculate TVD for raw predictions
                raw_tvd, raw_model_dist, raw_gt_dist = calculate_tvd(
                    combined_df['raw_predicted_answer'].tolist(),
                    combined_df['answer'].tolist(),
                    question_type=question_type
                )
                
                # Calculate TVD for corrected predictions
                corrected_tvd, corrected_model_dist, corrected_gt_dist = calculate_tvd(
                    combined_df['contextcalib_predicted_answer'].tolist(),
                    combined_df['answer'].tolist(),
                    question_type=question_type
                )
                
                raw_acc = combined_df['raw_is_correct'].mean()
                corrected_acc = combined_df['contextcalib_is_correct'].mean()
                
                # Content-free probabilities should be the same across all datasets for a given model
                cf_oa_prob = combined_df['cf_oa_prob'].iloc[0] if len(combined_df) > 0 else 0.0
                cf_ob_prob = combined_df['cf_ob_prob'].iloc[0] if len(combined_df) > 0 else 0.0
                cf_oc_prob = combined_df['cf_oc_prob'].iloc[0] if len(combined_df) > 0 else 0.0
                cf_od_prob = combined_df['cf_od_prob'].iloc[0] if len(combined_df) > 0 else 0.0
                
                cf_bias_estimate = {
                    "cf_oa_prob": cf_oa_prob,
                    "cf_ob_prob": cf_ob_prob,
                    "cf_oc_prob": cf_oc_prob,
                    "cf_od_prob": cf_od_prob
                }

            elif question_type == 'nli':
                tvd_question_type = 'nli'
                
                # Calculate TVD for raw predictions
                raw_tvd, raw_model_dist, raw_gt_dist = calculate_tvd(
                    combined_df['raw_predicted_answer'].tolist(),
                    combined_df['Correct Answer'].tolist(),
                    question_type=tvd_question_type
                )
                
                # Calculate TVD for corrected predictions
                corrected_tvd, corrected_model_dist, corrected_gt_dist = calculate_tvd(
                    combined_df['contextcalib_predicted_answer'].tolist(),
                    combined_df['Correct Answer'].tolist(),
                    question_type=tvd_question_type
                )
                
                raw_acc = combined_df['raw_is_correct'].mean()
                corrected_acc = combined_df['contextcalib_is_correct'].mean()
                
                # Content-free probabilities should be the same across all datasets for a given model
                cf_o0_prob = combined_df['cf_o0_prob'].iloc[0] if len(combined_df) > 0 else 0.0
                cf_o1_prob = combined_df['cf_o1_prob'].iloc[0] if len(combined_df) > 0 else 0.0
                cf_o2_prob = combined_df['cf_o2_prob'].iloc[0] if len(combined_df) > 0 else 0.0

                cf_bias_estimate = {
                    "cf_o0_prob": cf_o0_prob,
                    "cf_o1_prob": cf_o1_prob,
                    "cf_o2_prob": cf_o2_prob
                }
                
            aggregated_results[model_name] = {
                "raw_tvd": raw_tvd,
                "raw_model_dist": str(raw_model_dist),
                "raw_acc": raw_acc,
                "corrected_tvd": corrected_tvd,
                "corrected_model_dist": str(corrected_model_dist),
                "corrected_acc": corrected_acc,
                "cf_bias_estimate": cf_bias_estimate,
                "ground_truth_dist": str(raw_gt_dist)
            }
    
    return aggregated_results

def main():
    parser = argparse.ArgumentParser(description='Process contextual calibration metrics')
    parser.add_argument('--question_type', choices=['yesno', 'mcq', 'nli'], required=True,
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
                      help='Dataset to process (e.g., BABI, MMLU-HUMANITIES, SNLI, YESNO, MMLU)')
    parser.add_argument('--date', default='Sep-16-2025',
                      help='Date folder (default: Sep-16-2025)')
    
    args = parser.parse_args()
    
    # Set defaults if not provided - contextual calibration only works with instronly and fewshot
    model_families = [args.model_family] if args.model_family else list(get_model_configs().keys())
    prompt_types = [args.prompt_type] if args.prompt_type else ['zeroshot', 'instronly', 'fewshot']
    
    try:
        dataset_name, domains = get_dataset_info(args.dataset, args.aggregation_level)
        
        # Process each combination of prompt_type and model_family
        for prompt_type in prompt_types:
            for model_family in model_families:
                print(f"\nProcessing {prompt_type} - {model_family}")
                
                model_configs = get_model_configs()
                
                # Create output directory
                output_dir = Path(__file__).parent.parent / "results" / f"contextcalib_{args.question_type}_{args.aggregation_level}_TVD"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Initialize results structure
                results = {
                    prompt_type: {
                        dataset_name: {}
                    }
                }
                
                if args.aggregation_level == 'all':
                    # Handle aggregated case
                    if args.question_type == 'yesno':
                        datasets_to_aggregate = ['EWOK', 'COMPS', 'BABI', 'ARITH']
                        domain_key = 'all_yesno_domains'
                    elif args.question_type == 'mcq':  # mcq
                        datasets_to_aggregate = ['MMLU-STEM', 'MMLU-HUMANITIES', 'MMLU-SOCIAL_SCI', 'MMLU-OTHERS']
                        domain_key = 'all_mcq_domains'
                    elif args.question_type == 'nli':  # nli
                        datasets_to_aggregate = ['SNLI', 'MNLI']
                        domain_key = 'all_nli_domains'

                    aggregated_results = aggregate_across_datasets(
                        args.question_type, prompt_type, model_family, 
                        datasets_to_aggregate, args.date
                    )
                    
                    results[prompt_type][dataset_name][domain_key] = {
                        model_family: aggregated_results
                    }
                    
                else:
                    # Handle per-dataset case
                    for domain in domains:
                        results[prompt_type][dataset_name][domain] = {
                            model_family: {}
                        }
                        
                        for model_name in model_configs[model_family]:
                            csv_path = get_contextcalib_csv_path(args.date, prompt_type, args.dataset, 
                                                      model_family, domain, model_name, args.question_type)
                            
                            if csv_path.exists():
                                try:
                                    metrics = process_single_model(csv_path, args.question_type)
                                    results[prompt_type][dataset_name][domain][model_family][model_name] = metrics
                                    print(f"  Processed: {model_name}")
                                except Exception as e:
                                    print(f"  Error processing {model_name}: {e}")
                            else:
                                print(f"  File not found: {csv_path}")
                
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