#!/usr/bin/env python3
"""
BOS Metrics Processor - Processes BOS correction results and generates JSON metrics files.

This script reads BOS correction CSV files and calculates bias and accuracy metrics,
comparing raw (uncorrected) vs BOS-corrected performance.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
import sys

# Add shared_utils to path when it exists
# from shared_utils import calculate_bias_metric, get_csv_paths, write_json_results

def calculate_yesno_bias(df: pd.DataFrame, prediction_col: str = 'predicted_answer') -> float:
    """Calculate yes-no bias metric: (num_yes - num_no) / total"""
    # Handle both string ('Yes'/'No') and boolean (True/False) formats
    if df[prediction_col].dtype == 'bool':
        yes_count = df[prediction_col].sum()  # True values
        no_count = (~df[prediction_col]).sum()  # False values
    else:
        yes_count = (df[prediction_col] == 'Yes').sum()
        no_count = (df[prediction_col] == 'No').sum()
    
    total = len(df)
    return (yes_count - no_count) / total

def calculate_rstd(df: pd.DataFrame, prediction_col: str = 'predicted_answer') -> float:
    """Calculate Recall Standard Deviation for MCQ position bias"""
    recalls = []
    for option in ['A', 'B', 'C', 'D']:
        # Get questions where this option is correct
        correct_mask = (df['answer'] == option)
        if correct_mask.sum() == 0:
            recalls.append(0.0)
            continue
        
        # Calculate recall for this position
        predicted_correct = (df[prediction_col] == option) & correct_mask
        recall = predicted_correct.sum() / correct_mask.sum()
        recalls.append(recall)
    
    return np.std(recalls)

def get_bos_csv_path(date: str, prompt: str, dataset: str, model_family: str, 
                     domain: str, model_name: str, question_type: str) -> Path:
    """Construct path to BOS CSV file"""
    method = f"{question_type}bos"
    # Assume script is in scripts/ folder, so go up one level to reach outputs/
    base_path = Path(__file__).parent.parent / "outputs" / date / prompt / dataset / method / model_family / domain
    return base_path / f"{model_name}_results.csv"

def process_single_model(csv_path: Path, question_type: str) -> Dict[str, float]:
    """Process a single BOS CSV file and extract metrics"""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if question_type == 'yesno':
        # Yes-No metrics
        raw_bias = calculate_yesno_bias(df, 'raw_predicted_answer')
        corrected_bias = calculate_yesno_bias(df, 'bos_predicted_answer')
        raw_acc = df['raw_is_correct'].mean()
        corrected_acc = df['bos_is_correct'].mean()
    else:  # mcq
        # MCQ metrics
        raw_bias = calculate_rstd(df, 'plain_predicted_answer')
        corrected_bias = calculate_rstd(df, 'bos_predicted_answer')
        raw_acc = df['plain_is_correct'].mean()
        corrected_acc = df['bos_is_correct'].mean()
    
    return {
        "raw_bias": raw_bias,
        "raw_acc": raw_acc,
        "corrected_bias": corrected_bias,
        "corrected_acc": corrected_acc
    }

def get_dataset_info(dataset: str, aggregation_level: str) -> Tuple[str, List[str]]:
    """Get dataset and domain information based on aggregation level"""
    
    if aggregation_level == 'all':
        if dataset == 'YESNO':
            return 'ALL_YESNO', ['EWOK', 'COMPS', 'BABI', 'ARITH']
        elif dataset == 'MMLU':
            return 'ALL_MMLU', ['MMLU-STEM', 'MMLU-HUMANITIES', 'MMLU-SOCIAL_SCI', 'MMLU-OTHERS']
    else:  # per
        if dataset in ['EWOK', 'COMPS', 'BABI', 'ARITH']:
            domain_map = {
                'EWOK': 'all_domains',
                'COMPS': 'comps', 
                'BABI': 'babi',
                'ARITH': 'arith'
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
        "Qwen2": ["Qwen2.5-14B", "Qwen2.5-14B-Instruct", "Qwen2.5-32B", "Qwen2.5-32B-Instruct"],
        "Llama": ["Llama-2-7b-hf", "Llama-2-7b-chat-hf", "Llama-2-13b-hf", "Llama-2-13b-chat-hf"],
        "Llama3": ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"]
    }

def aggregate_across_datasets(question_type: str, prompt: str, model_family: str, 
                             datasets: List[str], date: str = "Sep-16-2025") -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across multiple datasets for 'all' aggregation level"""
    model_configs = get_model_configs()
    aggregated_results = {}
    
    for model_name in model_configs[model_family]:
        all_dfs = []
        
        # Collect data from all datasets
        for dataset in datasets:
            _, domains = get_dataset_info(dataset, 'per')
            domain = domains[0]  # Each dataset has one domain in this context
            
            csv_path = get_bos_csv_path(date, prompt, dataset, model_family, domain, model_name, question_type)
            
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                all_dfs.append(df)
        
        if all_dfs:
            # Combine all dataframes
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # Calculate metrics on combined data
            if question_type == 'yesno':
                raw_bias = calculate_yesno_bias(combined_df, 'raw_predicted_answer')
                corrected_bias = calculate_yesno_bias(combined_df, 'bos_predicted_answer')
                raw_acc = combined_df['raw_is_correct'].mean()
                corrected_acc = combined_df['bos_is_correct'].mean()
            else:  # mcq
                raw_bias = calculate_rstd(combined_df, 'plain_predicted_answer')
                corrected_bias = calculate_rstd(combined_df, 'bos_predicted_answer')
                raw_acc = combined_df['plain_is_correct'].mean()
                corrected_acc = combined_df['bos_is_correct'].mean()
            
            aggregated_results[model_name] = {
                "raw_bias": raw_bias,
                "raw_acc": raw_acc,
                "corrected_bias": corrected_bias,
                "corrected_acc": corrected_acc
            }
    
    return aggregated_results

def main():
    parser = argparse.ArgumentParser(description='Process BOS correction metrics')
    parser.add_argument('--question_type', choices=['yesno', 'mcq'], required=True,
                      help='Question type: yesno (yes-no) or mcq (multiple choice)')
    parser.add_argument('--aggregation_level', choices=['per', 'all'], required=True,
                      help='Aggregation level: per (individual datasets) or all (combined)')
    parser.add_argument('--model_family', 
                      choices=['Falcon', 'Gemma3', 'Qwen2', 'Llama', 'Llama3'],
                      help='Model family to process (default: all)')
    parser.add_argument('--prompt_type', 
                      choices=['zeroshot', 'instronly', 'fewshot'],
                      help='Prompt type (default: all)')
    parser.add_argument('--dataset', required=True,
                      help='Dataset to process (e.g., BABI, MMLU-HUMANITIES, YESNO, MMLU)')
    parser.add_argument('--date', default='Sep-16-2025',
                      help='Date folder (default: Sep-16-2025)')
    
    args = parser.parse_args()
    
    # Set defaults if not provided
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
                output_dir = Path(__file__).parent.parent / "results" / f"bos_{args.question_type}_{args.aggregation_level}"
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
                    else:  # mcq
                        datasets_to_aggregate = ['MMLU-STEM', 'MMLU-HUMANITIES', 'MMLU-SOCIAL_SCI', 'MMLU-OTHERS']
                        domain_key = 'all_mcq_domains'
                    
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
                            csv_path = get_bos_csv_path(args.date, prompt_type, args.dataset, 
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