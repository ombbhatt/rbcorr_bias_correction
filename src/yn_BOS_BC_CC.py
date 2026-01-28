# this is yn_BOS_BC_CC.py

import torch, pandas as pd
from tqdm import tqdm
import numpy as np
import os
import gc
from pathlib import Path
from get_query_logprobs import calculate_logprobs_batch_yesno
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, LlamaForCausalLM, BitsAndBytesConfig, Gemma3ForConditionalGeneration


falcon_models = ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"]
gemma3_models = ["gemma-3-27b-pt", "gemma-3-27b-it", "gemma-3-12b-pt", "gemma-3-12b-it"]
# qwen2_models = ["Qwen2.5-14B", "Qwen2.5-14B-Instruct", "Qwen2.5-32B", "Qwen2.5-32B-Instruct"]
# llama_models = ["Llama-2-7b-hf", "Llama-2-7b-chat-hf", "Llama-2-13b-hf", "Llama-2-13b-chat-hf"]
llama3_models = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"]

MODEL_CONFIGS = {
    "Falcon": {"models": falcon_models, "model_class": AutoModelForCausalLM, "tokenizer_class": AutoTokenizer, "prefix": "tiiuae/"},
    # "Qwen2": {"models": qwen2_models, "model_class": AutoModelForCausalLM, "tokenizer_class": AutoTokenizer, "prefix": "Qwen/"},
    # "Llama": {"models": llama_models, "model_class": LlamaForCausalLM, "tokenizer_class": AutoTokenizer, "prefix": "meta-llama/"},
    "Llama3": {"models": llama3_models, "model_class": LlamaForCausalLM, "tokenizer_class": AutoTokenizer, "prefix": "meta-llama/"},
    "Gemma3": {"models": gemma3_models, "model_class": Gemma3ForConditionalGeneration, "tokenizer_class": AutoTokenizer, "prefix": "google/"},   
}

def process_dataset_yesno(input_file, output_file, plain_file, impl, model_name, model, model_family, tokenizer, domain, calib_count=80, batch_size=8, dataset=None, prompt=None, calib_data=None, batch_size_param=None):

    print("impl: ", impl)

    # For correction implementations, check if plain results exist first
    if impl in ["yesnospecific", "yesnobos", "yesnocontextcalib", "yesnobatchcalib"]:
        # Get the model_family for path reconstruction
        yesnoplain_output_file = plain_file
        
        # If plain results exist, use them (regardless of whether model is loaded)
        if os.path.exists(yesnoplain_output_file):
            print(f"Found existing plain results: {yesnoplain_output_file}")
            df = pd.read_csv(yesnoplain_output_file)
            
            # Apply the requested correction
            if impl == "yesnospecific":
                print("Applying specific correction to existing plain results...")
                df = do_entire_specific_thing(df, dataset, calib_count, calib_data)
                df.to_csv(output_file, index=False)
                print(f"specific corrected results saved to {output_file}")
                return
            
            elif impl == "yesnobos":
                print("Applying BOS correction to existing plain results...")
                bos_df = pd.read_csv('../data/bos_logprobs_yesno.csv')
                row = bos_df[bos_df['model_name'] == model_name]
                bos_yes = row['bos_yes_logprob'].iloc[0]
                bos_no = row['bos_no_logprob'].iloc[0]
                df = do_entire_bos_thing(df, bos_yes, bos_no)
                df.to_csv(output_file, index=False)
                print(f"BOS corrected results saved to {output_file}")
                return
            
            elif impl == "yesnocontextcalib":
                print("Applying contextual calibration to existing plain results...")
                df = do_entire_contextcalib_thing(df, model, tokenizer, dataset, prompt, model_name)
                df.to_csv(output_file, index=False)
                print(f"Contextual calibration results saved to {output_file}")
                return

            elif impl == "yesnobatchcalib":
                print("Applying batch calibration to fresh inference results...")
                df = do_entire_batchcalib_thing(df, batch_size=batch_size_param)
                df.to_csv(output_file, index=False)
                print(f"Batch calibration results saved to {output_file}")
                return

        
        else:
            # Plain results don't exist
            if model is None and tokenizer is None:
                print(f"ERROR: Expected plain results at {yesnoplain_output_file} but file doesn't exist!")
                print("Cannot run correction-only mode without existing plain results.")
                return
            else:
                print(f"Plain results don't exist at {yesnoplain_output_file}")
                print("Will run full inference first, then apply corrections.")
                # Continue to inference section below

    # INFERENCE MODE - only reached if:
    # 1. impl is "yesnoplain", OR  
    # 2. impl is correction mode but plain results don't exist
    
    if model is None and tokenizer is None:
        print("ERROR: Cannot run inference without model and tokenizer!")
        return
    
    print(f"Running inference for {model_name}")
    
    # Load and process data
    if isinstance(input_file, pd.DataFrame):
        df = input_file
    else:
        df = pd.read_csv(input_file, encoding='utf8')
    print(f"Starting processing domain: {domain} for model: {model_name}")
    
    # Initialize columns
    df['yes_logprob'] = None
    df['no_logprob'] = None
    df['predicted_answer'] = None
    df['is_correct'] = None
    
    # Run inference
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i + batch_size]
        batch_results = calculate_logprobs_batch_yesno(
            batch_df['Context'].tolist(),
            batch_df['Question'].tolist(),
            tokenizer, model, dataset, prompt, content_free_mode=False, content_free_input=None
        )

        for j, result in enumerate(batch_results):
            if i + j >= len(df): break
            idx = i + j
            
            for key, value in result.items():
                df.loc[idx, key] = value

            df.loc[idx, 'predicted_answer'] = "Yes" if df.loc[idx, 'yes_logprob'] > df.loc[idx, 'no_logprob'] else "No"
            df.loc[idx, 'is_correct'] = True if df.loc[idx, 'predicted_answer'] == df.loc[idx, 'Correct Answer'] else False
        
        if i % (batch_size * 5) == 0:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if i % (batch_size * 20) == 0:
                gc.collect()

    # Save plain results (for future correction-only runs)
    if impl in ["yesnospecific", "yesnobos", "yesnocontextcalib", "yesnobatchcalib"]:
        # Reconstruct plain path
        yesnoplain_output_file = plain_file
        plain_output_file = Path(yesnoplain_output_file)
        plain_output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(yesnoplain_output_file, index=False)
        print(f"Plain results saved to {yesnoplain_output_file}")

    # Apply corrections if needed
    if impl == "yesnospecific":
        print("Applying specific correction to fresh inference results...")
        df = do_entire_specific_thing(df, dataset, calib_count, calib_data)
        df.to_csv(output_file, index=False)
        print(f"specific corrected results saved to {output_file}")
        return

    elif impl == "yesnobos":
        print("Applying BOS correction to existing plain results...")
        bos_df = pd.read_csv('../data/bos_logprobs_yesno.csv')
        
        # Handle model name matching more robustly
        if model_name in bos_df['model_name'].values:
            row = bos_df[bos_df['model_name'] == model_name]
        else:
            # Try with prefix from model family
            full_model_name = f"{MODEL_CONFIGS[model_family]['prefix']}{model_name}"
            if full_model_name in bos_df['model_name'].values:
                row = bos_df[bos_df['model_name'] == full_model_name]
            else:
                print(f"ERROR: Model {model_name} not found in BOS data")
                return
        
        bos_yes = row['bos_yes_logprob'].iloc[0]
        bos_no = row['bos_no_logprob'].iloc[0]
        df = do_entire_bos_thing(df, bos_yes, bos_no)
        df.to_csv(output_file, index=False)
        print(f"BOS corrected results saved to {output_file}")
        return
    
    elif impl == "yesnocontextcalib":
        print("Applying contextual calibration to fresh inference results...")
        df = do_entire_contextcalib_thing(df, model, tokenizer, dataset, prompt, model_name)
        df.to_csv(output_file, index=False)
        print(f"Contextual calibration results saved to {output_file}")
        return

    elif impl == "yesnobatchcalib":
        print("Applying batch calibration to fresh inference results...")
        df = do_entire_batchcalib_thing(df)
        df.to_csv(output_file, index=False)
        print(f"Batch calibration results saved to {output_file}")
        return
    
    # For plain implementation, save directly
    elif impl == "yesnoplain":
        df.to_csv(output_file, index=False)
        print(f"Plain results saved to {output_file}")
        return


def do_entire_bos_thing(plain_df, bos_yes_val, bos_no_val):
    """Apply BOS (Beginning of Sequence) correction to Yes/No logprobs"""
    
    total_questions = len(plain_df)
    new_df = _initialize_dataframe(plain_df, "bos")
    
    print(f"Applying BOS correction to {total_questions} questions...")

    # Store BOS values
    new_df['bos_yes_val'] = bos_yes_val
    new_df['bos_no_val'] = bos_no_val

    # Calculate corrected logprobs
    new_df['corrected_yes_logprob'] = new_df['raw_yes_logprob'] - bos_yes_val
    new_df['corrected_no_logprob'] = new_df['raw_no_logprob'] - bos_no_val
    
    # Calculate BOS predictions
    new_df['bos_predicted_answer'] = new_df['corrected_yes_logprob'] > new_df['corrected_no_logprob']
    new_df['bos_is_correct'] = new_df['bos_predicted_answer'] == (new_df['Correct Answer'] == "Yes")

    print("BOS correction completed successfully.")
    return new_df


def _initialize_dataframe(df, impl):
    """Initialize dataframe with list columns for specific correction"""
    new_df = df.copy()

    if impl == "bos":

        list_columns = [
            'raw_yes_logprob', 'raw_no_logprob', 'corrected_yes_logprob', 
            'corrected_no_logprob', 'bos_yes_val', 'bos_no_val', 'raw_predicted_answer', 
            'raw_is_correct', 'bos_predicted_answer', 'bos_is_correct'
        ]

        # Initialize columns
        for col in list_columns:
            new_df[col] = None

        # Store raw values and plain results
        new_df['raw_yes_logprob'] = new_df['yes_logprob']
        new_df['raw_no_logprob'] = new_df['no_logprob']
        new_df['raw_predicted_answer'] = new_df['predicted_answer']
        new_df['raw_is_correct'] = new_df['is_correct']
        
        new_df.drop(columns=['yes_logprob', 'no_logprob', 'predicted_answer', 'is_correct'], inplace=True)

    elif impl == "specific":
    
        # Define all columns that will store lists of results
        list_columns = [
            'eval_split_ratio', 'calib_set_ID', 'raw_yes_logprob', 'raw_no_logprob', 
            'corrected_yes_logprob', 'corrected_no_logprob', 'calib_yes_mean', 'calib_no_mean',
            'raw_predicted_answer', 'raw_is_correct', 'specific_predicted_answer', 'specific_is_correct'
        ]
        
        # Initialize all columns as lists
        for col in list_columns:
            new_df[col] = [[] for _ in range(len(new_df))]

        # Store raw values as single-item lists
        new_df['raw_yes_logprob'] = new_df['yes_logprob'].apply(lambda x: [x])
        new_df['raw_no_logprob'] = new_df['no_logprob'].apply(lambda x: [x])
        new_df['raw_predicted_answer'] = new_df['predicted_answer'].apply(lambda x: [x])
        new_df['raw_is_correct'] = new_df['is_correct'].apply(lambda x: [x])
        
        # Remove original columns
        new_df.drop(columns=['yes_logprob', 'no_logprob', 'predicted_answer', 'is_correct'], inplace=True)

    elif impl == "contextcalib":
        contextcalib_columns = [
            'raw_yes_logprob', 'raw_no_logprob', 'calibrated_yes_logprob', 'calibrated_no_logprob',
            'cf_yes_prob', 'cf_no_prob', 'raw_predicted_answer', 'raw_is_correct', 
            'contextcalib_predicted_answer', 'contextcalib_is_correct'
        ]
        
        # Initialize columns
        for col in contextcalib_columns:
            new_df[col] = None
        
        # Store raw values
        new_df['raw_yes_logprob'] = new_df['yes_logprob']
        new_df['raw_no_logprob'] = new_df['no_logprob']
        new_df['raw_predicted_answer'] = new_df['predicted_answer']
        new_df['raw_is_correct'] = new_df['is_correct']
        
        # Remove original columns
        new_df.drop(columns=['yes_logprob', 'no_logprob', 'predicted_answer', 'is_correct'], inplace=True)

    elif impl == "batchcalib":
        batchcalib_columns = [
            'raw_yes_logprob', 'raw_no_logprob', 'batch_yes_mean', 'batch_no_mean',
            'calibrated_yes_logprob', 'calibrated_no_logprob',
            'raw_predicted_answer', 'raw_is_correct', 
            'batchcalib_predicted_answer', 'batchcalib_is_correct'
        ]
        
        # Initialize columns
        for col in batchcalib_columns:
            new_df[col] = None
        
        # Store raw values
        new_df['raw_yes_logprob'] = new_df['yes_logprob']
        new_df['raw_no_logprob'] = new_df['no_logprob']
        new_df['raw_predicted_answer'] = new_df['predicted_answer']
        new_df['raw_is_correct'] = new_df['is_correct']
        
        # Remove original columns
        new_df.drop(columns=['yes_logprob', 'no_logprob', 'predicted_answer', 'is_correct'], inplace=True)
        
    return new_df


def _get_calibration_indices(dataset_name, calib_count, total_questions, calib_run, eval_df, calib_df=None):
    """Generate calibration and evaluation indices for a given run with class balancing
    
    Args:
        dataset_name: Name of the target dataset (for EWOK grouping logic)
        calib_count: Number of calibration samples to draw (0 means use entire dataset)
        total_questions: Total questions in target (evaluation) dataset
        calib_run: Random seed for this calibration run
        eval_df: Target dataset dataframe
        calib_df: External calibration dataframe (None for same-dataset, DataFrame for cross-dataset/model)
    """
    np.random.seed(calib_run)  # For reproducibility
    
    # Use external calibration data if provided (cross-dataset OR cross-model)
    calibration_source = calib_df if calib_df is not None else eval_df
    calib_total = len(calibration_source)
    
    if dataset_name == "EWOK" and calib_df is None:
        # Handle grouped questions (groups of 4) - only for same-dataset EWOK
        group_starts = list(range(0, total_questions - 3, 4))
        
        # Since each group contributes 2 Yes + 2 No, we need calib_count/4 groups total
        calib_groups_needed = max(1, round(calib_count / 4))
        
        # Sample groups with replacement
        selected_groups = np.random.choice(group_starts, size=calib_groups_needed, replace=True)
        
        # Expand to individual indices
        calib_indices = []
        for group_start in selected_groups:
            calib_indices.extend(range(group_start, min(group_start + 4, total_questions)))
            
    else:
        # Handle individual questions (COMPS, BABI, ARITH, cross-dataset, cross-model) with class balancing
        
        # Separate questions by class from calibration source
        yes_indices = calibration_source[calibration_source['Correct Answer'] == 'Yes'].index.tolist()
        no_indices = calibration_source[calibration_source['Correct Answer'] == 'No'].index.tolist()

        # print(f"  Calibration source: {len(yes_indices)} Yes indices + {len(no_indices)} No indices")
        
        # Calculate how many questions we need from each class
        questions_per_class = calib_count // 2
        
        # Handle odd calib_count - alternate which class gets the extra question
        if calib_count % 2 == 1:
            if calib_run % 2 == 0:
                yes_needed = questions_per_class + 1
                no_needed = questions_per_class
            else:
                yes_needed = questions_per_class
                no_needed = questions_per_class + 1
        else:
            yes_needed = questions_per_class
            no_needed = questions_per_class

        # print(f"  yes needed: {yes_needed}, no needed: {no_needed}")
        
        # Sample with replacement from each class
        selected_yes = np.random.choice(yes_indices, size=yes_needed, replace=True)
        selected_no = np.random.choice(no_indices, size=no_needed, replace=True)

        # print(f" selected yes: {len(selected_yes)}, selected no: {len(selected_no)}")
        
        # Combine and convert to list
        calib_indices = list(selected_yes) + list(selected_no)
    
    # Evaluation indices depend on whether we're using external calibration
    if calib_df is not None:
        # Cross-dataset OR cross-model mode: all target dataset questions are evaluated
        eval_indices = list(range(total_questions))
    else:
        # Same-dataset mode: exclude calibration questions
        eval_indices = [i for i in range(total_questions) if i not in set(calib_indices)]
    
    return calib_indices, eval_indices


def _apply_correction_to_eval_set(eval_df, eval_indices, calib_yes_mean, calib_no_mean, calib_run, total_questions, calib_df=None):
    """Apply bias correction to evaluation set questions"""
    eval_ratio = len(eval_indices) / total_questions
    
    for eval_idx in eval_indices:
        # Store calibration means
        eval_df.loc[eval_idx, 'calib_yes_mean'].append(calib_yes_mean)
        eval_df.loc[eval_idx, 'calib_no_mean'].append(calib_no_mean)
        
        # Calculate corrected logprobs
        raw_yes = eval_df.iloc[eval_idx]['raw_yes_logprob'][0]
        raw_no = eval_df.iloc[eval_idx]['raw_no_logprob'][0]
        
        corrected_yes = raw_yes - calib_yes_mean
        corrected_no = raw_no - calib_no_mean
        
        eval_df.loc[eval_idx, 'corrected_yes_logprob'].append(corrected_yes)
        eval_df.loc[eval_idx, 'corrected_no_logprob'].append(corrected_no)
        
        # Calculate predictions and accuracy
        specific_predicted = "Yes" if corrected_yes > corrected_no else "No"
        correct_answer = eval_df.iloc[eval_idx]['Correct Answer']
        specific_correct = (specific_predicted == correct_answer)
        
        eval_df.loc[eval_idx, 'specific_predicted_answer'].append(specific_predicted)
        eval_df.loc[eval_idx, 'specific_is_correct'].append(specific_correct)
        
        # Track metadata
        eval_df.loc[eval_idx, 'calib_set_ID'].append(calib_run + 1)
        eval_df.loc[eval_idx, 'eval_split_ratio'].append(eval_ratio)


def do_entire_specific_thing(plain_df, dataset_name, calib_count, calib_data=None):
    """Apply specific cross-validation bias correction to Yes/No logprobs
    
    Args:
        plain_df: Target dataset with plain inference results
        dataset_name: Name of target dataset (for EWOK grouping)
        calib_count: Number of calibration samples per run
        calib_data: External calibration data (for cross-dataset or cross-model correction)
                   If None, uses same-dataset correction
    """
    
    print(f"=== SPECIFIC CORRECTION START ===")
    print(f"Target Dataset: {dataset_name}, Calibration count: {calib_count} questions")
    
    # Check cross-dataset/model mode
    cross_transfer_mode = calib_data is not None
    if cross_transfer_mode:
        print(f"Cross-transfer mode: Using external calibration data with {len(calib_data)} samples")
        print(f"Note: External data could be from different dataset and/or different model")
    
    # Check if already processed
    if any(col in plain_df.columns for col in ['raw_yes_logprob', 'fold', 'corrected_yes_logprob']):
        print("❌ WARNING: Input DataFrame already contains specific columns!")
        return plain_df

    total_questions = len(plain_df)
    print(f"Processing {total_questions} target questions...")
    
    # Initialize dataframe with list columns
    new_df = _initialize_dataframe(plain_df, "specific")

    # Run 100 calibration rounds
    for calib_run in range(100):
        if calib_run % 20 == 0:  # Less verbose logging
            print(f"Calibration run {calib_run + 1}/100")
        
        # Get calibration and evaluation indices for this run
        calib_indices, eval_indices = _get_calibration_indices(
            dataset_name, calib_count, total_questions, calib_run, new_df, calib_data
        )

        # print(f"  Calibration set: {len(calib_indices)} questions")
        # print(f"  Evaluation set: {len(eval_indices)} questions")
        
        # Calculate calibration means from appropriate source
        if cross_transfer_mode:        
            calib_yes_mean = np.median([calib_data.iloc[i]['yes_logprob'] for i in calib_indices])
            calib_no_mean = np.median([calib_data.iloc[i]['no_logprob'] for i in calib_indices])
        else:
            calib_yes_mean = np.median([new_df.iloc[i]['raw_yes_logprob'][0] for i in calib_indices])
            calib_no_mean = np.median([new_df.iloc[i]['raw_no_logprob'][0] for i in calib_indices])
        
        # Apply correction to evaluation set
        _apply_correction_to_eval_set(
            new_df, eval_indices, calib_yes_mean, calib_no_mean, calib_run, total_questions, calib_data
        )

    # Summary statistics
    avg_evaluations = np.mean([len(new_df.iloc[i]['calib_set_ID']) for i in range(total_questions)])
    print("Specific correction completed successfully.")
    print(f"Average evaluations per question: {avg_evaluations:.1f}")
    
    return new_df


def do_entire_contextcalib_thing(plain_df, model, tokenizer, dataset, prompt, model_name):
    """Apply contextual calibration to Yes/No logprobs using pre-computed content-free probabilities"""
    
    print(f"=== CONTEXTUAL CALIBRATION START ===")
    total_questions = len(plain_df)
    print(f"Processing {total_questions} questions...")
    
    # Initialize new dataframe
    new_df = _initialize_dataframe(plain_df, "contextcalib")
    
    # Load pre-computed content-free probabilities from CSV
    print(f"Loading pre-computed content-free probabilities...")
    if "Falcon" in model_name:
        cf_df = pd.read_csv('../data/contextfree_probs_yesno_falcon.csv')
    else:
        cf_df = pd.read_csv('../data/contextfree_probs_yesno.csv')
    
    # Find the row matching this model, dataset, and prompt
    matching_rows = cf_df[
        (cf_df['model_name'] == model_name) & 
        (cf_df['dataset'] == dataset) & 
        (cf_df['prompt'] == prompt)
    ]
    
    if len(matching_rows) == 0:
        raise ValueError(
            f"No pre-computed content-free probabilities found for:\n"
            f"  Model: {model_name}\n"
            f"  Dataset: {dataset}\n"
            f"  Prompt: {prompt}\n"
            f"Please run contextfree_probs_getter_yesno.py first to generate these values."
        )
    
    # Extract the content-free probabilities
    p_cf_yes = matching_rows['cf_yes_prob'].iloc[0]
    p_cf_no = matching_rows['cf_no_prob'].iloc[0]
    
    print(f"Loaded content-free probabilities: Yes={p_cf_yes:.4f}, No={p_cf_no:.4f}")
    
    # Apply calibration to all questions
    for idx in range(len(new_df)):
        # Get raw probabilities
        raw_yes_logprob = new_df.iloc[idx]['raw_yes_logprob']
        raw_no_logprob = new_df.iloc[idx]['raw_no_logprob']
        
        # Convert to probabilities and normalize
        raw_yes_prob = np.exp(raw_yes_logprob)
        raw_no_prob = np.exp(raw_no_logprob)
        raw_total = raw_yes_prob + raw_no_prob
        raw_yes_prob = raw_yes_prob / raw_total
        raw_no_prob = raw_no_prob / raw_total
        
        # Apply contextual calibration: W = diag(p_cf)^(-1)
        calibrated_yes = raw_yes_prob / p_cf_yes
        calibrated_no = raw_no_prob / p_cf_no
        
        # Renormalize
        calib_total = calibrated_yes + calibrated_no
        calibrated_yes = calibrated_yes / calib_total
        calibrated_no = calibrated_no / calib_total
        
        # Convert back to logprobs and store
        new_df.loc[idx, 'calibrated_yes_logprob'] = np.log(calibrated_yes)
        new_df.loc[idx, 'calibrated_no_logprob'] = np.log(calibrated_no)
        
        # Store content-free probabilities for reference
        new_df.loc[idx, 'cf_yes_prob'] = p_cf_yes
        new_df.loc[idx, 'cf_no_prob'] = p_cf_no
        
        # Make predictions
        contextcalib_predicted = "Yes" if calibrated_yes > calibrated_no else "No"
        new_df.loc[idx, 'contextcalib_predicted_answer'] = contextcalib_predicted
        new_df.loc[idx, 'contextcalib_is_correct'] = contextcalib_predicted == new_df.iloc[idx]['Correct Answer']

    print("Contextual calibration completed successfully.")
    return new_df

def do_entire_batchcalib_thing(plain_df, batch_size=None):
    """Apply batch calibration with running mean correction
    
    Args:
        plain_df: DataFrame with plain inference results
        batch_size: Size of each batch. If None, uses entire dataset.
    """
    
    print(f"=== BATCH CALIBRATION START ===")
    total_questions = len(plain_df)
    
    # If no batch size specified, use entire dataset (original paper's approach)
    if batch_size == 0:
        batch_size = total_questions
    
    print(f"Processing {total_questions} questions with batch_size={batch_size}")
    
    # Initialize new dataframe
    new_df = _initialize_dataframe(plain_df, "batchcalib")
    
    # Shuffle dataset to avoid systematic ordering bias (e.g., all Yes first, then No)
    shuffled_indices = np.random.RandomState(seed=42).permutation(len(new_df))
    new_df = new_df.iloc[shuffled_indices].reset_index(drop=True)
    print(f"Dataset shuffled with fixed seed for reproducibility")

    # Process dataset in batches
    num_batches = (total_questions + batch_size - 1) // batch_size  # Ceiling division
    
    # Initialize running means (will be calculated from first batch)
    running_yes_mean = None
    running_no_mean = None
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_questions)
        
        # Get batch
        batch_indices = range(start_idx, end_idx)
        
        # Calculate current batch means
        current_batch_yes_mean = new_df.iloc[start_idx:end_idx]['raw_yes_logprob'].mean()
        current_batch_no_mean = new_df.iloc[start_idx:end_idx]['raw_no_logprob'].mean()
        
        # Update running means using equation (3)
        if batch_idx == 0:
            # First batch: initialize running mean
            running_yes_mean = current_batch_yes_mean
            running_no_mean = current_batch_no_mean
        else:
            # Subsequent batches: update running mean
            # p̂ᵣⁿ⁺¹(y|C) = (n/(n+1))p̂ᵣⁿ(y|C) + (1/(n+1))p̂ⁿ⁺¹(y|C)
            n = batch_idx  # number of batches seen so far (before this one)
            running_yes_mean = (n / (n + 1)) * running_yes_mean + (1 / (n + 1)) * current_batch_yes_mean
            running_no_mean = (n / (n + 1)) * running_no_mean + (1 / (n + 1)) * current_batch_no_mean
        
        print(f"  Batch {batch_idx + 1}/{num_batches} (questions {start_idx}-{end_idx-1}): " +
              f"Current_Yes={current_batch_yes_mean:.4f}, Current_No={current_batch_no_mean:.4f}, " +
              f"Running_Yes={running_yes_mean:.4f}, Running_No={running_no_mean:.4f}")
        
        # Apply calibration to questions in this batch using the running mean
        for idx in batch_indices:
            # Store running means for reference
            new_df.loc[idx, 'batch_yes_mean'] = running_yes_mean
            new_df.loc[idx, 'batch_no_mean'] = running_no_mean
            
            # Calculate corrected logprobs using running mean
            new_df.loc[idx, 'calibrated_yes_logprob'] = new_df.iloc[idx]['raw_yes_logprob'] - running_yes_mean
            new_df.loc[idx, 'calibrated_no_logprob'] = new_df.iloc[idx]['raw_no_logprob'] - running_no_mean
            
            # Make predictions
            calibrated_yes = new_df.iloc[idx]['calibrated_yes_logprob']
            calibrated_no = new_df.iloc[idx]['calibrated_no_logprob']
            batchcalib_predicted = "Yes" if calibrated_yes > calibrated_no else "No"
            new_df.loc[idx, 'batchcalib_predicted_answer'] = batchcalib_predicted
            new_df.loc[idx, 'batchcalib_is_correct'] = batchcalib_predicted == new_df.iloc[idx]['Correct Answer']

    print("Batch calibration completed successfully.")
    return new_df