# this is nli_BOS_BC_CC.py

import torch, pandas as pd
from tqdm import tqdm
import numpy as np
import os
import gc
from pathlib import Path
from get_query_logprobs import calculate_logprobs_batch_nli

# Model configurations for BOS correction
falcon_models = ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"]
gemma3_models = ["gemma-3-27b-pt", "gemma-3-27b-it", "gemma-3-12b-pt", "gemma-3-12b-it"]
# qwen2_models = ["Qwen2.5-14B", "Qwen2.5-14B-Instruct", "Qwen2.5-32B", "Qwen2.5-32B-Instruct"]
# llama_models = ["Llama-2-7b-hf", "Llama-2-7b-chat-hf", "Llama-2-13b-hf", "Llama-2-13b-chat-hf"]
llama3_models = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"]

MODEL_CONFIGS = {
    "Falcon": {"models": falcon_models, "prefix": "tiiuae/"},
    # "Qwen2": {"models": qwen2_models, "prefix": "Qwen/"},
    # "Llama": {"models": llama_models, "prefix": "meta-llama/"},
    "Llama3": {"models": llama3_models, "prefix": "meta-llama/"},
    "Gemma3": {"models": gemma3_models, "prefix": "google/"},   
}

def process_dataset_nli(input_file, output_file, plain_file, impl, model_name, model, model_family, tokenizer, domain, calib_count=80, batch_size=8, dataset=None, prompt=None, calib_data=None, batch_size_param=None):

    print("impl: ", impl)

    # For correction implementations, check if plain results exist first
    if impl in ["nlispecific", "nlibos", "nlicontextcalib", "nlibatchcalib"]:
        # Get the model_family for path reconstruction
        nliplain_output_file = plain_file
        
        # If plain results exist, use them (regardless of whether model is loaded)
        if os.path.exists(nliplain_output_file):
            print(f"Found existing plain results: {nliplain_output_file}")
            df = pd.read_csv(nliplain_output_file)
            
            # Apply the requested correction
            if impl == "nlispecific":
                print("Applying specific correction to existing plain results...")
                df = do_entire_specific_thing(df, dataset, calib_count, calib_data)
                df.to_csv(output_file, index=False)
                print(f"specific corrected results saved to {output_file}")
                return
            
            elif impl == "nlibos":
                print("Applying BOS correction to existing plain results...")
                bos_df = pd.read_csv('../data/bos_logprobs_nli.csv')
                
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
                
                bos_o0 = row['bos_o0_logprob'].iloc[0]
                bos_o1 = row['bos_o1_logprob'].iloc[0]
                bos_o2 = row['bos_o2_logprob'].iloc[0]
                df = do_entire_bos_thing(df, bos_o0, bos_o1, bos_o2)
                df.to_csv(output_file, index=False)
                print(f"BOS corrected results saved to {output_file}")
                return
            
            elif impl == "nlicontextcalib":
                print("Applying contextual calibration to existing plain results...")
                df = do_entire_contextcalib_thing(df, model, tokenizer, dataset, prompt, model_name)
                df.to_csv(output_file, index=False)
                print(f"Contextual calibration results saved to {output_file}")
                return

            elif impl == "nlibatchcalib":
                print("Applying batch calibration to existing plain results...")
                df = do_entire_batchcalib_thing(df, batch_size=batch_size_param)
                df.to_csv(output_file, index=False)
                print(f"Batch calibration results saved to {output_file}")
                return
        
        else:
            # Plain results don't exist
            if model is None and tokenizer is None:
                print(f"ERROR: Expected plain results at {nliplain_output_file} but file doesn't exist!")
                print("Cannot run correction-only mode without existing plain results.")
                return
            else:
                print(f"Plain results don't exist at {nliplain_output_file}")
                print("Will run full inference first, then apply corrections.")
                # Continue to inference section below

    # INFERENCE MODE - only reached if:
    # 1. impl is "nliplain", OR  
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
    df['o0_logprob'] = None
    df['o1_logprob'] = None
    df['o2_logprob'] = None
    df['predicted_answer'] = None
    df['is_correct'] = None
    
    # Run inference
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i + batch_size]
        batch_results = calculate_logprobs_batch_nli(
            batch_df, tokenizer, model, dataset, prompt, content_free_mode=False, content_free_input=None
        )
        
        for j, result in enumerate(batch_results):
            if i + j >= len(df): break
            idx = i + j
            
            for key, value in result.items():
                df.loc[idx, key] = value

            logprob_cols = ['o0_logprob', 'o1_logprob', 'o2_logprob']
            logprobs = [float(df.loc[idx, col]) for col in logprob_cols]
            max_idx = np.argmax(logprobs) # index of the maximum logprob
            df.loc[idx, 'predicted_answer'] = ['0', '1', '2'][max_idx]
            df.loc[idx, 'is_correct'] = df.loc[idx, 'predicted_answer'] == df.loc[idx, 'Correct Answer']
        
        if i % (batch_size * 5) == 0:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if i % (batch_size * 20) == 0:
                gc.collect()

    # Save plain results (for future correction-only runs)
    if impl in ["nlispecific", "nlibos", "nlicontextcalib", "nlibatchcalib"]:
        # Reconstruct plain path
        nliplain_output_file = plain_file
        plain_output_file = Path(nliplain_output_file)
        plain_output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(nliplain_output_file, index=False)
        print(f"Plain results saved to {nliplain_output_file}")

    # Apply corrections if needed
    if impl == "nlispecific":
        print("Applying specific correction to fresh inference results...")
        df = do_entire_specific_thing(df, dataset, calib_count, calib_data)
        df.to_csv(output_file, index=False)
        print(f"specific corrected results saved to {output_file}")
        return

    elif impl == "nlibos":
        print("Applying BOS correction to fresh inference results...")
        bos_df = pd.read_csv('../data/bos_logprobs_nli.csv')
        
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

        bos_o0 = row['bos_0_logprob'].iloc[0]
        bos_o1 = row['bos_1_logprob'].iloc[0]
        bos_o2 = row['bos_2_logprob'].iloc[0]
        df = do_entire_bos_thing(df, bos_o0, bos_o1, bos_o2)
        df.to_csv(output_file, index=False)
        print(f"BOS corrected results saved to {output_file}")
        return
    
    elif impl == "nlicontextcalib":
        print("Applying contextual calibration to fresh inference results...")
        df = do_entire_contextcalib_thing(df, model, tokenizer, dataset, prompt, model_name)
        df.to_csv(output_file, index=False)
        print(f"Contextual calibration results saved to {output_file}")
        return

    elif impl == "nlibatchcalib":
        print("Applying batch calibration to fresh inference results...")
        df = do_entire_batchcalib_thing(df)
        df.to_csv(output_file, index=False)
        print(f"Batch calibration results saved to {output_file}")
        return
    
    # For plain implementation, save directly
    elif impl == "nliplain":
        df.to_csv(output_file, index=False)
        print(f"Plain results saved to {output_file}")
        return


def do_entire_bos_thing(plain_df, bos_o0_logprob, bos_o1_logprob, bos_o2_logprob):
    """Apply BOS (Beginning of Sequence) correction to nli logprobs"""
    
    total_questions = len(plain_df)
    new_df = _initialize_dataframe(plain_df, "bos")
    
    print(f"Applying BOS correction to {total_questions} questions...")

    # Store BOS values
    new_df['bos_o0_logprob'] = bos_o0_logprob
    new_df['bos_o1_logprob'] = bos_o1_logprob
    new_df['bos_o2_logprob'] = bos_o2_logprob

    # Calculate corrected logprobs
    new_df['corrected_o0_logprob'] = new_df['raw_o0_logprob'] - bos_o0_logprob
    new_df['corrected_o1_logprob'] = new_df['raw_o1_logprob'] - bos_o1_logprob
    new_df['corrected_o2_logprob'] = new_df['raw_o2_logprob'] - bos_o2_logprob

    # Calculate BOS predictions
    max_logprobs = new_df[['corrected_o0_logprob', 'corrected_o1_logprob', 'corrected_o2_logprob']].idxmax(axis=1)
    new_df['bos_predicted_answer'] = max_logprobs.map({
        'corrected_o0_logprob': '0', 'corrected_o1_logprob': '1', 
        'corrected_o2_logprob': '2'
    })
    new_df['bos_is_correct'] = new_df['bos_predicted_answer'] == new_df['Correct Answer']

    print("BOS correction completed successfully.")
    return new_df


def do_entire_contextcalib_thing(plain_df, model, tokenizer, dataset, prompt, model_name):
    """Apply contextual calibration to nli logprobs using pre-computed content-free probabilities"""
    
    print(f"=== CONTEXTUAL CALIBRATION START ===")
    total_questions = len(plain_df)
    print(f"Processing {total_questions} questions...")
    
    # Initialize new dataframe
    new_df = _initialize_dataframe(plain_df, "contextcalib")
    
    # Load pre-computed content-free probabilities from CSV
    print(f"Loading pre-computed content-free probabilities...")
    cf_df = pd.read_csv('../data/contextfree_probs_nli.csv')
    
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
            f"Please run contextfree_probs_getter_nli.py first to generate these values."
        )
    
    # Extract the content-free probabilities
    p_cf_o0 = matching_rows['cf_o0_prob'].iloc[0]
    p_cf_o1 = matching_rows['cf_o1_prob'].iloc[0]
    p_cf_o2 = matching_rows['cf_o2_prob'].iloc[0]

    print(f"Loaded content-free probabilities: 0={p_cf_o0:.4f}, 1={p_cf_o1:.4f}, 2={p_cf_o2:.4f}")
    
    # Apply calibration to all questions
    for idx in range(len(new_df)):
        # Get raw probabilities
        raw_o0_logprob = new_df.iloc[idx]['raw_o0_logprob']
        raw_o1_logprob = new_df.iloc[idx]['raw_o1_logprob']
        raw_o2_logprob = new_df.iloc[idx]['raw_o2_logprob']
        
        # Convert to probabilities and normalize
        raw_o0_prob = np.exp(raw_o0_logprob)
        raw_o1_prob = np.exp(raw_o1_logprob)
        raw_o2_prob = np.exp(raw_o2_logprob)
        raw_total = raw_o0_prob + raw_o1_prob + raw_o2_prob
        raw_o0_prob = raw_o0_prob / raw_total
        raw_o1_prob = raw_o1_prob / raw_total
        raw_o2_prob = raw_o2_prob / raw_total

        # Apply contextual calibration: W = diag(p_cf)^(-1)
        calibrated_o0 = raw_o0_prob / p_cf_o0
        calibrated_o1 = raw_o1_prob / p_cf_o1
        calibrated_o2 = raw_o2_prob / p_cf_o2
        
        # Renormalize
        calib_total = calibrated_o0 + calibrated_o1 + calibrated_o2
        calibrated_o0 = calibrated_o0 / calib_total
        calibrated_o1 = calibrated_o1 / calib_total
        calibrated_o2 = calibrated_o2 / calib_total

        # Convert back to logprobs and store
        new_df.loc[idx, 'calibrated_o0_logprob'] = np.log(calibrated_o0)
        new_df.loc[idx, 'calibrated_o1_logprob'] = np.log(calibrated_o1)
        new_df.loc[idx, 'calibrated_o2_logprob'] = np.log(calibrated_o2)
        
        # Store content-free probabilities for reference
        new_df.loc[idx, 'cf_o0_prob'] = p_cf_o0
        new_df.loc[idx, 'cf_o1_prob'] = p_cf_o1
        new_df.loc[idx, 'cf_o2_prob'] = p_cf_o2
        
        # Make predictions
        calibrated_probs = [calibrated_o0, calibrated_o1, calibrated_o2]
        max_idx = np.argmax(calibrated_probs)
        contextcalib_predicted = ['0', '1', '2'][max_idx]
        new_df.loc[idx, 'contextcalib_predicted_answer'] = contextcalib_predicted
        new_df.loc[idx, 'contextcalib_is_correct'] = contextcalib_predicted == str(new_df.iloc[idx]['Correct Answer'])

    print("Contextual calibration completed successfully.")
    return new_df


def _initialize_dataframe(df, impl):
    """Initialize dataframe for nli corrections"""
    new_df = df.copy()

    if impl == "bos":
        # BOS correction columns
        bos_columns = [
            'raw_o0_logprob', 'raw_o1_logprob', 'raw_o2_logprob', 
            'corrected_o0_logprob', 'corrected_o1_logprob', 'corrected_o2_logprob', 
            'bos_0_logprob', 'bos_1_logprob', 'bos_2_logprob', 
            'plain_predicted_answer', 'bos_predicted_answer', 'plain_is_correct', 'bos_is_correct'
        ]
        
        # Initialize columns
        for col in bos_columns:
            new_df[col] = None

        # Store raw values and plain results
        new_df['raw_o0_logprob'] = new_df['o0_logprob'].astype(float)
        new_df['raw_o1_logprob'] = new_df['o1_logprob'].astype(float)
        new_df['raw_o2_logprob'] = new_df['o2_logprob'].astype(float)
        new_df['plain_predicted_answer'] = new_df['predicted_answer']
        new_df['plain_is_correct'] = new_df['is_correct']
        
        new_df.drop(columns=['o0_logprob', 'o1_logprob', 'o2_logprob', 'predicted_answer', 'is_correct'], inplace=True)
    elif impl == "specific":
        # Specific correction columns (all lists)
        specific_columns = [
            'eval_split_ratio', 'calib_set_ID', 'raw_o0_logprob', 'raw_o1_logprob', 'raw_o2_logprob', 
            'corrected_o0_logprob', 'corrected_o1_logprob', 'corrected_o2_logprob', 
            'calib_o0_mean', 'calib_o1_mean', 'calib_o2_mean', 'plain_predicted_answer', 
            'specific_predicted_answer', 'plain_is_correct', 'specific_is_correct'
        ]
        
        # Initialize all columns as lists
        for col in specific_columns:
            new_df[col] = [[] for _ in range(len(new_df))]

        # Store raw values as single-item lists
        new_df['raw_o0_logprob'] = new_df['o0_logprob'].apply(lambda x: [x])
        new_df['raw_o1_logprob'] = new_df['o1_logprob'].apply(lambda x: [x])
        new_df['raw_o2_logprob'] = new_df['o2_logprob'].apply(lambda x: [x])
        new_df['plain_predicted_answer'] = new_df['predicted_answer'].apply(lambda x: [x])
        new_df['plain_is_correct'] = new_df['is_correct'].apply(lambda x: [x])
        
        # Remove original columns
        new_df.drop(columns=['o0_logprob', 'o1_logprob', 'o2_logprob', 'predicted_answer', 'is_correct'], inplace=True)
        
    elif impl == "contextcalib":
        # Contextual calibration columns
        contextcalib_columns = [
            'raw_o0_logprob', 'raw_o1_logprob', 'raw_o2_logprob', 
            'calibrated_o0_logprob', 'calibrated_o1_logprob', 'calibrated_o2_logprob',
            'cf_o0_prob', 'cf_o1_prob', 'cf_o2_prob', 
            'raw_predicted_answer', 'raw_is_correct', 
            'contextcalib_predicted_answer', 'contextcalib_is_correct'
        ]
        
        # Initialize columns
        for col in contextcalib_columns:
            new_df[col] = None
        
        # Store raw values
        new_df['raw_o0_logprob'] = new_df['o0_logprob']
        new_df['raw_o1_logprob'] = new_df['o1_logprob']
        new_df['raw_o2_logprob'] = new_df['o2_logprob']
        new_df['raw_predicted_answer'] = new_df['predicted_answer']
        new_df['raw_is_correct'] = new_df['is_correct']
        
        # Remove original columns
        new_df.drop(columns=['o0_logprob', 'o1_logprob', 'o2_logprob', 'predicted_answer', 'is_correct'], inplace=True)

    elif impl == "batchcalib":
        batchcalib_columns = [
            'raw_o0_logprob', 'raw_o1_logprob', 'raw_o2_logprob',
            'batch_o0_mean', 'batch_o1_mean', 'batch_o2_mean',
            'calibrated_o0_logprob', 'calibrated_o1_logprob', 'calibrated_o2_logprob',
            'raw_predicted_answer', 'raw_is_correct', 
            'batchcalib_predicted_answer', 'batchcalib_is_correct'
        ]
        
        # Initialize columns
        for col in batchcalib_columns:
            new_df[col] = None
        
        # Store raw values
        new_df['raw_o0_logprob'] = new_df['o0_logprob']
        new_df['raw_o1_logprob'] = new_df['o1_logprob']
        new_df['raw_o2_logprob'] = new_df['o2_logprob']
        new_df['raw_predicted_answer'] = new_df['predicted_answer']
        new_df['raw_is_correct'] = new_df['is_correct']
        
        # Remove original columns
        new_df.drop(columns=['o0_logprob', 'o1_logprob', 'o2_logprob', 'predicted_answer', 'is_correct'], inplace=True)
        
    return new_df


def _get_calibration_indices_nli(calib_count, total_questions, calib_run, eval_df, calib_df=None):
    """Generate calibration and evaluation indices for nli with class balancing across 0, 1, 2
    
    Args:
        calib_count: Number of calibration samples to draw (0 means use entire dataset)
        total_questions: Total questions in target (evaluation) dataset
        calib_run: Random seed for this calibration run
        eval_df: Target dataset dataframe
        calib_df: External calibration dataframe (None for same-dataset, DataFrame for cross-dataset/model)
    """
    np.random.seed(calib_run)  # For reproducibility
    
    # Use external calibration data if provided (cross-dataset OR cross-model)
    calibration_source = calib_df if calib_df is not None else eval_df

    # print(f"Generating calibration indices for {calib_count} questions...")
    # print(f"Total questions in calibration source: {len(calibration_source)}")
    
    # Separate questions by correct answer from calibration source
    o0_indices = calibration_source[calibration_source['Correct Answer'] == 0].index.tolist()
    o1_indices = calibration_source[calibration_source['Correct Answer'] == 1].index.tolist()
    o2_indices = calibration_source[calibration_source['Correct Answer'] == 2].index.tolist()

    # print(f"Found {len(o0_indices)} 0s, {len(o1_indices)} 1s, {len(o2_indices)} 2s in calibration source.")

    # Calculate how many questions we need from each class
    questions_per_class = calib_count // 3
    remainder = calib_count % 3
    
    # Distribute remainder questions across classes, rotating which classes get extra
    # This ensures fairness across different runs
    class_counts = [questions_per_class] * 3
    for i in range(remainder):
        class_counts[(calib_run + i) % 3] += 1
    
    o0_needed, o1_needed, o2_needed = class_counts
    
    # Sample with replacement from each class
    selected_o0 = np.random.choice(o0_indices, size=o0_needed, replace=True) if o0_needed > 0 and len(o0_indices) > 0 else []
    selected_o1 = np.random.choice(o1_indices, size=o1_needed, replace=True) if o1_needed > 0 and len(o1_indices) > 0 else []
    selected_o2 = np.random.choice(o2_indices, size=o2_needed, replace=True) if o2_needed > 0 and len(o2_indices) > 0 else []
    
    # Combine all selected indices
    calib_indices = list(selected_o0) + list(selected_o1) + list(selected_o2)
    # Evaluation indices depend on whether we're using external calibration
    if calib_df is not None:
        # Cross-dataset OR cross-model mode: all target dataset questions are evaluated
        eval_indices = list(range(total_questions))
    else:
        # Same-dataset, same-model mode: exclude calibration questions
        eval_indices = [i for i in range(total_questions) if i not in set(calib_indices)]
    
    return calib_indices, eval_indices


def _calculate_calibration_means_nli(calibration_source, calib_indices, cross_domain_mode=False):
    """Calculate mean logprobs from calibration set for all nli options
    
    Args:
        calibration_source: DataFrame containing calibration data
        calib_indices: Indices to use for calibration
        cross_domain_mode: If True, calibration_source is external data (cross-dataset or cross-model)
    """
    if cross_domain_mode:
        # Use external calibration data (could be different dataset and/or different model)
        # calib_oa_mean = np.mean([calibration_source.iloc[i]['oa_logprob'] for i in calib_indices])
        # calib_ob_mean = np.mean([calibration_source.iloc[i]['ob_logprob'] for i in calib_indices])
        # calib_oc_mean = np.mean([calibration_source.iloc[i]['oc_logprob'] for i in calib_indices])
        # calib_od_mean = np.mean([calibration_source.iloc[i]['od_logprob'] for i in calib_indices])

        # Testing median instead of mean for robustness to outliers
        calib_o0_mean = np.median([calibration_source.iloc[i]['o0_logprob'] for i in calib_indices])
        calib_o1_mean = np.median([calibration_source.iloc[i]['o1_logprob'] for i in calib_indices])
        calib_o2_mean = np.median([calibration_source.iloc[i]['o2_logprob'] for i in calib_indices])
    else:
        # Use same-dataset, same-model calibration
        # calib_oa_mean = np.mean([calibration_source.iloc[i]['raw_oa_logprob'][0] for i in calib_indices])
        # calib_ob_mean = np.mean([calibration_source.iloc[i]['raw_ob_logprob'][0] for i in calib_indices])
        # calib_oc_mean = np.mean([calibration_source.iloc[i]['raw_oc_logprob'][0] for i in calib_indices])
        # calib_od_mean = np.mean([calibration_source.iloc[i]['raw_od_logprob'][0] for i in calib_indices])

        # Testing median instead of mean for robustness to outliers
        calib_o0_mean = np.median([calibration_source.iloc[i]['raw_o0_logprob'][0] for i in calib_indices])
        calib_o1_mean = np.median([calibration_source.iloc[i]['raw_o1_logprob'][0] for i in calib_indices])
        calib_o2_mean = np.median([calibration_source.iloc[i]['raw_o2_logprob'][0] for i in calib_indices])

    return calib_o0_mean, calib_o1_mean, calib_o2_mean


def _apply_correction_to_eval_set_nli(eval_df, eval_indices, calib_means, calib_run, total_questions, calib_df=None):
    """Apply bias correction to evaluation set questions for nli"""
    calib_o0_mean, calib_o1_mean, calib_o2_mean = calib_means
    eval_ratio = len(eval_indices) / total_questions
    
    for eval_idx in eval_indices:
        # Store calibration means
        eval_df.loc[eval_idx, 'calib_o0_mean'].append(calib_o0_mean)
        eval_df.loc[eval_idx, 'calib_o1_mean'].append(calib_o1_mean)
        eval_df.loc[eval_idx, 'calib_o2_mean'].append(calib_o2_mean)
        
        # Calculate corrected logprobs
        raw_o0 = eval_df.iloc[eval_idx]['raw_o0_logprob'][0]
        raw_o1 = eval_df.iloc[eval_idx]['raw_o1_logprob'][0]
        raw_o2 = eval_df.iloc[eval_idx]['raw_o2_logprob'][0]
        
        corrected_o0 = raw_o0 - calib_o0_mean
        corrected_o1 = raw_o1 - calib_o1_mean
        corrected_o2 = raw_o2 - calib_o2_mean
        
        eval_df.loc[eval_idx, 'corrected_o0_logprob'].append(corrected_o0)
        eval_df.loc[eval_idx, 'corrected_o1_logprob'].append(corrected_o1)
        eval_df.loc[eval_idx, 'corrected_o2_logprob'].append(corrected_o2)
        
        # Calculate predictions and accuracy
        corrected_logprobs = [corrected_o0, corrected_o1, corrected_o2]
        max_idx = np.argmax(corrected_logprobs)
        specific_predicted = ['0', '1', '2'][max_idx]
        correct_answer = eval_df.iloc[eval_idx]['Correct Answer']
        specific_correct = (specific_predicted == str(correct_answer))
        
        eval_df.loc[eval_idx, 'specific_predicted_answer'].append(specific_predicted)
        eval_df.loc[eval_idx, 'specific_is_correct'].append(specific_correct)
        
        # Track metadata
        eval_df.loc[eval_idx, 'calib_set_ID'].append(calib_run + 1)
        eval_df.loc[eval_idx, 'eval_split_ratio'].append(eval_ratio)


def do_entire_specific_thing(plain_df, dataset_name, calib_count, calib_data=None):
    """Apply specific cross-validation bias correction to nli logprobs
    
    Args:
        plain_df: Target dataset with plain inference results
        dataset_name: Name of target dataset (currently unused for nli, kept for API consistency)
        calib_count: Number of calibration samples per run
        calib_data: External calibration data (for cross-dataset or cross-model correction)
                   If None, uses same-dataset correction
    """
    
    print(f"=== SPECIFIC CORRECTION START ===")
    print(f"Target Dataset: {dataset_name}, Calibration count: {calib_count} questions")
    
    # Check cross-dataset/model mode
    cross_domain_mode = calib_data is not None
    if cross_domain_mode:
        print(f"Cross-transfer mode: Using external calibration data with {len(calib_data)} samples")
        print(f"Note: External data could be from different dataset and/or different model")
    
    # Check if already processed
    if any(col in plain_df.columns for col in ['raw_o0_logprob', 'fold', 'corrected_o0_logprob']):
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
        calib_indices, eval_indices = _get_calibration_indices_nli(
            calib_count, total_questions, calib_run, new_df, calib_data
        )

        # print(f"  Calibration set size: {len(calib_indices)} questions")
        
        # Calculate calibration means from appropriate source
        if cross_domain_mode:
            # Use external calibration data (could be different dataset and/or different model)
            calib_means = _calculate_calibration_means_nli(calib_data, calib_indices, cross_domain_mode=True)
        else:
            # Use same-dataset, same-model calibration
            calib_means = _calculate_calibration_means_nli(new_df, calib_indices, cross_domain_mode=False)
        
        # print(f"  Calibration means for run {calib_run + 1}: " + str(calib_means))

        # Apply correction to evaluation set
        _apply_correction_to_eval_set_nli(
            new_df, eval_indices, calib_means, calib_run, total_questions, calib_data
        )

    # Summary statistics
    avg_evaluations = np.mean([len(new_df.iloc[i]['calib_set_ID']) for i in range(total_questions)])
    print("Specific correction completed successfully.")
    print(f"Average evaluations per question: {avg_evaluations:.1f}")
    
    return new_df


def do_entire_batchcalib_thing(plain_df, batch_size=None):
    """Apply batch calibration to nli logprobs with running mean correction
    
    Args:
        plain_df: DataFrame with plain inference results
        batch_size: Size of each batch. If None, uses entire dataset.
    """
    
    print(f"=== BATCH CALIBRATION START ===")
    total_questions = len(plain_df)
    
    # If no batch size specified, use entire dataset (original paper's approach)
    if batch_size is None:
        batch_size = total_questions
    
    print(f"Processing {total_questions} questions with batch_size={batch_size}")
    
    # Initialize new dataframe
    new_df = _initialize_dataframe(plain_df, "batchcalib")

    # Shuffle dataset to avoid systematic ordering bias (e.g., all A answers first)
    shuffled_indices = np.random.RandomState(seed=42).permutation(len(new_df))
    new_df = new_df.iloc[shuffled_indices].reset_index(drop=True)
    print(f"Dataset shuffled with fixed seed for reproducibility")
    
    # Process dataset in batches
    num_batches = (total_questions + batch_size - 1) // batch_size  # Ceiling division
    
    # Initialize running means (will be calculated from first batch)
    running_o0_mean = None
    running_o1_mean = None
    running_o2_mean = None
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_questions)
        
        # Get batch
        batch_indices = range(start_idx, end_idx)
        
        # Calculate current batch means
        current_batch_o0_mean = new_df.iloc[start_idx:end_idx]['raw_o0_logprob'].mean()
        current_batch_o1_mean = new_df.iloc[start_idx:end_idx]['raw_o1_logprob'].mean()
        current_batch_o2_mean = new_df.iloc[start_idx:end_idx]['raw_o2_logprob'].mean()

        # Update running means using equation (3)
        if batch_idx == 0:
            # First batch: initialize running mean
            running_o0_mean = current_batch_o0_mean
            running_o1_mean = current_batch_o1_mean
            running_o2_mean = current_batch_o2_mean
        else:
            # Subsequent batches: update running mean
            # p̂ᵣⁿ⁺¹(y|C) = (n/(n+1))p̂ᵣⁿ(y|C) + (1/(n+1))p̂ⁿ⁺¹(y|C)
            n = batch_idx  # number of batches seen so far (before this one)
            running_o0_mean = (n / (n + 1)) * running_o0_mean + (1 / (n + 1)) * current_batch_o0_mean
            running_o1_mean = (n / (n + 1)) * running_o1_mean + (1 / (n + 1)) * current_batch_o1_mean
            running_o2_mean = (n / (n + 1)) * running_o2_mean + (1 / (n + 1)) * current_batch_o2_mean

        print(f"  Batch {batch_idx + 1}/{num_batches} (questions {start_idx}-{end_idx-1}): " +
              f"Current 0={current_batch_o0_mean:.4f}, 1={current_batch_o1_mean:.4f}, 2={current_batch_o2_mean:.4f}, " +
              f"Running 0={running_o0_mean:.4f}, 1={running_o1_mean:.4f}, 2={running_o2_mean:.4f}")
        
        # Apply calibration to questions in this batch using the running mean
        for idx in batch_indices:
            # Store running means for reference
            new_df.loc[idx, 'batch_o0_mean'] = running_o0_mean
            new_df.loc[idx, 'batch_o1_mean'] = running_o1_mean
            new_df.loc[idx, 'batch_o2_mean'] = running_o2_mean
            
            # Calculate corrected logprobs using running mean
            new_df.loc[idx, 'calibrated_o0_logprob'] = new_df.iloc[idx]['raw_o0_logprob'] - running_o0_mean
            new_df.loc[idx, 'calibrated_o1_logprob'] = new_df.iloc[idx]['raw_o1_logprob'] - running_o1_mean
            new_df.loc[idx, 'calibrated_o2_logprob'] = new_df.iloc[idx]['raw_o2_logprob'] - running_o2_mean

            # Make predictions
            calibrated_logprobs = [
                new_df.iloc[idx]['calibrated_o0_logprob'],
                new_df.iloc[idx]['calibrated_o1_logprob'], 
                new_df.iloc[idx]['calibrated_o2_logprob']
            ]
            max_idx = np.argmax(calibrated_logprobs)
            batchcalib_predicted = ['0', '1', '2'][max_idx]
            new_df.loc[idx, 'batchcalib_predicted_answer'] = batchcalib_predicted
            new_df.loc[idx, 'batchcalib_is_correct'] = batchcalib_predicted == str(new_df.iloc[idx]['Correct Answer'])

    print("Batch calibration completed successfully.")
    return new_df


# def do_entire_batchcalib_thing(plain_df, batch_size=None):
#     """Apply batch calibration to nli logprobs with within-batch correction
    
#     Args:
#         plain_df: DataFrame with plain inference results
#         batch_size: Size of each batch. If None, uses entire dataset.
#     """
    
#     print(f"=== BATCH CALIBRATION START ===")
#     total_questions = len(plain_df)
    
#     # If no batch size specified, use entire dataset (original paper's approach)
#     if batch_size is None:
#         batch_size = total_questions
    
#     print(f"Processing {total_questions} questions with batch_size={batch_size}")
    
#     # Initialize new dataframe
#     new_df = _initialize_dataframe(plain_df, "batchcalib")

#     # Shuffle dataset to avoid systematic ordering bias (e.g., all A answers first)
#     shuffled_indices = np.random.RandomState(seed=42).permutation(len(new_df))
#     new_df = new_df.iloc[shuffled_indices].reset_index(drop=True)
#     print(f"Dataset shuffled with fixed seed for reproducibility")
    
#     # Process dataset in batches
#     num_batches = (total_questions + batch_size - 1) // batch_size  # Ceiling division
    
#     for batch_idx in range(num_batches):
#         start_idx = batch_idx * batch_size
#         end_idx = min(start_idx + batch_size, total_questions)
        
#         # Get batch
#         batch_indices = range(start_idx, end_idx)
        
#         # Calculate batch means (no class balancing, use all questions in this batch)
#         batch_o0_mean = new_df.iloc[start_idx:end_idx]['raw_o0_logprob'].mean()
#         batch_o1_mean = new_df.iloc[start_idx:end_idx]['raw_o1_logprob'].mean()
#         batch_o2_mean = new_df.iloc[start_idx:end_idx]['raw_o2_logprob'].mean()

#         print(f"  Batch {batch_idx + 1}/{num_batches} (questions {start_idx}-{end_idx-1}): " +
#               f"0={batch_o0_mean:.4f}, 1={batch_o1_mean:.4f}, 2={batch_o2_mean:.4f}")
        
#         # Apply calibration to questions in this batch only
#         for idx in batch_indices:
#             # Store batch means for reference
#             new_df.loc[idx, 'batch_o0_mean'] = batch_o0_mean
#             new_df.loc[idx, 'batch_o1_mean'] = batch_o1_mean
#             new_df.loc[idx, 'batch_o2_mean'] = batch_o2_mean
            
#             # Calculate corrected logprobs
#             new_df.loc[idx, 'calibrated_o0_logprob'] = new_df.iloc[idx]['raw_o0_logprob'] - batch_o0_mean
#             new_df.loc[idx, 'calibrated_o1_logprob'] = new_df.iloc[idx]['raw_o1_logprob'] - batch_o1_mean
#             new_df.loc[idx, 'calibrated_o2_logprob'] = new_df.iloc[idx]['raw_o2_logprob'] - batch_o2_mean

#             # Make predictions
#             calibrated_logprobs = [
#                 new_df.iloc[idx]['calibrated_o0_logprob'],
#                 new_df.iloc[idx]['calibrated_o1_logprob'], 
#                 new_df.iloc[idx]['calibrated_o2_logprob']
#             ]
#             max_idx = np.argmax(calibrated_logprobs)
#             batchcalib_predicted = ['0', '1', '2'][max_idx]
#             new_df.loc[idx, 'batchcalib_predicted_answer'] = batchcalib_predicted
#             new_df.loc[idx, 'batchcalib_is_correct'] = batchcalib_predicted == new_df.iloc[idx]['Correct Answer']

#     print("Batch calibration completed successfully.")
#     return new_df