# calibration_core.py
# Unified calibration logic for all question types

import torch
import pandas as pd
import numpy as np
from pathlib import Path


class QuestionTypeConfig:
    """Configuration for different question types"""
    
    def __init__(self, name, options, logprob_cols, raw_cols, corrected_cols, 
                 calib_mean_cols, batch_mean_cols, cf_prob_cols, bos_cols,
                 answer_col, predicted_col, correct_col, bos_csv_file):
        self.name = name
        self.options = options  # ['Yes', 'No'] or ['0', '1', '2'] or ['A', 'B', 'C', 'D']
        self.num_options = len(options)
        
        # Column names for different stages
        self.logprob_cols = logprob_cols  # ['yes_logprob', 'no_logprob']
        self.raw_cols = raw_cols  # ['raw_yes_logprob', 'raw_no_logprob']
        self.corrected_cols = corrected_cols  # ['corrected_yes_logprob', ...]
        self.calib_mean_cols = calib_mean_cols  # ['calib_yes_mean', ...]
        self.batch_mean_cols = batch_mean_cols  # ['batch_yes_mean', ...]
        self.cf_prob_cols = cf_prob_cols  # ['cf_yes_prob', 'cf_no_prob']
        self.bos_cols = bos_cols  # ['bos_yes_logprob', 'bos_no_logprob']
        
        # Answer column names
        self.answer_col = answer_col  # 'Correct Answer' or 'answer'
        self.predicted_col = predicted_col  # 'predicted_answer'
        self.correct_col = correct_col  # 'is_correct'
        
        # BOS data file
        self.bos_csv_file = bos_csv_file


# Define configurations for each question type
YESNO_CONFIG = QuestionTypeConfig(
    name='yesno',
    options=['Yes', 'No'],
    logprob_cols=['yes_logprob', 'no_logprob'],
    raw_cols=['raw_yes_logprob', 'raw_no_logprob'],
    corrected_cols=['corrected_yes_logprob', 'corrected_no_logprob'],
    calib_mean_cols=['calib_yes_mean', 'calib_no_mean'],
    batch_mean_cols=['batch_yes_mean', 'batch_no_mean'],
    cf_prob_cols=['cf_yes_prob', 'cf_no_prob'],
    bos_cols=['bos_yes_val', 'bos_no_val'],
    answer_col='Correct Answer',
    predicted_col='predicted_answer',
    correct_col='is_correct',
    bos_csv_file='../data/bos_logprobs_yesno.csv'
)

NLI_CONFIG = QuestionTypeConfig(
    name='nli',
    options=['0', '1', '2'],
    logprob_cols=['o0_logprob', 'o1_logprob', 'o2_logprob'],
    raw_cols=['raw_o0_logprob', 'raw_o1_logprob', 'raw_o2_logprob'],
    corrected_cols=['corrected_o0_logprob', 'corrected_o1_logprob', 'corrected_o2_logprob'],
    calib_mean_cols=['calib_o0_mean', 'calib_o1_mean', 'calib_o2_mean'],
    batch_mean_cols=['batch_o0_mean', 'batch_o1_mean', 'batch_o2_mean'],
    cf_prob_cols=['cf_o0_prob', 'cf_o1_prob', 'cf_o2_prob'],
    bos_cols=['bos_o0_logprob', 'bos_o1_logprob', 'bos_o2_logprob'],
    answer_col='Correct Answer',
    predicted_col='predicted_answer',
    correct_col='is_correct',
    bos_csv_file='../data/bos_logprobs_nli.csv'
)

MCQ_CONFIG = QuestionTypeConfig(
    name='mcq',
    options=['A', 'B', 'C', 'D'],
    logprob_cols=['oa_logprob', 'ob_logprob', 'oc_logprob', 'od_logprob'],
    raw_cols=['raw_oa_logprob', 'raw_ob_logprob', 'raw_oc_logprob', 'raw_od_logprob'],
    corrected_cols=['corrected_oa_logprob', 'corrected_ob_logprob', 'corrected_oc_logprob', 'corrected_od_logprob'],
    calib_mean_cols=['calib_oa_mean', 'calib_ob_mean', 'calib_oc_mean', 'calib_od_mean'],
    batch_mean_cols=['batch_oa_mean', 'batch_ob_mean', 'batch_oc_mean', 'batch_od_mean'],
    cf_prob_cols=['cf_oa_prob', 'cf_ob_prob', 'cf_oc_prob', 'cf_od_prob'],
    bos_cols=['bos_oa_logprob', 'bos_ob_logprob', 'bos_oc_logprob', 'bos_od_logprob'],
    answer_col='answer',
    predicted_col='predicted_answer',
    correct_col='is_correct',
    bos_csv_file='../data/bos_logprobs_mcq.csv'
)


def initialize_dataframe(df, impl, config):
    """Initialize dataframe with appropriate columns for given implementation"""
    new_df = df.copy()
    
    if impl == "bos":
        # BOS correction columns
        bos_columns = (
            config.raw_cols + 
            config.corrected_cols + 
            config.bos_cols +
            [f'raw_{config.predicted_col}', f'raw_{config.correct_col}',
             f'bos_{config.predicted_col}', f'bos_{config.correct_col}']
        )
        
        for col in bos_columns:
            new_df[col] = None
        
        # Store raw values
        for i, logprob_col in enumerate(config.logprob_cols):
            new_df[config.raw_cols[i]] = new_df[logprob_col].astype(float)
        
        new_df[f'raw_{config.predicted_col}'] = new_df[config.predicted_col]
        new_df[f'raw_{config.correct_col}'] = new_df[config.correct_col]
        new_df.drop(columns=config.logprob_cols + [config.predicted_col, config.correct_col], inplace=True)
    
    elif impl == "specific":
        # Specific correction columns (all lists)
        specific_columns = (
            ['eval_split_ratio', 'calib_set_ID'] +
            config.raw_cols + 
            config.corrected_cols + 
            config.calib_mean_cols +
            [f'raw_{config.predicted_col}', f'raw_{config.correct_col}',
             f'specific_{config.predicted_col}', f'specific_{config.correct_col}']
        )
        
        for col in specific_columns:
            new_df[col] = [[] for _ in range(len(new_df))]
        
        # Store raw values as single-item lists
        for i, logprob_col in enumerate(config.logprob_cols):
            new_df[config.raw_cols[i]] = new_df[logprob_col].apply(lambda x: [x])
        
        new_df[f'raw_{config.predicted_col}'] = new_df[config.predicted_col].apply(lambda x: [x])
        new_df[f'raw_{config.correct_col}'] = new_df[config.correct_col].apply(lambda x: [x])
        new_df.drop(columns=config.logprob_cols + [config.predicted_col, config.correct_col], inplace=True)
    
    elif impl == "contextcalib":
        # Contextual calibration columns
        contextcalib_columns = (
            config.raw_cols + 
            [col.replace('raw', 'calibrated') for col in config.raw_cols] +
            config.cf_prob_cols +
            [f'raw_{config.predicted_col}', f'raw_{config.correct_col}',
             f'contextcalib_{config.predicted_col}', f'contextcalib_{config.correct_col}']
        )
        
        for col in contextcalib_columns:
            new_df[col] = None
        
        for i, logprob_col in enumerate(config.logprob_cols):
            new_df[config.raw_cols[i]] = new_df[logprob_col]
        
        new_df[f'raw_{config.predicted_col}'] = new_df[config.predicted_col]
        new_df[f'raw_{config.correct_col}'] = new_df[config.correct_col]
        new_df.drop(columns=config.logprob_cols + [config.predicted_col, config.correct_col], inplace=True)
    
    elif impl == "batchcalib":
        batchcalib_columns = (
            config.raw_cols + 
            config.batch_mean_cols +
            [col.replace('raw', 'calibrated') for col in config.raw_cols] +
            [f'raw_{config.predicted_col}', f'raw_{config.correct_col}',
             f'batchcalib_{config.predicted_col}', f'batchcalib_{config.correct_col}']
        )
        
        for col in batchcalib_columns:
            new_df[col] = None
        
        for i, logprob_col in enumerate(config.logprob_cols):
            new_df[config.raw_cols[i]] = new_df[logprob_col]
        
        new_df[f'raw_{config.predicted_col}'] = new_df[config.predicted_col]
        new_df[f'raw_{config.correct_col}'] = new_df[config.correct_col]
        new_df.drop(columns=config.logprob_cols + [config.predicted_col, config.correct_col], inplace=True)
    
    return new_df


def do_bos_correction(plain_df, bos_values, config):
    """Apply BOS correction - unified for all question types"""
    total_questions = len(plain_df)
    new_df = initialize_dataframe(plain_df, "bos", config)
    
    print(f"Applying BOS correction to {total_questions} questions...")
    
    # Store BOS values and calculate corrected logprobs
    for i in range(config.num_options):
        new_df[config.bos_cols[i]] = bos_values[i]
        new_df[config.corrected_cols[i]] = new_df[config.raw_cols[i]] - bos_values[i]
    
    # Make predictions
    if config.name == 'yesno':
        new_df[f'bos_{config.predicted_col}'] = new_df[config.corrected_cols[0]] > new_df[config.corrected_cols[1]]
        new_df[f'bos_{config.predicted_col}'] = new_df[f'bos_{config.predicted_col}'].map({True: 'Yes', False: 'No'})
        new_df[f'bos_{config.correct_col}'] = new_df[f'bos_{config.predicted_col}'] == new_df[config.answer_col]
    else:
        max_logprobs = new_df[config.corrected_cols].idxmax(axis=1)
        col_to_option = {config.corrected_cols[i]: config.options[i] for i in range(config.num_options)}
        new_df[f'bos_{config.predicted_col}'] = max_logprobs.map(col_to_option)
        
        if config.name == 'mcq':
            new_df[f'bos_{config.correct_col}'] = new_df[f'bos_{config.predicted_col}'] == new_df[config.answer_col]
        else:  # NLI
            new_df[f'bos_{config.correct_col}'] = new_df[f'bos_{config.predicted_col}'] == new_df[config.answer_col].astype(str)
    
    print("BOS correction completed successfully.")
    return new_df


def get_calibration_indices(dataset_name, calib_count, total_questions, calib_run, eval_df, calib_df, config):
    """Generate calibration and evaluation indices with class balancing"""
    np.random.seed(calib_run)
    
    # Special case for EWOK with grouped questions (only for same-dataset)
    if dataset_name == "EWOK" and calib_df is None:
        group_starts = list(range(0, total_questions - 3, 4))
        calib_groups_needed = max(1, round(calib_count / 4))
        selected_groups = np.random.choice(group_starts, size=calib_groups_needed, replace=True)
        calib_indices = []
        for group_start in selected_groups:
            calib_indices.extend(range(group_start, min(group_start + 4, total_questions)))
    else:
        # Class-balanced sampling
        calibration_source = calib_df if calib_df is not None else eval_df
        
        # Get indices for each answer option
        option_indices = []
        for option in config.options:
            if config.name == 'nli' and calib_df is not None:
                # For NLI cross-transfer, answer is int
                indices = calibration_source[calibration_source[config.answer_col] == int(option)].index.tolist()
            elif config.name == 'nli':
                # For NLI same-dataset, answer is int
                indices = calibration_source[calibration_source[config.answer_col] == int(option)].index.tolist()
            else:
                indices = calibration_source[calibration_source[config.answer_col] == option].index.tolist()
            option_indices.append(indices)
        
        # Calculate samples needed per class
        questions_per_class = calib_count // config.num_options
        remainder = calib_count % config.num_options
        
        class_counts = [questions_per_class] * config.num_options
        for i in range(remainder):
            class_counts[(calib_run + i) % config.num_options] += 1
        
        # Sample with replacement
        calib_indices = []
        for i, indices in enumerate(option_indices):
            if class_counts[i] > 0 and len(indices) > 0:
                selected = np.random.choice(indices, size=class_counts[i], replace=True)
                calib_indices.extend(selected)
    
    # Determine evaluation indices
    if calib_df is not None:
        eval_indices = list(range(total_questions))
    else:
        eval_indices = [i for i in range(total_questions) if i not in set(calib_indices)]
    
    return calib_indices, eval_indices


def calculate_calibration_means(calibration_source, calib_indices, cross_domain_mode, config):
    """Calculate mean/median logprobs from calibration set"""
    calib_means = []
    
    for i, raw_col in enumerate(config.raw_cols):
        if cross_domain_mode:
            values = [calibration_source.iloc[idx][config.logprob_cols[i]] for idx in calib_indices]
        else:
            values = [calibration_source.iloc[idx][raw_col][0] for idx in calib_indices]
        calib_means.append(np.median(values))
    
    return calib_means


def apply_correction_to_eval_set(eval_df, eval_indices, calib_means, calib_run, total_questions, calib_df, config):
    """Apply bias correction to evaluation set"""
    eval_ratio = len(eval_indices) / total_questions
    
    for eval_idx in eval_indices:
        # Store calibration means
        for i, mean_col in enumerate(config.calib_mean_cols):
            eval_df.loc[eval_idx, mean_col].append(calib_means[i])
        
        # Calculate corrected logprobs
        raw_logprobs = [eval_df.iloc[eval_idx][raw_col][0] for raw_col in config.raw_cols]
        corrected_logprobs = [raw - calib_mean for raw, calib_mean in zip(raw_logprobs, calib_means)]
        
        for i, corr_col in enumerate(config.corrected_cols):
            eval_df.loc[eval_idx, corr_col].append(corrected_logprobs[i])
        
        # Make prediction
        max_idx = np.argmax(corrected_logprobs)
        specific_predicted = config.options[max_idx]
        
        correct_answer = eval_df.iloc[eval_idx][config.answer_col]
        if config.name == 'nli':
            specific_correct = (specific_predicted == str(correct_answer))
        else:
            specific_correct = (specific_predicted == correct_answer)
        
        eval_df.loc[eval_idx, f'specific_{config.predicted_col}'].append(specific_predicted)
        eval_df.loc[eval_idx, f'specific_{config.correct_col}'].append(specific_correct)
        eval_df.loc[eval_idx, 'calib_set_ID'].append(calib_run + 1)
        eval_df.loc[eval_idx, 'eval_split_ratio'].append(eval_ratio)


def do_specific_correction(plain_df, dataset_name, calib_count, calib_data, config):
    """Apply specific cross-validation bias correction - unified for all question types"""
    print(f"=== SPECIFIC CORRECTION START ===")
    print(f"Target Dataset: {dataset_name}, Calibration count: {calib_count} questions")
    
    cross_domain_mode = calib_data is not None
    if cross_domain_mode:
        print(f"Cross-transfer mode: Using external calibration data with {len(calib_data)} samples")
    
    total_questions = len(plain_df)
    print(f"Processing {total_questions} target questions...")
    
    new_df = initialize_dataframe(plain_df, "specific", config)
    
    for calib_run in range(100):
        if calib_run % 20 == 0:
            print(f"Calibration run {calib_run + 1}/100")
        
        calib_indices, eval_indices = get_calibration_indices(
            dataset_name, calib_count, total_questions, calib_run, new_df, calib_data, config
        )
        
        calib_means = calculate_calibration_means(
            calib_data if cross_domain_mode else new_df,
            calib_indices, cross_domain_mode, config
        )
        
        apply_correction_to_eval_set(
            new_df, eval_indices, calib_means, calib_run, total_questions, calib_data, config
        )
    
    avg_evaluations = np.mean([len(new_df.iloc[i]['calib_set_ID']) for i in range(total_questions)])
    print("Specific correction completed successfully.")
    print(f"Average evaluations per question: {avg_evaluations:.1f}")
    
    return new_df


def do_contextcalib_correction(plain_df, model, tokenizer, dataset, prompt, model_name, config):
    """Apply contextual calibration - unified for all question types"""
    print(f"=== CONTEXTUAL CALIBRATION START ===")
    total_questions = len(plain_df)
    print(f"Processing {total_questions} questions...")
    
    new_df = initialize_dataframe(plain_df, "contextcalib", config)
    
    # Load pre-computed content-free probabilities
    print(f"Loading pre-computed content-free probabilities...")
    if "Falcon" in model_name and config.name == 'yesno':
        cf_df = pd.read_csv('../data/contextfree_probs_yesno_falcon.csv')
    else:
        cf_df = pd.read_csv(f'../data/contextfree_probs_{config.name}.csv')
    
    matching_rows = cf_df[
        (cf_df['model_name'] == model_name) & 
        (cf_df['dataset'] == dataset) & 
        (cf_df['prompt'] == prompt)
    ]
    
    if len(matching_rows) == 0:
        raise ValueError(
            f"No pre-computed content-free probabilities found for:\n"
            f"  Model: {model_name}, Dataset: {dataset}, Prompt: {prompt}"
        )
    
    # Extract content-free probabilities
    cf_probs = [matching_rows[col].iloc[0] for col in config.cf_prob_cols]
    print(f"Loaded content-free probabilities: {dict(zip(config.options, cf_probs))}")
    
    # Apply calibration
    for idx in range(len(new_df)):
        # Get raw logprobs and convert to probabilities
        raw_logprobs = [new_df.iloc[idx][raw_col] for raw_col in config.raw_cols]
        raw_probs = np.exp(raw_logprobs)
        raw_probs = raw_probs / raw_probs.sum()
        
        # Apply contextual calibration: W = diag(p_cf)^(-1)
        calibrated_probs = raw_probs / np.array(cf_probs)
        calibrated_probs = calibrated_probs / calibrated_probs.sum()
        
        # Store calibrated logprobs and cf probs
        calibrated_logprobs = np.log(calibrated_probs)
        for i in range(config.num_options):
            new_df.loc[idx, config.corrected_cols[i].replace('corrected', 'calibrated')] = calibrated_logprobs[i]
            new_df.loc[idx, config.cf_prob_cols[i]] = cf_probs[i]
        
        # Make prediction
        max_idx = np.argmax(calibrated_probs)
        contextcalib_predicted = config.options[max_idx]
        new_df.loc[idx, f'contextcalib_{config.predicted_col}'] = contextcalib_predicted
        
        correct_answer = new_df.iloc[idx][config.answer_col]
        if config.name == 'nli':
            new_df.loc[idx, f'contextcalib_{config.correct_col}'] = contextcalib_predicted == str(correct_answer)
        else:
            new_df.loc[idx, f'contextcalib_{config.correct_col}'] = contextcalib_predicted == correct_answer
    
    print("Contextual calibration completed successfully.")
    return new_df


def do_batchcalib_correction(plain_df, batch_size, config):
    """Apply batch calibration - unified for all question types"""
    print(f"=== BATCH CALIBRATION START ===")
    total_questions = len(plain_df)
    
    if batch_size == 0:
        batch_size = total_questions
    
    print(f"Processing {total_questions} questions with batch_size={batch_size}")
    
    new_df = initialize_dataframe(plain_df, "batchcalib", config)
    
    # Shuffle dataset
    shuffled_indices = np.random.RandomState(seed=42).permutation(len(new_df))
    new_df = new_df.iloc[shuffled_indices].reset_index(drop=True)
    print(f"Dataset shuffled with fixed seed for reproducibility")
    
    num_batches = (total_questions + batch_size - 1) // batch_size
    running_means = None
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_questions)
        batch_indices = range(start_idx, end_idx)
        
        # Calculate current batch means
        current_means = [new_df.iloc[start_idx:end_idx][raw_col].mean() for raw_col in config.raw_cols]
        
        # Update running means
        if batch_idx == 0:
            running_means = current_means
        else:
            n = batch_idx
            running_means = [(n/(n+1)) * rm + (1/(n+1)) * cm 
                           for rm, cm in zip(running_means, current_means)]
        
        # Apply calibration to batch
        for idx in batch_indices:
            for i in range(config.num_options):
                new_df.loc[idx, config.batch_mean_cols[i]] = running_means[i]
                calibrated_col = config.corrected_cols[i].replace('corrected', 'calibrated')
                new_df.loc[idx, calibrated_col] = new_df.iloc[idx][config.raw_cols[i]] - running_means[i]
            
            # Make prediction
            calibrated_logprobs = [new_df.iloc[idx][col.replace('corrected', 'calibrated')] 
                                  for col in config.corrected_cols]
            max_idx = np.argmax(calibrated_logprobs)
            batchcalib_predicted = config.options[max_idx]
            new_df.loc[idx, f'batchcalib_{config.predicted_col}'] = batchcalib_predicted
            
            correct_answer = new_df.iloc[idx][config.answer_col]
            if config.name == 'nli':
                new_df.loc[idx, f'batchcalib_{config.correct_col}'] = batchcalib_predicted == str(correct_answer)
            else:
                new_df.loc[idx, f'batchcalib_{config.correct_col}'] = batchcalib_predicted == correct_answer
    
    print("Batch calibration completed successfully.")
    return new_df