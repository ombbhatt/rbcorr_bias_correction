# yn_BOS_BC_CC.py (REFACTORED)

import torch
import pandas as pd
from tqdm import tqdm
import os
import gc
from pathlib import Path
from get_query_logprobs import calculate_logprobs_batch_yesno
from calibration_core import (
    YESNO_CONFIG,
    do_bos_correction,
    do_specific_correction,
    do_contextcalib_correction,
    do_batchcalib_correction
)

# Model configurations for BOS correction
gpt2_models = ["gpt2"]
falcon_models = ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"]
gemma3_models = ["gemma-3-27b-pt", "gemma-3-27b-it", "gemma-3-12b-pt", "gemma-3-12b-it"]
llama3_models = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"]

MODEL_CONFIGS = {
    "GPT2": {"models": gpt2_models, "prefix": ""},
    "Falcon": {"models": falcon_models, "prefix": "tiiuae/"},
    "Llama3": {"models": llama3_models, "prefix": "meta-llama/"},
    "Gemma3": {"models": gemma3_models, "prefix": "google/"},   
}


def process_dataset_yesno(input_file, output_file, plain_file, impl, model_name, model, 
                          model_family, tokenizer, domain, calib_count=80, batch_size=8, 
                          dataset=None, prompt=None, calib_data=None, batch_size_param=None):
    
    print("impl: ", impl)
    config = YESNO_CONFIG

    # Check if plain results exist for correction implementations
    if impl in ["yesnospecific", "yesnobos", "yesnocontextcalib", "yesnobatchcalib"]:
        yesnoplain_output_file = plain_file
        
        if os.path.exists(yesnoplain_output_file):
            print(f"Found existing plain results: {yesnoplain_output_file}")
            df = pd.read_csv(yesnoplain_output_file)
            
            # Apply the requested correction using unified functions
            if impl == "yesnospecific":
                print("Applying specific correction to existing plain results...")
                df = do_specific_correction(df, dataset, calib_count, calib_data, config)
                df.to_csv(output_file, index=False)
                print(f"specific corrected results saved to {output_file}")
                return
            
            elif impl == "yesnobos":
                print("Applying BOS correction to existing plain results...")
                bos_df = pd.read_csv(config.bos_csv_file)
                row = bos_df[bos_df['model_name'] == model_name]
                if len(row) == 0:
                    full_model_name = f"{MODEL_CONFIGS[model_family]['prefix']}{model_name}"
                    row = bos_df[bos_df['model_name'] == full_model_name]
                    if len(row) == 0:
                        print(f"ERROR: Model {model_name} not found in BOS data")
                        return
                
                bos_values = [row['bos_yes_logprob'].iloc[0], row['bos_no_logprob'].iloc[0]]
                df = do_bos_correction(df, bos_values, config)
                df.to_csv(output_file, index=False)
                print(f"BOS corrected results saved to {output_file}")
                return
            
            elif impl == "yesnocontextcalib":
                print("Applying contextual calibration to existing plain results...")
                df = do_contextcalib_correction(df, model, tokenizer, dataset, prompt, model_name, config)
                df.to_csv(output_file, index=False)
                print(f"Contextual calibration results saved to {output_file}")
                return

            elif impl == "yesnobatchcalib":
                print("Applying batch calibration to existing plain results...")
                df = do_batchcalib_correction(df, batch_size=batch_size_param, config=config)
                df.to_csv(output_file, index=False)
                print(f"Batch calibration results saved to {output_file}")
                return
        
        else:
            if model is None and tokenizer is None:
                print(f"ERROR: Expected plain results at {yesnoplain_output_file} but file doesn't exist!")
                return
            else:
                print(f"Plain results don't exist. Will run full inference first.")

    # INFERENCE MODE
    if model is None and tokenizer is None:
        print("ERROR: Cannot run inference without model and tokenizer!")
        return
    
    print(f"Running inference for {model_name}")
    
    # Load data
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
            tokenizer, model, dataset, prompt, 
            content_free_mode=False, content_free_input=None
        )

        for j, result in enumerate(batch_results):
            if i + j >= len(df): break
            idx = i + j
            
            for key, value in result.items():
                df.loc[idx, key] = value

            df.loc[idx, 'predicted_answer'] = "Yes" if df.loc[idx, 'yes_logprob'] > df.loc[idx, 'no_logprob'] else "No"
            df.loc[idx, 'is_correct'] = df.loc[idx, 'predicted_answer'] == df.loc[idx, 'Correct Answer']
        
        if i % (batch_size * 5) == 0:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if i % (batch_size * 20) == 0:
                gc.collect()

    # Save plain results
    if impl in ["yesnospecific", "yesnobos", "yesnocontextcalib", "yesnobatchcalib"]:
        plain_output_file = Path(plain_file)
        plain_output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(plain_file, index=False)
        print(f"Plain results saved to {plain_file}")

    # Apply corrections if needed using unified functions
    if impl == "yesnospecific":
        print("Applying specific correction to fresh inference results...")
        df = do_specific_correction(df, dataset, calib_count, calib_data, config)
    elif impl == "yesnobos":
        print("Applying BOS correction to fresh inference results...")
        bos_df = pd.read_csv(config.bos_csv_file)
        row = bos_df[bos_df['model_name'] == model_name]
        if len(row) == 0:
            full_model_name = f"{MODEL_CONFIGS[model_family]['prefix']}{model_name}"
            row = bos_df[bos_df['model_name'] == full_model_name]
        bos_values = [row['bos_yes_logprob'].iloc[0], row['bos_no_logprob'].iloc[0]]
        df = do_bos_correction(df, bos_values, config)
    elif impl == "yesnocontextcalib":
        print("Applying contextual calibration to fresh inference results...")
        df = do_contextcalib_correction(df, model, tokenizer, dataset, prompt, model_name, config)
    elif impl == "yesnobatchcalib":
        print("Applying batch calibration to fresh inference results...")
        df = do_batchcalib_correction(df, batch_size=batch_size_param, config=config)
    
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")