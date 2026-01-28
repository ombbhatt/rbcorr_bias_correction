# this is unified_transprompt_transmodel_transdata.py

import argparse, os, gc, torch, sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig, Gemma3ForConditionalGeneration
from torch.cuda import empty_cache
import pandas as pd
import threading
import time

def periodic_cache_clear(interval=300):  # Every 5 minutes
    while True:
        time.sleep(interval)
        torch.cuda.empty_cache()
        print(f"Cleared GPU cache at {time.strftime('%H:%M:%S')}")

gc.collect()
torch.cuda.empty_cache()

# Import implementations
from yn_BOS_BC_CC import process_dataset_yesno
from mcq_BOS_BC_CC import process_dataset_mcq
from nli_BOC_CC_BC import process_dataset_nli
DATE = "Sep-16-2025"

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

EWOK_DOMAINS = [
    "social_interactions", "social_properties", "material_dynamics", "social_relations", "quantitative_properties", "physical_dynamics", "physical_interactions", "material_properties", "physical_relations", "spatial_relations", "agent_properties", "all_domains"
    ]

COMPS_DOMAINS = ["comps"]
BABI_DOMAINS = ["babi"]
ARITH_DOMAINS = ["arith"]
SNLI_DOMAINS = ["snli"]
MNLI_DOMAINS = ["mnli"]

DATASET_PATHS = {
    "EWOK": Path("../data/ewokynq-scripts-data/t2q_nodup_nominpairs"),
    "COMPS": Path("../data/compsynq-scripts-data"),
    "BABI": Path("../data/babiynq-scripts-data"),
    "ARITH": Path("../data/arithynq-scripts-data"),
    "MMLU-STEM": Path("../data/mmlu-scripts-data"),
    "MMLU-HUMANITIES": Path("../data/mmlu-scripts-data"),
    "MMLU-SOCIAL_SCI": Path("../data/mmlu-scripts-data"),
    "MMLU-OTHERS": Path("../data/mmlu-scripts-data"),
    "SNLI": Path("../data/snli-scripts-data"),
    "MNLI": Path("../data/mnli-scripts-data")   
}

DATASET_DOMAINS = {
    "EWOK": EWOK_DOMAINS,
    "COMPS": COMPS_DOMAINS,
    "BABI": BABI_DOMAINS,
    "ARITH": ARITH_DOMAINS,
    "MMLU-STEM": ["STEM"],
    "MMLU-HUMANITIES": ["HUMANITIES"],
    "MMLU-SOCIAL_SCI": ["SOCIAL_SCI"],
    "MMLU-OTHERS": ["OTHERS"],
    "SNLI": SNLI_DOMAINS,
    "MNLI": MNLI_DOMAINS
}

IMPLEMENTATIONS = {
    "yesnoplain": process_dataset_yesno,
    "yesnospecific": process_dataset_yesno,
    "yesnobos": process_dataset_yesno,
    "yesnocontextcalib": process_dataset_yesno,
    "yesnobatchcalib": process_dataset_yesno,  # Add this line
    "mcqplain": process_dataset_mcq,
    "mcqspecific": process_dataset_mcq,
    "mcqbos": process_dataset_mcq,
    "mcqcontextcalib": process_dataset_mcq,
    "mcqbatchcalib": process_dataset_mcq,  # Add this line
    "nliplain": process_dataset_nli,
    "nlispecific": process_dataset_nli,
    "nlibos": process_dataset_nli,
    "nlicontextcalib": process_dataset_nli,
    "nlibatchcalib": process_dataset_nli,  # Add this line
}


def infer_model_family(model_name: str) -> str:
    """Infer model family from model name"""
    if model_name.startswith("Falcon3"):
        return "Falcon"
    elif model_name.startswith("gemma-3"):
        return "Gemma3"
    elif model_name.startswith("Qwen2.5"):
        return "Qwen2"
    elif model_name.startswith("Llama-2"):
        return "Llama"
    elif model_name.startswith("Llama-3"):
        return "Llama3"
    else:
        raise ValueError(f"Cannot infer model family from model name: {model_name}")


def combine_mmlu_domain(dataset_path, domain_name):
    """
    Combine all CSV files from a domain folder into a single DataFrame.
    
    Args:
        domain_name (str): Domain folder name ('STEM', 'SOCIAL_SCI', 'HUMANITIES', 'OTHERS')
    
    Returns:
        pd.DataFrame: Combined DataFrame with all CSV data
    """
    # Define the path to your domain folders here
    domain_folder = dataset_path / domain_name
    
    # Get all CSV files in the domain folder
    csv_files = list(domain_folder.glob("*.csv"))
    
    # Read and combine all CSV files
    dataframes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dataframes.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    return combined_df

def load_calibration_data(calib_dataset, calib_model_name, calib_model_family, calib_prompt, target_dataset=None):
    """Load plain results from calibration dataset AND model for cross-dataset/model correction"""
    
    if is_mmlu_dataset(calib_dataset):
        # For MMLU domains, load the combined domain data
        mmlu_domain = extract_mmlu_domain(calib_dataset)
        calib_path = Path(f"../outputs/{DATE}/{calib_prompt}/MMLU/mcqplain/{calib_model_family}/{mmlu_domain}/{calib_model_name}_results.csv")
        if calib_path.exists():
            df = pd.read_csv(calib_path)
            print(f"Loaded {len(df)} calibration samples from {calib_dataset} using {calib_model_name}")
            return df
        else:
            raise FileNotFoundError(f"No calibration data found for {calib_dataset} at {calib_path}")
        
    else:
        if calib_dataset == "EWOK":
            domains = ["all_domains"]
        else:
            domains = DATASET_DOMAINS[calib_dataset]
    
        # Load and combine all domain data for calibration dataset
        all_calib_data = []
        for domain in domains:
            if calib_dataset != "SNLI" and calib_dataset != "MNLI":
                calib_path = Path(f"../outputs/{DATE}/{calib_prompt}/{calib_dataset}/yesnoplain/{calib_model_family}/{domain}/{calib_model_name}_results.csv")
            else:
                calib_path = Path(f"../outputs/{DATE}/{calib_prompt}/{calib_dataset}/nliplain/{calib_model_family}/{domain}/{calib_model_name}_results.csv")
            if calib_path.exists():
                df = pd.read_csv(calib_path)
                all_calib_data.append(df)
            else:
                print(f"Warning: Could not load calibration data from {calib_path}: {e}")

        if all_calib_data:
            combined_df = pd.concat(all_calib_data, ignore_index=True)
            print(f"Loaded {len(combined_df)} calibration samples from {calib_dataset} using {calib_model_name}")
            return combined_df
        else:
            raise FileNotFoundError(f"No calibration data found for {calib_dataset} with {calib_model_name}")
        

def extract_mmlu_domain(dataset_name):
    """Extract MMLU domain from dataset name like 'MMLU-STEM' -> 'STEM'"""
    if dataset_name.startswith("MMLU-"):
        return dataset_name.split("-", 1)[1]
    return None

def is_mmlu_dataset(dataset_name):
    """Check if dataset is an MMLU domain"""
    return dataset_name.startswith("MMLU-")


def setup_model_and_tokenizer(model_name, model_family):
    config = MODEL_CONFIGS[model_family]
    full_model_name = f"{config['prefix']}{model_name}"
    
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True
    }
    
    print(f"Loading model {full_model_name}...")
    model = config['model_class'].from_pretrained(
            full_model_name, **model_kwargs, 
            token=config.get('token'), 
            **({"load_in_4bit" : True} if "70B" in full_model_name else {})
        )
            
    print("Loading tokenizer...")
    tokenizer = config['tokenizer_class'].from_pretrained(
        full_model_name,
        padding_side='left',
        truncation=True,
        token=config.get('token')
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    
    return model, tokenizer

def process_single_output(impl, model, tokenizer, model_name, model_family, implementation_fn, 
                         input_file, output_file, plain_file, domain, calib_count=80, batch_size=4, 
                         target_dataset=None, calib_dataset=None, calib_prompt=None, target_prompt=None, calib_data=None, force=False):
    if output_file.exists() and not force:  # CHANGED: Check force flag
        print(f"Results exist for {model_name}, skipping...")
        return
    
    if output_file.exists() and force:  # ADDED: Log when forcing re-run
        print(f"Results exist for {model_name}, but forcing re-run...")
        
    kwargs = {
        "impl": impl,
        "model_name": model_name,
        "model": model,
        "model_family": model_family,
        "tokenizer": tokenizer,
        "domain": domain,
        "batch_size": batch_size,
        "dataset": "MMLU" if is_mmlu_dataset(target_dataset) else target_dataset,
        "prompt": target_prompt
    }
    
    # Add specific parameters based on implementation
    if "specific" in impl:
        kwargs["calib_count"] = calib_count
        if calib_data is not None:
            kwargs["calib_data"] = calib_data
    elif "batchcalib" in impl:
        # For batch calibration, calib_count acts as batch_size
        kwargs["batch_size_param"] = calib_count
    
    implementation_fn(input_file, output_file, plain_file, **kwargs)
    if model is not None:
        if hasattr(model, 'clear_kv_cache'):
            model.clear_kv_cache()
        elif hasattr(model, 'reset_kv_cache'):
            model.reset_kv_cache()


def check_all_outputs_exist(model_name, domains, output_base, force=False):
    """Check if all required output files exist for a model across all domains"""
    if force:
        return False  # Always return False when forcing, so we proceed with processing
    
    for domain in domains:
            output_file = output_base / domain / f"{model_name}_results.csv"
            if not output_file.exists():
                return False
    return True

def get_plain_path(model_family, domain, model_name, impl, target_dataset, target_prompt):
    # For MMLU domains, normalize to "MMLU" for plain results path
    plain_dataset = "MMLU" if is_mmlu_dataset(target_dataset) else target_dataset
    # plain_dataset = target_dataset
    
    # Reconstruct the plain path with normalized dataset name
    base_outputs = Path("../outputs") / DATE / target_prompt
    
    if "mcq" in impl:
        return base_outputs / plain_dataset / "mcqplain" / model_family / domain / f"{model_name}_results.csv"
    elif "yesno" in impl:
        return base_outputs / plain_dataset / "yesnoplain" / model_family / domain / f"{model_name}_results.csv"
    elif "nli" in impl:
        return base_outputs / plain_dataset / "nliplain" / model_family / domain / f"{model_name}_results.csv"


def check_corrections_possible(impl, model_name, model_family, domains, output_base, target_dataset, target_prompt, 
                               calib_dataset=None, calib_prompt=None, calib_model_name=None, calib_model_family=None):
    """Check if we can do corrections without loading model (all plain results exist)"""
    if impl not in ["mcqspecific", "mcqbos", "mcqcontextcalib", "mcqbatchcalib", "yesnospecific", "yesnobos", "yesnocontextcalib", "yesnobatchcalib", "nlispecific", "nlibos", "nlicontextcalib", "nlibatchcalib"]:
        return False
    
    # Check if target plain results exist
    print(f"Checking if all target plain results exist for {model_name}...")
    for domain in domains:
        plain_file = get_plain_path(model_family, domain, model_name, impl, target_dataset, target_prompt)
        if not plain_file.exists():
            print(f"Target plain results missing for domain {domain}: {plain_file}")
            return False
    
    print(f"✅ All target plain results exist for {model_name}")
    
    # For cross-dataset or cross-model or cross-prompt modes, also check calibration data exists
    cross_dataset_mode = (calib_dataset is not None and calib_dataset != target_dataset and "specific" in impl)
    cross_model_mode = (calib_model_name is not None and calib_model_name != model_name and "specific" in impl)
    cross_prompt_mode = (calib_prompt is not None and calib_prompt != target_prompt and "specific" in impl)
    
    if cross_dataset_mode or cross_model_mode or cross_prompt_mode:
        print(f"Checking if calibration data exists for cross-transfer...")
        
        # Determine which model/dataset/prompt to use for calibration
        actual_calib_model_name = calib_model_name if calib_model_name else model_name
        actual_calib_model_family = calib_model_family if calib_model_family else model_family
        actual_calib_dataset = calib_dataset if calib_dataset else target_dataset
        actual_calib_prompt = calib_prompt if calib_prompt else target_prompt
        
        try:
            # Try to load calibration data to verify it exists
            if is_mmlu_dataset(actual_calib_dataset):
                mmlu_domain = extract_mmlu_domain(actual_calib_dataset)
                calib_path = Path(f"../outputs/{DATE}/{actual_calib_prompt}/MMLU/mcqplain/{actual_calib_model_family}/{mmlu_domain}/{actual_calib_model_name}_results.csv")
                if not calib_path.exists():
                    print(f"Calibration data missing: {calib_path}")
                    return False
            else:
                # For yes-no datasets, check all domains
                if actual_calib_dataset == "EWOK":
                    calib_domains = ["all_domains"]
                else:
                    calib_domains = DATASET_DOMAINS[actual_calib_dataset]
                
                for calib_domain in calib_domains:
                    if actual_calib_dataset != "SNLI" and actual_calib_dataset != "MNLI":
                        calib_path = Path(f"../outputs/{DATE}/{actual_calib_prompt}/{actual_calib_dataset}/yesnoplain/{actual_calib_model_family}/{calib_domain}/{actual_calib_model_name}_results.csv")
                    else:
                        calib_path = Path(f"../outputs/{DATE}/{actual_calib_prompt}/{actual_calib_dataset}/nliplain/{actual_calib_model_family}/{calib_domain}/{actual_calib_model_name}_results.csv")
                    if not calib_path.exists():
                        print(f"Calibration data missing for domain {calib_domain}: {calib_path}")
                        return False
            
            print(f"✅ All calibration data exists")
            return True
            
        except Exception as e:
            print(f"Error checking calibration data: {e}")
            return False
    
    # For same-dataset/model/prompt, we only need target plain results
    return True

def process_model_across_domains(impl, target_model_name, target_model_family, implementation_fn, 
                                domains, dataset_path, output_base, calib_count=80, batch_size=4, 
                                target_dataset=None, calib_dataset=None, 
                                calib_model_name=None, calib_model_family=None, calib_prompt=None, target_prompt=None, force=False):  # CHANGED: Add force parameter
    """Process a single target model across all domains, optionally using different calibration model"""

    print(f"Output base: {output_base}")
    
    # Special handling for contextual calibration with zeroshot
    # if "contextcalib" in impl and calib_prompt == "zeroshot":
    #     print(f"Cannot do contextual calibration using zeroshot prompting on the calibration prompt, skipping...")
    #     return
    
    # Check if all outputs exist before loading model
    if check_all_outputs_exist(target_model_name, domains, output_base, force):  # CHANGED: Pass force
        print(f"All results exist for {target_model_name}, skipping...")
        return

    # Check if we can do correction-only (plain results exist)
    # can_do_corrections_only = check_corrections_possible(impl, target_model_name, target_model_family, 
    #                                                      domains, output_base, target_dataset, target_prompt)
    can_do_corrections_only = check_corrections_possible(
        impl, target_model_name, target_model_family, 
        domains, output_base, target_dataset, target_prompt,
        calib_dataset, calib_prompt, calib_model_name, calib_model_family  # ADD THESE PARAMETERS
    )
    
    # Determine if we're in cross-dataset or cross-model mode or cross-prompt mode
    cross_dataset_mode = (calib_dataset != target_dataset and "specific" in impl)
    cross_model_mode = (calib_model_name is not None and 
                       calib_model_name != target_model_name and 
                       "specific" in impl)
    cross_prompt_mode = (calib_prompt is not None and
                        calib_prompt != target_prompt and
                        "specific" in impl)

    
    # NEW LOGIC: Only load model if we actually need inference
    needs_model = not can_do_corrections_only
    
    # Load calibration data early if needed for cross-dataset/model corrections
    calib_data = None
    if (cross_dataset_mode or cross_model_mode or cross_prompt_mode) and can_do_corrections_only:
        print(f"Loading calibration data for cross-dataset/model corrections...")
        print(f"  Calib dataset: {calib_dataset}, Calib model: {calib_model_name}, Calib prompt: {calib_prompt}")
        print(f"  Target dataset: {target_dataset}, Target model: {target_model_name}, Target prompt: {target_prompt}")
        try:
            # Use calib_model if specified, otherwise use target_model
            actual_calib_model_name = calib_model_name if calib_model_name else target_model_name
            actual_calib_model_family = calib_model_family if calib_model_family else target_model_family
            
            calib_data = load_calibration_data(calib_dataset, actual_calib_model_name, 
                                             actual_calib_model_family, calib_prompt, target_dataset)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return
    
    if needs_model:
        print("Loading model for inference...")
        model, tokenizer = setup_model_and_tokenizer(target_model_name, target_model_family)
        
        # Load calibration data if needed and not already loaded
        if (cross_dataset_mode or cross_model_mode or cross_prompt_mode) and calib_data is None:
            try:
                actual_calib_model_name = calib_model_name if calib_model_name else target_model_name
                actual_calib_model_family = calib_model_family if calib_model_family else target_model_family
                
                calib_data = load_calibration_data(calib_dataset, actual_calib_model_name,
                                                  actual_calib_model_family, calib_prompt, target_dataset)
            except FileNotFoundError as e:
                print(f"ERROR: {e}")
                return
    else:
        print("All plain results exist, doing corrections without loading model...")
        model, tokenizer = None, None

    try:
        for domain in domains:
            print(f"\nProcessing domain: {domain}")
            print(f"Processing model: {target_model_name}")
            
            # Setup input file based on target dataset
            if target_dataset == "EWOK":
                input_file = dataset_path / f"processed_t2q_{domain}.csv"
            elif target_dataset == "COMPS":
                input_file = dataset_path / f"comps_yn_rand_2prop_2100.csv"
            elif target_dataset == "BABI":
                input_file = dataset_path / f"babi-ynq-big.csv"
            elif target_dataset == "ARITH":
                input_file = dataset_path / f"arith-ynq-big.csv"
            elif is_mmlu_dataset(target_dataset):
                mmlu_domain = extract_mmlu_domain(target_dataset)
                input_file = combine_mmlu_domain(dataset_path, mmlu_domain)
            elif target_dataset == "SNLI":
                input_file = dataset_path / f"snli_1.0" / "snli_1.0_test_balanced_smaller.csv"
            elif target_dataset == "MNLI":
                input_file = dataset_path / f"multinli_1.0" / "multinli_1.0_dev_matched_balanced_genre.csv"

            # Read file to get count for this domain
            if isinstance(input_file, pd.DataFrame):
                curr_input = input_file
            else:
                curr_input = pd.read_csv(input_file)
            curr_count = len(curr_input)
            
            if "specific" in impl or "batchcalib" in impl:
                # use full range of sizes only in same-condition mode
                if "specific" in impl and not (cross_dataset_mode or cross_model_mode or cross_prompt_mode):
                    calib_counts = [20, 50, 100, 500, 1000] # 0 signals to use full dataset
                elif "specific" in impl and (cross_dataset_mode or cross_model_mode or cross_prompt_mode):
                    calib_counts = [500] # only one size for cross-condition
                elif "batchcalib" in impl:
                    calib_counts = [20, 50, 100, 500, 1000]  # batch sizes to try

                for calib_count in calib_counts:
                    # if calib_count > 0 and calib_count > 0.5 * curr_count:
                    #     print("Reached calib_count more than half of total domain/dataset size, no more")
                    #     continue

                    if "specific" in impl:
                        if calib_count == 0:
                            print(f"  Running specific with full dataset as calibration")
                        else:
                            print(f"  Running specific with calib_count={calib_count} questions")
                    elif "batchcalib" in impl:
                        if calib_count == 0:
                            print(f"  Running batch calibration with full dataset as batch")
                        else:
                            print(f"  Running batch calibration with batch_size={calib_count}")

                    dataset_suffix = f"_from{calib_dataset}" if cross_dataset_mode else ""
                    prompt_suffix = f"_from{calib_prompt}" if cross_prompt_mode else ""

                    output_base_batchcalib = Path(f"../outputs/{DATE}/{target_prompt}{prompt_suffix}/{target_dataset}/{impl}_fixedcounts{dataset_suffix}/{target_model_family}")
                    output_base_specific = Path(f"../outputs/{DATE}/{target_prompt}{prompt_suffix}/{target_dataset}/{impl}_fixedcounts{dataset_suffix}_median/{target_model_family}")

                    # Build filename with cross-model suffix if applicable
                    model_suffix = f"_from{calib_model_name}" if cross_model_mode else ""

                    if "batchcalib" in impl:
                        base_suffix = output_base_batchcalib
                    elif "specific" in impl:
                        base_suffix = output_base_specific

                    if calib_count == 0:
                        # Full dataset batch calibration
                        output_file = base_suffix / domain / f"{target_model_name}{model_suffix}_fullbatch_results.csv"
                    else:
                        output_file = base_suffix / domain / f"{target_model_name}{model_suffix}_calib{calib_count}_results.csv"
                    
                    output_file.parent.mkdir(parents=True, exist_ok=True)

                    plain_file = get_plain_path(target_model_family, domain, target_model_name, impl, target_dataset, target_prompt)
                    
                    process_single_output(
                        impl, model, tokenizer, target_model_name, target_model_family, implementation_fn,
                        input_file, output_file, plain_file, domain, calib_count,
                        batch_size, target_dataset, calib_dataset, calib_prompt, target_prompt, calib_data, force  # CHANGED: Pass force
                    )
        
            else:
                output_file = output_base / domain / f"{target_model_name}_results.csv"
                output_file.parent.mkdir(parents=True, exist_ok=True)

                plain_file = get_plain_path(target_model_family, domain, target_model_name, impl, target_dataset, target_prompt)

                process_single_output(
                    impl, model, tokenizer, target_model_name, target_model_family, implementation_fn,
                    input_file, output_file, plain_file, domain, calib_count,
                    batch_size, target_dataset, calib_dataset, calib_prompt, target_prompt, calib_data, force  # CHANGED: Pass force
                )

    finally:
        # Clean up model after all domains are processed (only if we loaded it)
        if needs_model:
            del model, tokenizer
            try:
                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
            except RuntimeError:
                pass
            gc.collect()
          

def run_single_configuration(calib_dataset, target_dataset, target_model_name, target_model_family,
                            calib_model_name, calib_model_family, calib_prompt, target_prompt, implementation, 
                            batch_size, domain, calib_count, force=False):  # CHANGED: Add force parameter
    """Run a single configuration of calib_dataset, target_dataset, target_model, and calib_model"""
    
    print(f"\n{'='*80}")
    print(f"Running configuration:")
    print(f"  Calib Dataset: {calib_dataset}")
    print(f"  Target Dataset: {target_dataset}")
    print(f"  Target Model: {target_model_name} ({target_model_family})")
    if calib_model_name:
        print(f"  Calib Model: {calib_model_name} ({calib_model_family})")
    print(f"  Calib Prompt: {calib_prompt}")
    print(f"  Target Prompt: {target_prompt}")
    print(f"  Implementation: {implementation}")
    if force:  # ADDED: Log force mode
        print(f"  Force Mode: ENABLED (will overwrite existing results)")
    print(f"{'='*80}")
    
    # Validate datasets exist
    if calib_dataset not in DATASET_PATHS or target_dataset not in DATASET_PATHS:
        print(f"Invalid dataset specified")
        return

    # Setup target dataset path
    dataset_path = DATASET_PATHS[target_dataset]

    # Determine if we're working with MMLU domains
    if is_mmlu_dataset(calib_dataset) or is_mmlu_dataset(target_dataset):
        impl = "mcq" + implementation
        implementation_fn = IMPLEMENTATIONS[impl]
    else:
        if target_dataset == "SNLI" or calib_dataset == "SNLI" or target_dataset == "MNLI" or calib_dataset == "MNLI":
            impl = "nli" + implementation
        else:
            impl = "yesno" + implementation
        implementation_fn = IMPLEMENTATIONS[impl]
    
    # Determine cross-dataset and cross-model mode
    cross_dataset_mode = (calib_dataset != target_dataset)
    cross_model_mode = (calib_model_name is not None and calib_model_name != target_model_name)
    cross_prompt_mode = (calib_prompt != target_prompt)
    
    # Build dataset suffix
    dataset_suffix = f"_from{calib_dataset}" if cross_dataset_mode else ""
    prompt_suffix = f"_from{calib_prompt}" if cross_prompt_mode else ""
    
    if 'specific' in implementation:
        output_base = Path(f"../outputs/{DATE}/{target_prompt}{prompt_suffix}/{target_dataset}/{impl}{calib_count}calib{dataset_suffix}/{target_model_family}")
    elif any(x in implementation for x in ['bos', 'contextcalib', 'batchcalib', 'plain']):
        output_base = Path(f"../outputs/{DATE}/{target_prompt}{prompt_suffix}/{target_dataset}/{impl}{dataset_suffix}/{target_model_family}")

    # Setup target dataset domains
    if target_dataset == "EWOK":
        domains = ["all_domains"]
    else:
        domains = DATASET_DOMAINS[target_dataset]
    
    try:
        print(f"\nProcessing model: {target_model_name}")
        process_model_across_domains(
            impl,
            target_model_name=target_model_name,
            target_model_family=target_model_family,
            implementation_fn=implementation_fn,
            domains=domains,
            dataset_path=dataset_path,
            output_base=output_base,
            calib_count=calib_count,
            batch_size=batch_size,
            target_dataset=target_dataset,
            calib_dataset=calib_dataset,
            calib_model_name=calib_model_name,
            calib_model_family=calib_model_family,
            calib_prompt=calib_prompt,
            target_prompt=target_prompt,
            force=force  # CHANGED: Pass force
        )
                    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
        empty_cache()
        gc.collect()
        raise

def main():
    cache_thread = threading.Thread(target=periodic_cache_clear, daemon=True)
    cache_thread.start()
    
    parser = argparse.ArgumentParser(description="Run model inference across different models and domains")
    parser.add_argument("--calib_dataset", '-cd', type=str, choices=["EWOK", "COMPS", "BABI", "ARITH", "SNLI", "MNLI", "MMLU-STEM", "MMLU-HUMANITIES", "MMLU-SOCIAL_SCI", "MMLU-OTHERS"], required=True)
    parser.add_argument("--target_dataset", '-td', type=str, choices=["EWOK", "COMPS", "BABI", "ARITH", "SNLI", "MNLI", "MMLU-STEM", "MMLU-HUMANITIES", "MMLU-SOCIAL_SCI", "MMLU-OTHERS"], required=True)
    parser.add_argument("--target_model", '-tm', type=str, required=True,
                       help="Target model name (e.g., Falcon3-3B-Base). Model family will be inferred.")
    parser.add_argument("--calib_model", '-cm', type=str, required=False, default=None,
                       help="Calibration model name (e.g., Falcon3-10B-Instruct). Defaults to target_model if not specified.")
    parser.add_argument("--implementation", '-i', type=str, choices=["plain", "bos", "specific", "contextcalib", "batchcalib"], required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--domain", type=str, default='all')
    parser.add_argument("--calib_count", type=int, default=1)
    parser.add_argument("--target_prompt", '-tp', choices=["fewshot", "zeroshot", "instronly"], required=True)
    parser.add_argument("--calib_prompt", '-cp', choices=["fewshot", "zeroshot", "instronly"], required=True)
    parser.add_argument("--force", action="store_true",  # ADDED: Force flag
                       help="Force re-run even if results already exist (will overwrite existing files)")
    args = parser.parse_args()
    
    # Infer model families
    try:
        target_model_family = infer_model_family(args.target_model)
        print(f"Inferred target model family: {target_model_family}")
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Handle calibration model
    if args.calib_model:
        try:
            calib_model_family = infer_model_family(args.calib_model)
            print(f"Inferred calibration model family: {calib_model_family}")
            calib_model_name = args.calib_model
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    else:
        # Default to target model
        calib_model_name = args.target_model
        calib_model_family = target_model_family
        print(f"No calibration model specified, using target model: {calib_model_name}")
    
    try:
    
        run_single_configuration(
            calib_dataset=args.calib_dataset,
            target_dataset=args.target_dataset,
            target_model_name=args.target_model,
            target_model_family=target_model_family,
            calib_model_name=calib_model_name,
            calib_model_family=calib_model_family,
            calib_prompt=args.calib_prompt,
            target_prompt=args.target_prompt,
            implementation=args.implementation,
            batch_size=args.batch_size,
            domain=args.domain,
            calib_count=args.calib_count,
            force=args.force  # CHANGED: Pass force
        )
                    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
        empty_cache()
        gc.collect()
        sys.exit(1)

if __name__ == "__main__":
    main()