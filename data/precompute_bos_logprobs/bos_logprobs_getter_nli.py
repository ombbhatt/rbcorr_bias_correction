import torch
import pandas as pd
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, Gemma3ForConditionalGeneration
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.get_query_logprobs import get_query_logprobs

# Model configurations
falcon_models = ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"]
llama3_models = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"]
gemma3_models = ["gemma-3-12b-pt", "gemma-3-12b-it", "gemma-3-27b-pt", "gemma-3-27b-it"]

MODEL_CONFIGS = {
    "Falcon": {"models": falcon_models, "model_class": AutoModelForCausalLM, "tokenizer_class": AutoTokenizer, "prefix": "tiiuae/"},
    "Llama3": {"models": llama3_models, "model_class": LlamaForCausalLM, "tokenizer_class": AutoTokenizer, "prefix": "meta-llama/"},
    "Gemma3": {"models": gemma3_models, "model_class": Gemma3ForConditionalGeneration, "tokenizer_class": AutoTokenizer, "prefix": "google/"},
}

def setup_model_and_tokenizer(model_name, model_family):
    config = MODEL_CONFIGS[model_family]
    full_model_name = f"{config['prefix']}{model_name}"
    is_large_model = any(x in model_name.lower() for x in ['7b', '8b', '9b', "10b", "12b", "13b", "14b", "27b", "30b", "32b", "40b", "70b"])
    
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if is_large_model else None,
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

def get_bos_logprobs_v2(model, tokenizer):
    # Get the log probabilities for the BOS token for each option 0, 1, 2. If the tokenizer does not have a BOS token, use the PAD or EOS token or an empty string as a fallback.
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else \
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else \
        tokenizer.eos_token_id if tokenizer.eos_token_id is not None else \
        tokenizer.convert_tokens_to_ids("")

    o0_queries = [f"{tokenizer.decode(bos_token_id)}0", f"{tokenizer.decode(bos_token_id)} 0"]
    o1_queries = [f"{tokenizer.decode(bos_token_id)}1", f"{tokenizer.decode(bos_token_id)} 1"]
    o2_queries = [f"{tokenizer.decode(bos_token_id)}2", f"{tokenizer.decode(bos_token_id)} 2"]


    o0_logprobs = []
    o1_logprobs = []
    o2_logprobs = []


    for query in o0_queries:
        input_ids = tokenizer([query], return_tensors="pt", padding=True, truncation=True)
        logprob = get_query_logprobs(model, input_ids['input_ids'])[0]
        o0_logprobs.append(logprob)

    for query in o1_queries:
        input_ids = tokenizer([query], return_tensors="pt", padding=True, truncation=True)
        logprob = get_query_logprobs(model, input_ids['input_ids'])[0]
        o1_logprobs.append(logprob)

    for query in o2_queries:
        input_ids = tokenizer([query], return_tensors="pt", padding=True, truncation=True)
        logprob = get_query_logprobs(model, input_ids['input_ids'])[0]
        o2_logprobs.append(logprob)


    bos_o0_logprob = torch.logsumexp(torch.tensor(o0_logprobs), dim=0).item()
    bos_o1_logprob = torch.logsumexp(torch.tensor(o1_logprobs), dim=0).item()
    bos_o2_logprob = torch.logsumexp(torch.tensor(o2_logprobs), dim=0).item()

    print(f"BOS logprobs:")
    print(f"0: {bos_o0_logprob}, 1: {bos_o1_logprob}, 2: {bos_o2_logprob}")
    return bos_o0_logprob, bos_o1_logprob, bos_o2_logprob

# Main execution
results = []
total_models = sum(len(config["models"]) for config in MODEL_CONFIGS.values())
current_model = 0

print(f"Starting BOS logprob generation (0,1,2) for {total_models} models...")
print("=" * 60)

for family_name, config in MODEL_CONFIGS.items():
    print(f"\nProcessing {family_name} family ({len(config['models'])} models)")
    print("-" * 40)
    
    for model_name in config["models"]:
        current_model += 1
        print(f"\n[{current_model}/{total_models}] Processing {family_name}/{model_name}...")
        
        try:
            model, tokenizer = setup_model_and_tokenizer(model_name, family_name)
            print("✓ Model and tokenizer loaded successfully")
            
            print("Computing BOS logprobs for 0, 1, 2...")
            bos_o0_logprob, bos_o1_logprob, bos_o2_logprob = get_bos_logprobs_v2(model, tokenizer)
            
            results.append({
                'model_family': family_name,
                'model_name': model_name,
                'bos_0_logprob': bos_o0_logprob,
                'bos_1_logprob': bos_o1_logprob,
                'bos_2_logprob': bos_o2_logprob
            })
            
            print(f"Completed {model_name} successfully!")
            print(f"   0: {bos_o0_logprob:.4f}, 1: {bos_o1_logprob:.4f}, 2: {bos_o2_logprob:.4f}")

            # Clean up memory
            print("Cleaning up memory...")
            del model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            print(f"   Skipping and continuing...")

print("\n" + "=" * 60)
print(f"Processing complete! Processed {len(results)} models successfully.")

# Save to CSV
print("\nSaving results to CSV...")
df = pd.DataFrame(results)
df.to_csv('bos_logprobs_nli.csv', index=False)
print(f"Saved results to bos_logprobs_nli.csv")

print(f"\nFinal Results Summary:")
print(f"   Total models attempted: {total_models}")
print(f"   Successfully processed: {len(results)}")
print(f"   Failed: {total_models - len(results)}")

print(f"\nPreview of results:")
print(df)