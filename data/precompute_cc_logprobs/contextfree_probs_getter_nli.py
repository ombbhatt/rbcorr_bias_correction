# this is contextfree_probs_getter_mcq.py:

import torch
import pandas as pd
import numpy as np
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, Gemma3ForConditionalGeneration
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.plain_inference import calculate_logprobs_batch_nli

# Model configurations
falcon_models = ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"]
llama3_models = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"]
gemma3_models = ["gemma-3-12b-pt", "gemma-3-12b-it", "gemma-3-27b-pt", "gemma-3-27b-it"]

MODEL_CONFIGS = {
    "Llama3": {"models": llama3_models, "model_class": LlamaForCausalLM, "tokenizer_class": AutoTokenizer, "prefix": "meta-llama/"},
    "Falcon": {"models": falcon_models, "model_class": AutoModelForCausalLM, "tokenizer_class": AutoTokenizer, "prefix": "tiiuae/"},
    "Gemma3": {"models": gemma3_models, "model_class": Gemma3ForConditionalGeneration, "tokenizer_class": AutoTokenizer, "prefix": "google/"},
}

DATASETS = ["SNLI"]
PROMPTS = ["instronly", "fewshot"] 

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
        **({"load_in_4bit": True} if "70B" in full_model_name else {})
    )
            
    tokenizer = config['tokenizer_class'].from_pretrained(
        full_model_name,
        padding_side='left',
        truncation=True,
        token=config.get('token')
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    
    return model, tokenizer

def get_contentfree_probs_nli(model, tokenizer, dataset, prompt):
    """
    Get content-free probabilities for NLI using ensemble of N/A, [MASK], and empty string
    Matches the logic in do_entire_contextcalib_thing() for NLI
    """
    print(f"  Computing content-free probs for {dataset}/{prompt}...")
    
    # Get content-free probabilities using ensemble of three inputs
    cf_inputs = ["N/A", "[MASK]", ""]
    cf_results = []

    for cf_input in cf_inputs:
        # Create a dummy dataframe with content-free inputs
        sample_row = pd.DataFrame({
            'Context': '',
            'Question': '',
            'Correct Answer': ['0']  # Dummy answer (not used for content-free)
        })

        # Get content-free probabilities for this input
        cf_result = calculate_logprobs_batch_nli(
            sample_row, tokenizer, model, dataset, prompt,
            content_free_mode=True, content_free_input=cf_input
        )[0]
        # cf_result = calculate_logprobs_batch_nli(
        #     [""], [cf_input], tokenizer, model, dataset, prompt,
        #     content_free_mode=True, content_free_input=cf_input
        # )[0]
        cf_results.append(cf_result)

    # Average the logprobs from all three content-free inputs
    cf_o0_logprob = np.mean([result['o0_logprob'] for result in cf_results])
    cf_o1_logprob = np.mean([result['o1_logprob'] for result in cf_results])
    cf_o2_logprob = np.mean([result['o2_logprob'] for result in cf_results])

    # Convert to probabilities and normalize (matches contextual calibration logic)
    p_cf_o0 = np.exp(cf_o0_logprob)
    p_cf_o1 = np.exp(cf_o1_logprob)
    p_cf_o2 = np.exp(cf_o2_logprob)
    p_cf_total = p_cf_o0 + p_cf_o1 + p_cf_o2
    p_cf_o0 = p_cf_o0 / p_cf_total
    p_cf_o1 = p_cf_o1 / p_cf_total
    p_cf_o2 = p_cf_o2 / p_cf_total

    assert abs((p_cf_o0 + p_cf_o1 + p_cf_o2) - 1.0) < 1e-6, "Probs don't sum to 1!"  # MCQ

    print(f"    0={p_cf_o0:.4f}, 1={p_cf_o1:.4f}, 2={p_cf_o2:.4f}")

    return p_cf_o0, p_cf_o1, p_cf_o2

# Main execution
results = []
total_combinations = sum(len(config["models"]) for config in MODEL_CONFIGS.values()) * len(DATASETS) * len(PROMPTS)
current_combination = 0

print(f"Starting content-free probability generation for MCQ")
print(f"Total combinations: {total_combinations} (models × datasets × prompts)")
print("=" * 80)

for family_name, config in MODEL_CONFIGS.items():
    print(f"\n🔄 Processing {family_name} family ({len(config['models'])} models)")
    print("-" * 60)
    
    for model_name in config["models"]:
        print(f"\n📦 Loading {family_name}/{model_name}...")
        
        try:
            model, tokenizer = setup_model_and_tokenizer(model_name, family_name)
            print("✓ Model and tokenizer loaded successfully")
            
            # Process all dataset × prompt combinations for this model
            for dataset in DATASETS:
                for prompt in PROMPTS:
                    current_combination += 1
                    print(f"\n[{current_combination}/{total_combinations}] {dataset}/{prompt}")
                    
                    cf_o0_prob, cf_o1_prob, cf_o2_prob = get_contentfree_probs_nli(
                        model, tokenizer, dataset, prompt
                    )
                    
                    results.append({
                        'model_family': family_name,
                        'model_name': model_name,
                        'dataset': dataset,
                        'prompt': prompt,
                        'cf_o0_prob': cf_o0_prob,
                        'cf_o1_prob': cf_o1_prob,
                        'cf_o2_prob': cf_o2_prob
                    })
            
            print(f"\n✓ Completed all datasets/prompts for {model_name}")
            
            # Clean up memory
            print("Cleaning up memory...")
            del model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            # print full traceback for debugging
            import traceback
            traceback.print_exc()
            print(f"   Skipping and continuing...")

print("\n" + "=" * 80)
print(f"Processing complete! Generated {len(results)} content-free probability entries.")

# Save to CSV
print("\nSaving results to CSV...")
df = pd.DataFrame(results)
df.to_csv('contextfree_probs_nli.csv', index=False)
print(f"✓ Saved results to contextfree_probs_nli.csv")

print(f"\nFinal Results Summary:")
print(f"   Total combinations attempted: {total_combinations}")
print(f"   Successfully processed: {len(results)}")
print(f"   Failed: {total_combinations - len(results)}")

print(f"\nPreview of results:")
print(df.head(10))