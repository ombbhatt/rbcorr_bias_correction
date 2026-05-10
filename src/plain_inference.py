import argparse, gc, sys, torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    BitsAndBytesConfig, Gemma3ForConditionalGeneration,
)

DATE = "May-10-2026"

MODEL_CONFIGS = {
    # "GPT2":   {"models": ["gpt2-medium", "gpt2-large"], "model_class": GPT2LMHeadModel, "tokenizer_class": GPT2Tokenizer,  "prefix": ""},
    "Falcon": {"models": ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"],"model_class": AutoModelForCausalLM,           "tokenizer_class": AutoTokenizer, "prefix": "tiiuae/"},
    "Llama3": {"models": ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"],"model_class": AutoModelForCausalLM, "tokenizer_class": AutoTokenizer, "prefix": "meta-llama/"},
    "Gemma3": {"models": ["gemma-3-27b-pt", "gemma-3-27b-it", "gemma-3-12b-pt", "gemma-3-12b-it"], "model_class": Gemma3ForConditionalGeneration, "tokenizer_class": AutoTokenizer, "prefix": "google/"},
}

DATASET_DOMAINS = {
    "EWOK":           ["all_domains"],
    "COMPS":          ["comps"],
    "BABI":           ["babi"],
    "ARITH":          ["arith"],
    "MMLU-STEM":      ["STEM"],
    "MMLU-HUMANITIES":["HUMANITIES"],
    "MMLU-SOCIAL_SCI":["SOCIAL_SCI"],
    "MMLU-OTHERS":    ["OTHERS"],
    "SNLI":           ["snli"],
    "MNLI":           ["mnli"],
}


# --- dataset prompts (from dataset_prompts.py) ---

def get_dataset_prompts(dataset, content_free_mode=False, content_free_input="N/A"):
    if dataset == "COMPS":
        prompt_cond1 = "#INSTRUCTIONS\nAnswer the following yes-no questions:\n\n#EXAMPLE\nQuestion: Does a blueberry fire bullets?\nResponse: No\n\n#EXAMPLE\nQuestion: Does a turtle have a hard shell?\nResponse: Yes\n\n#EXAMPLE\nQuestion: "
        prompt_cond2 = "#INSTRUCTIONS\nAnswer the following yes-no question:\n\n#EXAMPLE\nQuestion: "
    elif dataset == "EWOK":
        prompt_cond1 = "#INSTRUCTIONS\nAnswer the following yes-no questions:\n\n#EXAMPLE\nQuestion: Claire sees something that is fabric. Can Claire pour it?\nResponse: No\n\n#EXAMPLE\nQuestion: Sally pays salary to Harry. Is Sally Harry's boss?\nResponse: Yes\n\n#EXAMPLE\nQuestion: "
        prompt_cond2 = "#INSTRUCTIONS\nAnswer the following yes-no question:\n\n#EXAMPLE\nQuestion: "
    elif dataset == "BABI":
        prompt_cond1 = "#INSTRUCTIONS\nAnswer the following yes-no questions:\n\n#EXAMPLE\nQuestion: Marshall is in the car. Is Marshall in the building?\nResponse: No\n\n#EXAMPLE\nQuestion: Nathan is a pianist. Pianists like oranges. Does Nathan like oranges?\nResponse: Yes\n\n#EXAMPLE\nQuestion: "
        prompt_cond2 = "#INSTRUCTIONS\nAnswer the following yes-no question:\n\n#EXAMPLE\nQuestion: "
    elif dataset == "ARITH":
        prompt_cond1 = "#INSTRUCTIONS\nAnswer the following yes-no questions:\n\n#EXAMPLE\nQuestion: Is 7 minus 9 equal to 4?\nResponse: No\n\n#EXAMPLE\nQuestion: Is 17 plus 15 equal to 32?\nResponse: Yes\n\n#EXAMPLE\nQuestion: "
        prompt_cond2 = "#INSTRUCTIONS\nAnswer the following yes-no question:\n\n#EXAMPLE\nQuestion: "
    elif dataset == "MMLU":
        prompt_cond1 = "#INSTRUCTIONS\nAnswer the following multiple choice questions:\n\n#EXAMPLE\nQuestion: What is the shape of the Earth?\nOptions: (A) Cone, (B) Cube, (C) Sphere, (D) Cylinder\nResponse: C\n\n#EXAMPLE\nQuestion: What is the color of the sky?\nOptions: (A) Red, (B) Blue, (C) Green, (D) Yellow\nResponse: B\n\n#EXAMPLE\nQuestion: "
        prompt_cond2 = "#INSTRUCTIONS\nAnswer the following multiple choice question:\n\n#EXAMPLE\nQuestion: "
    elif dataset in ("SNLI", "MNLI"):
        prompt_cond1 = "#INSTRUCTIONS\nAnswer the following Recognizing Textual Entailment questions using a single digit. Entailment (0) implies the hypothesis is true given the premise. Neutral (1) implies the premise doesn't provide enough information to determine the hypothesis. Contradiction (2) implies the hypothesis is false given the premise.\n\n#EXAMPLE\:\nPremise: A man is playing a guitar. Hypothesis: A person is making music.\nResponse: 0\n\n#EXAMPLE\:\nPremise: A woman is reading a book in the library. Hypothesis: A woman is swimming. \nResponse: 2\n\n#EXAMPLE\:\n"
        prompt_cond2 = "#INSTRUCTIONS\nAnswer the following Recognizing Textual Entailment question using a single digit. Entailment (0) implies the hypothesis is true given the premise. Neutral (1) implies the premise doesn't provide enough information to determine the hypothesis. Contradiction (2) implies the hypothesis is false given the premise.\n\n#EXAMPLE\:\n"
    return prompt_cond1, prompt_cond2


# --- logprob helpers (from get_query_logprobs.py) ---

def get_query_logprobs(model, query_input_ids):
    with torch.no_grad():
        device = next(model.parameters()).device
        outputs = model(query_input_ids.to(device))
        log_probs = torch.log_softmax(outputs.logits[:, -2, :], dim=-1)
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized() and "Qwen" not in str(model) and "Falcon" not in str(model):
                torch.cuda.empty_cache()
        except RuntimeError:
            pass
        return [log_probs[0, query_input_ids[0, -1]].item()]


def calculate_logprobs_batch_yesno(contexts, questions, tokenizer, model, dataset=None, prompt=None, content_free_mode=False, content_free_input="N/A"):
    prompt_cond1, prompt_cond2 = get_dataset_prompts(dataset, content_free_mode, content_free_input)
    results = []

    yes_variants = [" Yes", "Yes"]
    no_variants = [" No", "No"]
    all_variants = yes_variants + no_variants

    for ctx, q in zip(contexts, questions):
        yes_logprobs = []
        no_logprobs = []

        for variant in all_variants:
            if dataset in ("COMPS", "ARITH"):
                query = ''.join(filter(None, [q.strip()]))
            elif dataset in ("EWOK", "BABI"):
                query = ' '.join(filter(None, [ctx.strip(), q.strip()]))

            if prompt == "fewshot":
                query = f"{prompt_cond1 + query}\nResponse:{variant}"
            elif prompt == "instronly":
                query = f"{prompt_cond2 + query}\nResponse:{variant}"
            elif prompt == "zeroshot":
                query = query + variant

            input_ids = tokenizer([query], return_tensors="pt", padding=True, truncation=True)
            logprob = get_query_logprobs(model, input_ids['input_ids'])[0]
            if variant in yes_variants:
                yes_logprobs.append(logprob)
            else:
                no_logprobs.append(logprob)
            del input_ids

        yes_combined = torch.logsumexp(torch.tensor(yes_logprobs), dim=0).item()
        no_combined  = torch.logsumexp(torch.tensor(no_logprobs),  dim=0).item()
        results.append({'yes_logprob': yes_combined, 'no_logprob': no_combined})

    return results


def calculate_logprobs_batch_mcq(df, tokenizer, model, dataset=None, prompt=None, content_free_mode=False, content_free_input="N/A"):
    prompt_cond1, prompt_cond2 = get_dataset_prompts(dataset, content_free_mode, content_free_input)
    results = []

    oa_variants = ["A", " A"]
    ob_variants = ["B", " B"]
    oc_variants = ["C", " C"]
    od_variants = ["D", " D"]
    all_variants = oa_variants + ob_variants + oc_variants + od_variants

    for i in range(len(df)):
        oa_logprobs, ob_logprobs, oc_logprobs, od_logprobs = [], [], [], []
        row = df.iloc[i]
        og_query = f"{row['question']}\nOptions: {row['A']}, {row['B']}, {row['C']}, {row['D']}"

        for variant in all_variants:
            if prompt == "fewshot":
                query = f"{prompt_cond1 + og_query}\nResponse:{variant}"
            elif prompt == "instronly":
                query = f"{prompt_cond2 + og_query}\nResponse:{variant}"
            elif prompt == "zeroshot":
                query = f"{og_query}\nResponse:{variant}"

            input_ids = tokenizer([query], return_tensors="pt", padding=True, truncation=True)
            logprob = get_query_logprobs(model, input_ids['input_ids'])[0]
            if variant in oa_variants:   oa_logprobs.append(logprob)
            elif variant in ob_variants: ob_logprobs.append(logprob)
            elif variant in oc_variants: oc_logprobs.append(logprob)
            elif variant in od_variants: od_logprobs.append(logprob)

        results.append({
            'oa_logprob': torch.logsumexp(torch.tensor(oa_logprobs), dim=0).item(),
            'ob_logprob': torch.logsumexp(torch.tensor(ob_logprobs), dim=0).item(),
            'oc_logprob': torch.logsumexp(torch.tensor(oc_logprobs), dim=0).item(),
            'od_logprob': torch.logsumexp(torch.tensor(od_logprobs), dim=0).item(),
        })

    return results


def calculate_logprobs_batch_nli(df, tokenizer, model, dataset=None, prompt=None, content_free_mode=False, content_free_input="N/A"):
    prompt_cond1, prompt_cond2 = get_dataset_prompts(dataset, content_free_mode, content_free_input)
    results = []

    o0_variants = [" 0", "0"]
    o1_variants = [" 1", "1"]
    o2_variants = [" 2", "2"]
    all_variants = o0_variants + o1_variants + o2_variants

    for i in range(len(df)):
        o0_logprobs, o1_logprobs, o2_logprobs = [], [], []
        og_query = df.iloc[i]['Question']

        for variant in all_variants:
            if prompt == "fewshot":
                query = f"{prompt_cond1 + og_query}\nResponse:{variant}"
            elif prompt == "instronly":
                query = f"{prompt_cond2 + og_query}\nResponse:{variant}"

            input_ids = tokenizer([query], return_tensors="pt", padding=True, truncation=True)
            logprob = get_query_logprobs(model, input_ids['input_ids'])[0]
            if variant in o0_variants:   o0_logprobs.append(logprob)
            elif variant in o1_variants: o1_logprobs.append(logprob)
            elif variant in o2_variants: o2_logprobs.append(logprob)

        results.append({
            'o0_logprob': torch.logsumexp(torch.tensor(o0_logprobs), dim=0).item(),
            'o1_logprob': torch.logsumexp(torch.tensor(o1_logprobs), dim=0).item(),
            'o2_logprob': torch.logsumexp(torch.tensor(o2_logprobs), dim=0).item(),
        })

    return results


# --- model utils ---

def is_mmlu_dataset(name):
    return name.startswith("MMLU-")

def extract_mmlu_domain(name):
    return name.split("-", 1)[1] if name.startswith("MMLU-") else None

def infer_model_family(model_name):
    if model_name.startswith("Falcon3"):    return "Falcon"
    if model_name.startswith("gemma-3"):    return "Gemma3"
    if model_name.startswith(("Meta-Llama-3", "Llama-3")): return "Llama3"
    if model_name.startswith("gpt2"):       return "GPT2"
    raise ValueError(f"Cannot infer model family from model name: {model_name}")

def setup_model_and_tokenizer(model_name, model_family):
    cfg = MODEL_CONFIGS[model_family]
    full_name = f"{cfg['prefix']}{model_name}"
    model_kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True}
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    print(f"Loading model {full_name}...")
    model = cfg['model_class'].from_pretrained(
        full_name, **model_kwargs,
        token=cfg.get('token'),
        **({"quantization_config": quantization_config} if "27b" in full_name else {}),
    )

    print("Loading tokenizer...")
    tokenizer = cfg['tokenizer_class'].from_pretrained(
        full_name, padding_side='left', truncation=True, token=cfg.get('token')
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return model, tokenizer


# --- plain inference runners ---

def run_yesnoplain(input_file, output_file, model_name, model, tokenizer, domain, dataset, prompt, batch_size):
    df = pd.read_csv(input_file, encoding='utf8') if not isinstance(input_file, pd.DataFrame) else input_file.copy()
    print(f"Starting yesnoplain for domain: {domain}, model: {model_name}")

    df['yes_logprob']      = None
    df['no_logprob']       = None
    df['predicted_answer'] = None
    df['is_correct']       = None

    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i + batch_size]
        batch_results = calculate_logprobs_batch_yesno(
            batch_df['Context'].tolist(), batch_df['Question'].tolist(),
            tokenizer, model, dataset, prompt, content_free_mode=False, content_free_input=None,
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

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def run_mcqplain(input_file, output_file, model_name, model, tokenizer, domain, dataset, prompt, batch_size):
    df = pd.read_csv(input_file, encoding='utf8') if not isinstance(input_file, pd.DataFrame) else input_file.copy()
    print(f"Starting mcqplain for domain: {domain}, model: {model_name}")

    df['oa_logprob']       = None
    df['ob_logprob']       = None
    df['oc_logprob']       = None
    df['od_logprob']       = None
    df['predicted_answer'] = None
    df['is_correct']       = None

    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i + batch_size]
        batch_results = calculate_logprobs_batch_mcq(
            batch_df, tokenizer, model, dataset, prompt, content_free_mode=False, content_free_input=None,
        )
        for j, result in enumerate(batch_results):
            if i + j >= len(df): break
            idx = i + j
            for key, value in result.items():
                df.loc[idx, key] = value
            logprobs = [float(df.loc[idx, c]) for c in ['oa_logprob', 'ob_logprob', 'oc_logprob', 'od_logprob']]
            df.loc[idx, 'predicted_answer'] = ['A', 'B', 'C', 'D'][np.argmax(logprobs)]
            df.loc[idx, 'is_correct'] = df.loc[idx, 'predicted_answer'] == df.loc[idx, 'answer']
        if i % (batch_size * 5) == 0:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if i % (batch_size * 20) == 0:
                gc.collect()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def run_nliplain(input_file, output_file, model_name, model, tokenizer, domain, dataset, prompt, batch_size):
    df = pd.read_csv(input_file, encoding='utf8') if not isinstance(input_file, pd.DataFrame) else input_file.copy()
    print(f"Starting nliplain for domain: {domain}, model: {model_name}")

    df['o0_logprob']       = None
    df['o1_logprob']       = None
    df['o2_logprob']       = None
    df['predicted_answer'] = None
    df['is_correct']       = None

    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i + batch_size]
        batch_results = calculate_logprobs_batch_nli(
            batch_df, tokenizer, model, dataset, prompt, content_free_mode=False, content_free_input=None,
        )
        for j, result in enumerate(batch_results):
            if i + j >= len(df): break
            idx = i + j
            for key, value in result.items():
                df.loc[idx, key] = value
            logprobs = [float(df.loc[idx, c]) for c in ['o0_logprob', 'o1_logprob', 'o2_logprob']]
            df.loc[idx, 'predicted_answer'] = ['0', '1', '2'][np.argmax(logprobs)]
            df.loc[idx, 'is_correct'] = df.loc[idx, 'predicted_answer'] == df.loc[idx, 'Correct Answer']
        if i % (batch_size * 5) == 0:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if i % (batch_size * 20) == 0:
                gc.collect()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run plain logprob inference for all models and datasets")
    parser.add_argument("--target_dataset", "-td", required=True,
                        choices=list(DATASET_DOMAINS.keys()))
    parser.add_argument("--target_model",   "-tm", required=True,
                        help="Model name (e.g. Llama-3.1-8B). Family is inferred automatically.")
    parser.add_argument("--target_prompt",  "-tp", required=True,
                        choices=["fewshot", "zeroshot", "instronly"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files")
    args = parser.parse_args()

    try:
        model_family = infer_model_family(args.target_model)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Inferred model family: {model_family}")

    # Determine impl type from dataset
    if is_mmlu_dataset(args.target_dataset):
        impl = "mcqplain"
        dataset_key = "MMLU"
    elif args.target_dataset in ("SNLI", "MNLI"):
        impl = "nliplain"
        dataset_key = args.target_dataset
    else:
        impl = "yesnoplain"
        dataset_key = args.target_dataset

    domains      = DATASET_DOMAINS[args.target_dataset]
    dataset_path = Path("../data/big_data")
    output_base  = Path(f"../outputs/{DATE}/{args.target_prompt}/{args.target_dataset}/{impl}/{model_family}")

    model, tokenizer = setup_model_and_tokenizer(args.target_model, model_family)

    try:
        for domain in domains:
            output_file = output_base / domain / f"{args.target_model}_results.csv"

            if output_file.exists() and not args.force:
                print(f"Results exist at {output_file}, skipping (use --force to overwrite)")
                continue

            # Resolve input file
            if args.target_dataset == "EWOK":
                input_file = dataset_path / "ewok-ynq-big.csv"
            elif args.target_dataset == "COMPS":
                input_file = dataset_path / "comps-ynq-big.csv"
            elif args.target_dataset == "BABI":
                input_file = dataset_path / "babi-ynq-big.csv"
            elif args.target_dataset == "ARITH":
                input_file = dataset_path / "arith-ynq-big.csv"
            elif is_mmlu_dataset(args.target_dataset):
                mmlu_domain = extract_mmlu_domain(args.target_dataset)
                input_file  = dataset_path / "mmlu-scripts-data" / mmlu_domain / f"{mmlu_domain}_sampled.csv"
            elif args.target_dataset == "SNLI":
                input_file = dataset_path / "snli-nli-balanced.csv"
            elif args.target_dataset == "MNLI":
                input_file = dataset_path / "multinli-nli-balanced.csv"

            print(f"\nProcessing domain: {domain} | model: {args.target_model}")

            if impl == "yesnoplain":
                run_yesnoplain(input_file, output_file, args.target_model, model, tokenizer,
                               domain, dataset_key, args.target_prompt, args.batch_size)
            elif impl == "mcqplain":
                run_mcqplain(input_file, output_file, args.target_model, model, tokenizer,
                             domain, dataset_key, args.target_prompt, args.batch_size)
            elif impl == "nliplain":
                run_nliplain(input_file, output_file, args.target_model, model, tokenizer,
                             domain, dataset_key, args.target_prompt, args.batch_size)

    finally:
        del model, tokenizer
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        except RuntimeError:
            pass
        gc.collect()


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()
