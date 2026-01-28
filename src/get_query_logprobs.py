# this is get_query_logprobs.py:

import torch
from dataset_prompts import get_dataset_prompts

def get_query_logprobs(model, query_input_ids):
    with torch.no_grad():
        device = next(model.parameters()).device
        outputs = model(query_input_ids.to(device))
        log_probs = torch.log_softmax(outputs.logits[:, -2, :], dim=-1) # Get logprobs for second-to-last token (just before the appended answer token)
        # logits dimension: (batch_size, sequence_length, vocab_size)
        # so ':' means all batch_size, '-2' means second-to-last token, ':' means all vocab_size
        
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
    
    # Process each context-question pair
    for ctx, q in zip(contexts, questions):
        yes_logprobs = []
        no_logprobs = []
        
        # Process variants
        for variant in all_variants:
            if dataset == "COMPS" or dataset == "ARITH":
                query = ''.join(filter(None, [q.strip()]))
            elif dataset == "EWOK" or dataset == "BABI":
                query = ' '.join(filter(None, [ctx.strip(), q.strip()]))
            
            if prompt == "fewshot":
                query = f"{prompt_cond1 + query}\nResponse:{variant}"
            elif prompt == "instronly":
                query = f"{prompt_cond2 + query}\nResponse:{variant}"
            elif prompt == "zeroshot":       # <--- query == context_free_input for yesno when we calculate precomputed contextcalib vals
                query = query + variant

            print(query)

            input_ids = tokenizer([query], return_tensors="pt", padding=True, truncation=True)
            logprob = get_query_logprobs(model, input_ids['input_ids'])[0]
            if variant in yes_variants: 
                yes_logprobs.append(logprob)
            else:
                no_logprobs.append(logprob)
            del input_ids
        
        # Log-sum-exp for both yes and no variants
        yes_tensor = torch.tensor(yes_logprobs)
        no_tensor = torch.tensor(no_logprobs)
        
        yes_combined = torch.logsumexp(yes_tensor, dim=0).item()
        no_combined = torch.logsumexp(no_tensor, dim=0).item()
        
        results.append({
            'yes_logprob': yes_combined,
            'no_logprob': no_combined,
        })
    
    return results



def calculate_logprobs_batch_mcq(df, tokenizer, model, dataset=None, prompt=None, content_free_mode=False, content_free_input="N/A"):

    prompt_cond1, prompt_cond2 = get_dataset_prompts(dataset, content_free_mode, content_free_input)
    results = []
        
    oa_variants = ["A", " A"]
    ob_variants = ["B", " B"]
    oc_variants = ["C", " C"]
    od_variants = ["D", " D"]

    all_variants = oa_variants + ob_variants + oc_variants + od_variants
    # print(f"all_variants: {all_variants}")
    
    # iterate through each row
    for i in range(len(df)):

        oa_logprobs = []
        ob_logprobs = []
        oc_logprobs = []
        od_logprobs = []
        
        row = df.iloc[i]
        question = row['question']
        option_a = row['A']
        option_b = row['B']
        option_c = row['C']
        option_d = row['D']

        og_query = f"{question}\nOptions: {option_a}, {option_b}, {option_c}, {option_d}"

        for variant in all_variants:
            if prompt == "fewshot":
                query = f"{prompt_cond1 + og_query}\nResponse:{variant}"
            elif prompt == "instronly":
                query = f"{prompt_cond2 + og_query}\nResponse:{variant}"
            elif prompt == "zeroshot":
                query = f"{og_query}\nResponse:{variant}"

            print(query)

            input_ids = tokenizer([query], return_tensors="pt", padding=True, truncation=True)
            logprob = get_query_logprobs(model, input_ids['input_ids'])[0]
            if variant in oa_variants:
                oa_logprobs.append(logprob)
            elif variant in ob_variants:
                ob_logprobs.append(logprob)
            elif variant in oc_variants:
                oc_logprobs.append(logprob)
            elif variant in od_variants:
                od_logprobs.append(logprob)
        
        oa_combined = torch.logsumexp(torch.tensor(oa_logprobs), dim=0).item()
        ob_combined = torch.logsumexp(torch.tensor(ob_logprobs), dim=0).item()
        oc_combined = torch.logsumexp(torch.tensor(oc_logprobs), dim=0).item()
        od_combined = torch.logsumexp(torch.tensor(od_logprobs), dim=0).item()
        
        results.append({
            'oa_logprob': oa_combined,
            'ob_logprob': ob_combined,
            'oc_logprob': oc_combined,
            'od_logprob': od_combined
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

        o0_logprobs = []
        o1_logprobs = []
        o2_logprobs = []

        og_query = df.iloc[i]['Question']

        for variant in all_variants:
            if prompt == "fewshot":
                query = f"{prompt_cond1 + og_query}\nResponse:{variant}"
            elif prompt == "instronly":
                query = f"{prompt_cond2 + og_query}\nResponse:{variant}"

            # print(query)

            input_ids = tokenizer([query], return_tensors="pt", padding=True, truncation=True)
            logprob = get_query_logprobs(model, input_ids['input_ids'])[0]
            if variant in o0_variants:
                o0_logprobs.append(logprob)
            elif variant in o1_variants:
                o1_logprobs.append(logprob)
            elif variant in o2_variants:
                o2_logprobs.append(logprob)

        o0_combined = torch.logsumexp(torch.tensor(o0_logprobs), dim=0).item()
        o1_combined = torch.logsumexp(torch.tensor(o1_logprobs), dim=0).item()
        o2_combined = torch.logsumexp(torch.tensor(o2_logprobs), dim=0).item()

        results.append({
            'o0_logprob': o0_combined,
            'o1_logprob': o1_combined,
            'o2_logprob': o2_combined
        })

    return results
