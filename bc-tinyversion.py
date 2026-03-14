import os
import numpy as np
import pandas as pd

# ── Dataset / path configuration ────────────────────────────────────────────

DATASET_CONFIG = {
    # yes-no
    "ARITH":            dict(task="yesno", task_plain="yesnoplain", task_corr="yesnobccorr", domain="arith",        answer_col="Correct Answer", options=["Yes","No"]),
    "BABI":             dict(task="yesno", task_plain="yesnoplain", task_corr="yesnobccorr", domain="babi",         answer_col="Correct Answer", options=["Yes","No"]),
    "COMPS":            dict(task="yesno", task_plain="yesnoplain", task_corr="yesnobccorr", domain="comps",        answer_col="Correct Answer", options=["Yes","No"]),
    "EWOK":             dict(task="yesno", task_plain="yesnoplain", task_corr="yesnobccorr", domain="all_domains",  answer_col="Correct Answer", options=["Yes","No"]),
    # nli
    "SNLI":             dict(task="nli",   task_plain="nliplain",   task_corr="nlibccorr",   domain="snli",         answer_col="Correct Answer", options=[0,1,2]),
    "MNLI":             dict(task="nli",   task_plain="nliplain",   task_corr="nlibccorr",   domain="mnli",         answer_col="Correct Answer", options=[0,1,2]),
    # mcq
    "MMLU-HUMANITIES":  dict(task="mcq",   task_plain="mcqplain",   task_corr="mcqbccorr",   domain="HUMANITIES",   answer_col="answer",         options=["A","B","C","D"]),
    "MMLU-OTHERS":      dict(task="mcq",   task_plain="mcqplain",   task_corr="mcqbccorr",   domain="OTHERS",       answer_col="answer",         options=["A","B","C","D"]),
    "MMLU-SOCIAL_SCI":  dict(task="mcq",   task_plain="mcqplain",   task_corr="mcqbccorr",   domain="SOCIAL_SCI",   answer_col="answer",         options=["A","B","C","D"]),
    "MMLU-STEM":        dict(task="mcq",   task_plain="mcqplain",   task_corr="mcqbccorr",   domain="STEM",         answer_col="answer",         options=["A","B","C","D"]),
}

MODEL_FAMILIES = {
    "Falcon": ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"],
    "Llama3": ["Llama-3.1-8B",    "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"],
    "Gemma3": ["gemma-3-27b-pt",  "gemma-3-27b-it",  "gemma-3-12b-pt", "gemma-3-12b-it"],
}

PROMPT_LEVELS = ["zeroshot", "instronly", "fewshot"]
NO_ZEROSHOT   = {"SNLI", "MNLI"}

LOGPROB_COLS = {
    "yesno": ["yes_logprob", "no_logprob"],
    "nli":   ["o0_logprob", "o1_logprob", "o2_logprob"],
    "mcq":   ["oa_logprob", "ob_logprob", "oc_logprob", "od_logprob"],
}

ROOT_IN  = "outputs/Mar-06-2026"
ROOT_OUT = "outputs/Mar-06-2026"  

BATCH_SIZE = 40
SEED       = 42

# ── Core correction logic ────────────────────────────────────────────────────

def bc_correct(df, task):
    """
    Batch-mean correction (BC).

    Steps:
      1. Shuffle the dataframe with a fixed seed (preserving original index)
      2. Split into sequential batches of BATCH_SIZE
      3. For each batch: subtract per-option mean (computed on that same batch)
      4. Restore original row order

    Output columns added:
      corrected_{col}             single corrected logprob value
      corrected_predicted_answer  argmax over corrected logprobs
      plain_predicted_answer      rename of original predicted_answer
    (is_correct is dropped)
    """
    lp_cols = LOGPROB_COLS[task]

    # Shuffle, preserving original index for later restoration
    shuffled = df.sample(frac=1, random_state=SEED)

    raw      = shuffled[lp_cols].values.astype(float)   # (200, n_options)
    n        = len(shuffled)
    corrected = np.empty_like(raw)

    for start in range(0, n, BATCH_SIZE):
        end        = start + BATCH_SIZE
        batch      = raw[start:end]
        batch_mean = batch.mean(axis=0)                 # (n_options,)
        corrected[start:end] = batch - batch_mean

    # Build output on shuffled frame, then restore original order
    out = shuffled.copy()

    if "predicted_answer" in out.columns:
        out.rename(columns={"predicted_answer": "plain_predicted_answer"}, inplace=True)
    if "is_correct" in out.columns:
        out.drop(columns=["is_correct"], inplace=True)

    for j, col in enumerate(lp_cols):
        out[f"corrected_{col}"] = corrected[:, j]

    # Corrected predicted answer
    best_idx = np.argmax(corrected, axis=1)
    if task == "yesno":
        label_map = {0: "Yes", 1: "No"}
    elif task == "nli":
        label_map = {0: 0, 1: 1, 2: 2}
    else:
        label_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    out["corrected_predicted_answer"] = [label_map[i] for i in best_idx]

    # Restore original row order
    out = out.sort_index()

    return out

# ── Path helpers ─────────────────────────────────────────────────────────────

def build_input_path(prompt_level, dataset_name, family, model, cfg):
    return os.path.join(
        ROOT_IN, prompt_level, dataset_name,
        cfg["task_plain"], family, cfg["domain"],
        f"{model}_results.csv"
    )

def build_output_path(prompt_level, dataset_name, family, model, cfg):
    return os.path.join(
        ROOT_OUT, prompt_level, dataset_name,
        cfg["task_corr"], family, cfg["domain"],
        f"{model}_results.csv"
    )

# ── Main pipeline ────────────────────────────────────────────────────────────

def run_correction_pipeline():
    processed, skipped = 0, 0

    for prompt_level in PROMPT_LEVELS:
        for dataset_name, cfg in DATASET_CONFIG.items():

            if prompt_level == "zeroshot" and dataset_name in NO_ZEROSHOT:
                continue

            for family, models in MODEL_FAMILIES.items():
                for model in models:

                    in_path  = build_input_path(prompt_level,  dataset_name, family, model, cfg)
                    out_path = build_output_path(prompt_level, dataset_name, family, model, cfg)

                    if not os.path.exists(in_path):
                        print(f"[SKIP] {in_path}")
                        skipped += 1
                        continue

                    df = pd.read_csv(in_path)

                    if cfg["task"] == "nli":
                        df[cfg["answer_col"]] = df[cfg["answer_col"]].astype(int)

                    out_df = bc_correct(df, cfg["task"])

                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    out_df.to_csv(out_path, index=False)
                    print(f"[OK]   {out_path}")
                    processed += 1

    print(f"\nDone. {processed} files corrected, {skipped} skipped.")

if __name__ == "__main__":
    run_correction_pipeline()