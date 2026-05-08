# CC CORRECTION

import os
import numpy as np
import pandas as pd

# ── Dataset / path configuration ────────────────────────────────────────────

DATASET_CONFIG = {
    # yes-no
    "ARITH":            dict(task="yesno", task_plain="yesnoplain", task_corr="yesnocccorr", domain="arith",        answer_col="Correct Answer", options=["Yes","No"]),
    "BABI":             dict(task="yesno", task_plain="yesnoplain", task_corr="yesnocccorr", domain="babi",         answer_col="Correct Answer", options=["Yes","No"]),
    "COMPS":            dict(task="yesno", task_plain="yesnoplain", task_corr="yesnocccorr", domain="comps",        answer_col="Correct Answer", options=["Yes","No"]),
    "EWOK":             dict(task="yesno", task_plain="yesnoplain", task_corr="yesnocccorr", domain="all_domains",  answer_col="Correct Answer", options=["Yes","No"]),
    # nli
    "SNLI":             dict(task="nli",   task_plain="nliplain",   task_corr="nlicccorr",   domain="snli",         answer_col="Correct Answer", options=[0,1,2]),
    "MNLI":             dict(task="nli",   task_plain="nliplain",   task_corr="nlicccorr",   domain="mnli",         answer_col="Correct Answer", options=[0,1,2]),
    # mcq
    "MMLU-HUMANITIES":  dict(task="mcq",   task_plain="mcqplain",   task_corr="mcqcccorr",   domain="HUMANITIES",   answer_col="answer",         options=["A","B","C","D"]),
    "MMLU-OTHERS":      dict(task="mcq",   task_plain="mcqplain",   task_corr="mcqcccorr",   domain="OTHERS",       answer_col="answer",         options=["A","B","C","D"]),
    "MMLU-SOCIAL_SCI":  dict(task="mcq",   task_plain="mcqplain",   task_corr="mcqcccorr",   domain="SOCIAL_SCI",   answer_col="answer",         options=["A","B","C","D"]),
    "MMLU-STEM":        dict(task="mcq",   task_plain="mcqplain",   task_corr="mcqcccorr",   domain="STEM",         answer_col="answer",         options=["A","B","C","D"]),
}

MODEL_FAMILIES = {
    "Falcon": ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"],
    "Llama3": ["Llama-3.1-8B",    "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"],
    "Gemma3": ["gemma-3-27b-pt",  "gemma-3-27b-it",  "gemma-3-12b-pt", "gemma-3-12b-it"],
}

PROMPT_LEVELS = ["zeroshot", "instronly", "fewshot"]
NO_ZEROSHOT   = {"SNLI", "MNLI"}

# Logprob column names per task, and their corresponding cf_* column names in the CF csv
LOGPROB_COLS = {
    "yesno": ["yes_logprob",  "no_logprob"],
    "nli":   ["o0_logprob",   "o1_logprob",  "o2_logprob"],
    "mcq":   ["oa_logprob",   "ob_logprob",  "oc_logprob",  "od_logprob"],
}

CF_COLS = {
    "yesno": ["cf_yes_prob",  "cf_no_prob"],
    "nli":   ["cf_o0_prob",   "cf_o1_prob",  "cf_o2_prob"],
    "mcq":   ["cf_oa_prob",   "cf_ob_prob",  "cf_oc_prob",  "cf_od_prob"],
}

ROOT_IN  = "outputs/Mar-23-2026"
ROOT_OUT = "outputs/Mar-23-2026"   # corrected folders sit alongside plain ones under the same root

# Paths to pre-computed content-free probabilities CSVs, one per task type
CF_CSV_PATHS = {
    "yesno": "data/precompute_cc_logprobs/contextfree_probs_yesno.csv",
    "nli":   "data/precompute_cc_logprobs/contextfree_probs_nli.csv",
    "mcq":   "data/precompute_cc_logprobs/contextfree_probs_mcq.csv",
}

# ── Load content-free lookup table ──────────────────────────────────────────

def load_cf_tables(cf_csv_paths):
    """
    Load one CF lookup table per task type.
    Returns a dict: task -> { (model_family, model_name, dataset, prompt) -> np.array }
    """
    cf_tables = {}
    for task, path in cf_csv_paths.items():
        cf_df = pd.read_csv(path)
        table = {}
        for _, row in cf_df.iterrows():
            key = (row["model_family"], row["model_name"], row["dataset"], row["prompt"])
            table[key] = np.array([row[c] for c in CF_COLS[task]], dtype=float)
        cf_tables[task] = table
    return cf_tables

# ── Core correction logic ────────────────────────────────────────────────────

def cc_correct(df, task, cf_probs):
    """
    Contextual Calibration (Zhao et al., 2021).

    For each item:
      1. Convert raw logprobs → probs via exp()
      2. Divide each option's prob by its content-free prob  (W = diag(p_cf)^-1)
      3. Store corrected values as log(corrected_prob) for consistency
      4. Corrected predicted answer = argmax over corrected probs

    cf_probs: np.array of shape (n_options,) — the pre-computed content-free
              probabilities for this model/dataset/prompt combination.

    Output columns added:
      corrected_{col}             log of the CC-corrected probability
      corrected_predicted_answer  argmax over corrected probs
      plain_predicted_answer      rename of original predicted_answer
    (is_correct is dropped)
    """
    lp_cols  = LOGPROB_COLS[task]
    raw_lp   = df[lp_cols].values.astype(float)   # (n, n_options)

    # Step 1: logprobs → probs
    raw_probs = np.exp(raw_lp)                     # (n, n_options)

    # Step 2: divide by content-free probs (broadcast across rows)
    corrected_probs = raw_probs / cf_probs[np.newaxis, :]   # (n, n_options)

    # Step 3: back to log space
    corrected_lp = np.log(corrected_probs)         # (n, n_options)

    # Build output dataframe
    out = df.copy()

    if "predicted_answer" in out.columns:
        out.rename(columns={"predicted_answer": "plain_predicted_answer"}, inplace=True)
    if "is_correct" in out.columns:
        out.drop(columns=["is_correct"], inplace=True)

    for j, col in enumerate(lp_cols):
        out[f"corrected_{col}"] = corrected_lp[:, j]

    # Step 4: corrected predicted answer
    best_idx = np.argmax(corrected_probs, axis=1)
    if task == "yesno":
        label_map = {0: "Yes", 1: "No"}
    elif task == "nli":
        label_map = {0: 0, 1: 1, 2: 2}
    else:
        label_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    out["corrected_predicted_answer"] = [label_map[i] for i in best_idx]

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
    cf_tables = load_cf_tables(CF_CSV_PATHS)
    processed, skipped, missing_cf = 0, 0, 0

    for prompt_level in PROMPT_LEVELS:
        for dataset_name, cfg in DATASET_CONFIG.items():

            if prompt_level == "zeroshot" and dataset_name in NO_ZEROSHOT:
                continue

            for family, models in MODEL_FAMILIES.items():
                for model in models:

                    in_path  = build_input_path(prompt_level,  dataset_name, family, model, cfg)
                    out_path = build_output_path(prompt_level, dataset_name, family, model, cfg)

                    if not os.path.exists(in_path):
                        print(f"[SKIP]    {in_path}")
                        skipped += 1
                        continue

                    cf_dataset = "MMLU" if cfg["task"] == "mcq" else dataset_name
                    cf_key = (family, model, cf_dataset, prompt_level)
                    if cf_key not in cf_tables[cfg["task"]]:
                        print(f"[NO CF]   {cf_key}")
                        missing_cf += 1
                        continue

                    df = pd.read_csv(in_path)

                    if cfg["task"] == "nli":
                        df[cfg["answer_col"]] = df[cfg["answer_col"]].astype(int)

                    out_df = cc_correct(df, cfg["task"], cf_tables[cfg["task"]][cf_key])

                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    out_df.to_csv(out_path, index=False)
                    print(f"[OK]      {out_path}")
                    processed += 1

    print(f"\nDone. {processed} corrected, {skipped} skipped (no input file), {missing_cf} skipped (no CF entry).")

if __name__ == "__main__":
    run_correction_pipeline()