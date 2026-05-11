# RB CORRECTION

import os
import numpy as np
import pandas as pd

# ── Dataset / path configuration ────────────────────────────────────────────

DATASET_CONFIG = {
    # yes-no
    "ARITH":            dict(task="yesno", task_plain="yesnoplain", task_corr="yesnorbcorr", domain="arith",        answer_col="Correct Answer", options=["Yes","No"]),
    "BABI":             dict(task="yesno", task_plain="yesnoplain", task_corr="yesnorbcorr", domain="babi",         answer_col="Correct Answer", options=["Yes","No"]),
    "COMPS":            dict(task="yesno", task_plain="yesnoplain", task_corr="yesnorbcorr", domain="comps",        answer_col="Correct Answer", options=["Yes","No"]),
    "EWOK":             dict(task="yesno", task_plain="yesnoplain", task_corr="yesnorbcorr", domain="all_domains",  answer_col="Correct Answer", options=["Yes","No"]),
    # nli
    "SNLI":             dict(task="nli",   task_plain="nliplain",   task_corr="nlirbcorr",   domain="snli",         answer_col="Correct Answer", options=[0,1,2]),
    "MNLI":             dict(task="nli",   task_plain="nliplain",   task_corr="nlirbcorr",   domain="mnli",         answer_col="Correct Answer", options=[0,1,2]),
    # mcq
    "MMLU-HUMANITIES":  dict(task="mcq",   task_plain="mcqplain",   task_corr="mcqrbcorr",   domain="HUMANITIES",   answer_col="answer",         options=["A","B","C","D"]),
    "MMLU-OTHERS":      dict(task="mcq",   task_plain="mcqplain",   task_corr="mcqrbcorr",   domain="OTHERS",       answer_col="answer",         options=["A","B","C","D"]),
    "MMLU-SOCIAL_SCI":  dict(task="mcq",   task_plain="mcqplain",   task_corr="mcqrbcorr",   domain="SOCIAL_SCI",   answer_col="answer",         options=["A","B","C","D"]),
    "MMLU-STEM":        dict(task="mcq",   task_plain="mcqplain",   task_corr="mcqrbcorr",   domain="STEM",         answer_col="answer",         options=["A","B","C","D"]),
}

MODEL_FAMILIES = {
    # "GPT2": ["gpt2-medium", "gpt2-large"],
    "Falcon": ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"],
    "Llama3": ["Llama-3.1-8B",    "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"],
    "Gemma3": ["gemma-3-27b-pt",  "gemma-3-27b-it",  "gemma-3-12b-pt", "gemma-3-12b-it"],
}

# PROMPT_LEVELS = ["zeroshot", "instronly", "fewshot"]
PROMPT_LEVELS = ["instronly", "fewshot"]
NO_ZEROSHOT   = {"SNLI", "MNLI"}

LOGPROB_COLS = {
    "yesno": ["yes_logprob", "no_logprob"],
    "nli":   ["o0_logprob", "o1_logprob", "o2_logprob"],
    "mcq":   ["oa_logprob", "ob_logprob", "oc_logprob", "od_logprob"],
}

ROOT_IN   = "../outputs/May-10-2026"
ROOT_OUT  = "../outputs/May-10-2026"

TOTAL         = 1200    # total items in each dataset 
SEED          = 75      # random seed for consistent fold generation across runs
# CALIB_SIZES   = [24, 60, 120, 180, 240]   # must each divide evenly into 2,3,4 for yesno,nli,mcq respectively
CALIB_SIZES  = [60]
# CALIB_SIZES   = [80, 120, 200, 300]

# ── K-fold class-balanced index generation ──────────────────────────────────

# FIXED_K = TOTAL // min(CALIB_SIZES)   # 1200 // 24 = 50, need atleast 50 to give a chance to 24 to cover all items in dataset
# FIXED_K = 15 # effective access to only 120 items in the entire correction process.
FIXED_K = 5

def get_calibration_indices(df, answer_col, options, calib_count, calib_run, seed=SEED):
    """
    Returns (calib_indices, eval_indices) for one fold.

    Uses a fixed k=FIXED_K across all calib sizes. When calib_count is large
    enough that k folds would exceed the available class samples
    (calib_count * FIXED_K > TOTAL), sampling is done WITH replacement while
    remaining class-balanced. Otherwise sampling is without replacement.

    Each class gets its own deterministic RNG (seed + class_i) so fold
    assignments are stable across calib sizes.
    """
    samples_per_class = calib_count // len(options)
    # with_replacement  = (calib_count * FIXED_K) > TOTAL
    with_replacement  = False # we need to cover the 60 items frok the first k-fold in the second k-fold so that the whole dataset is actually covered
    # with_replacement  = True

    calib_indices = []
    for i, option in enumerate(options):
        indices = df[df[answer_col] == option].index.tolist()
        rng     = np.random.default_rng(seed + i)

        if with_replacement:
            # Draw calib_count * FIXED_K samples with replacement, then slice fold
            total_needed = samples_per_class * FIXED_K
            pool = rng.choice(indices, size=total_needed, replace=True)
        else:
            # Shuffle once; folds tile the dataset without replacement
            pool = rng.permutation(indices)

        fold_start = calib_run * samples_per_class
        fold_end   = fold_start + samples_per_class
        calib_indices.extend(pool[fold_start:fold_end].tolist())

    calib_set    = set(calib_indices)
    # eval = everything not chosen for this fold's calibration set
    eval_indices = [i for i in df.index if i not in calib_set]
    return calib_indices, eval_indices

# ── Core correction logic ────────────────────────────────────────────────────

def rb_correct(df, task, answer_col, options, calib_count):
    """
    Response-bias correction via class-balanced calibration.

    k is fixed at FIXED_K for all calib sizes (= TOTAL // min(CALIB_SIZES)).
    For calib sizes where k * calib_count > TOTAL, calibration folds are
    drawn with replacement (still class-balanced).

    Returns df with extra columns per logprob col:
      corrected_{col}         list of fold-ordered corrections (NaN for calib fold)
      avg_corrected_{col}     mean of the above
    Plus:
      corrected_predicted_answer
      plain_predicted_answer  (rename of predicted_answer)
    """
    k       = FIXED_K
    lp_cols = LOGPROB_COLS[task]
    raw     = df[lp_cols].values.astype(float)   # (n, n_options)
    n       = len(df)

    # Storage: (n_items, k_folds) — NaN where item was in calibration set.
    # With replacement, an item may appear in multiple calib folds, so all
    # such folds are NaN for that item; the rest are averaged normally.
    corrections = {col: np.full((n, k), np.nan) for col in lp_cols}

    for fold in range(k):
        calib_idx, eval_idx = get_calibration_indices(
            df, answer_col, options, calib_count, calib_run=fold
        )

        calib_pos   = df.index.get_indexer(calib_idx)
        calib_means = raw[calib_pos, :].mean(axis=0)   # (n_options,)

        eval_pos = df.index.get_indexer(eval_idx)
        for j, col in enumerate(lp_cols):
            corrections[col][eval_pos, fold] = raw[eval_pos, j] - calib_means[j]

    # Build output dataframe
    out = df.copy()

    if "predicted_answer" in out.columns:
        out.rename(columns={"predicted_answer": "plain_predicted_answer"}, inplace=True)
    if "is_correct" in out.columns:
        out.drop(columns=["is_correct"], inplace=True)

    avg_corrected = {}
    for col in lp_cols:
        out[f"corrected_{col}"]     = [list(corrections[col][i]) for i in range(n)]
        out[f"avg_corrected_{col}"] = np.nanmean(corrections[col], axis=1)
        avg_corrected[col]          = out[f"avg_corrected_{col}"].values

    avg_matrix = np.stack([avg_corrected[col] for col in lp_cols], axis=1)
    best_idx   = np.argmax(avg_matrix, axis=1)

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

def build_output_path(prompt_level, dataset_name, family, model, cfg, calib_count):
    # k is always FIXED_K; flag with-replacement runs in the folder name
    with_replacement = (calib_count * FIXED_K) > TOTAL
    repl_tag  = "_wr" if with_replacement else ""
    task_corr = f"{cfg['task_corr']}_n{calib_count}_k{FIXED_K}{repl_tag}"
    return os.path.join(
        ROOT_OUT, prompt_level, dataset_name,
        task_corr, family, cfg["domain"],
        f"{model}_results.csv"
    )

# ── Main pipeline ────────────────────────────────────────────────────────────

def run_correction_pipeline():
    processed, skipped = 0, 0

    for calib_count in CALIB_SIZES:
        # k = TOTAL // calib_count
        k = FIXED_K
        print(f"\n── calib_count={calib_count}, k={k} folds ──")

        for prompt_level in PROMPT_LEVELS:
            for dataset_name, cfg in DATASET_CONFIG.items():

                if prompt_level == "zeroshot" and dataset_name in NO_ZEROSHOT:
                    continue

                for family, models in MODEL_FAMILIES.items():
                    for model in models:

                        in_path  = build_input_path(prompt_level, dataset_name, family, model, cfg)
                        out_path = build_output_path(prompt_level, dataset_name, family, model, cfg, calib_count)

                        if not os.path.exists(in_path):
                            print(f"  [SKIP] {in_path}")
                            skipped += 1
                            continue

                        df = pd.read_csv(in_path)

                        if cfg["task"] == "nli":
                            df[cfg["answer_col"]] = df[cfg["answer_col"]].astype(int)

                        out_df = rb_correct(df, cfg["task"], cfg["answer_col"], cfg["options"], calib_count)

                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        out_df.to_csv(out_path, index=False)
                        print(f"  [OK]   {out_path}")
                        processed += 1

    print(f"\nDone. {processed} files corrected, {skipped} skipped.")

if __name__ == "__main__":
    run_correction_pipeline()