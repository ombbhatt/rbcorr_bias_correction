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
    "SNLI":             dict(task="nli",   task_plain="nliplain",   task_corr="nlorbcorr",   domain="snli",         answer_col="Correct Answer", options=[0,1,2]),
    "MNLI":             dict(task="nli",   task_plain="nliplain",   task_corr="nlorbcorr",   domain="mnli",         answer_col="Correct Answer", options=[0,1,2]),
    # mcq
    "MMLU-HUMANITIES":  dict(task="mcq",   task_plain="mcqplain",   task_corr="mcqrbcorr",   domain="HUMANITIES",   answer_col="answer",         options=["A","B","C","D"]),
    "MMLU-OTHERS":      dict(task="mcq",   task_plain="mcqplain",   task_corr="mcqrbcorr",   domain="OTHERS",       answer_col="answer",         options=["A","B","C","D"]),
    "MMLU-SOCIAL_SCI":  dict(task="mcq",   task_plain="mcqplain",   task_corr="mcqrbcorr",   domain="SOCIAL_SCI",   answer_col="answer",         options=["A","B","C","D"]),
    "MMLU-STEM":        dict(task="mcq",   task_plain="mcqplain",   task_corr="mcqrbcorr",   domain="STEM",         answer_col="answer",         options=["A","B","C","D"]),
}

MODEL_FAMILIES = {
    "Falcon": ["Falcon3-3B-Base", "Falcon3-3B-Instruct", "Falcon3-10B-Base", "Falcon3-10B-Instruct"],
    "Llama3": ["Llama-3.1-8B",    "Llama-3.1-8B-Instruct", "Llama-3.1-70B", "Llama-3.1-70B-Instruct"],
    "Gemma3": ["gemma-3-27b-pt",  "gemma-3-27b-it",  "gemma-3-12b-pt", "gemma-3-12b-it"],
}

PROMPT_LEVELS   = ["zeroshot", "instronly", "fewshot"]
NO_ZEROSHOT     = {"SNLI", "MNLI"}   # datasets that have no zeroshot files

LOGPROB_COLS = {
    "yesno": ["yes_logprob", "no_logprob"],
    "nli":   ["o0_logprob", "o1_logprob", "o2_logprob"],
    "mcq":   ["oa_logprob", "ob_logprob", "oc_logprob", "od_logprob"],
}

ROOT_IN  = "outputs/Mar-06-2026"
ROOT_OUT = "outputs/Mar-06-2026"   # corrected folders sit alongside plain ones under the same root

K_FOLDS      = 5
CALIB_COUNT  = 40
TOTAL        = 200
SEED         = 42

# ── K-fold class-balanced index generation ──────────────────────────────────

def get_calibration_indices(df, answer_col, options, calib_count, calib_run, k=K_FOLDS, seed=SEED):
    """
    Returns (calib_indices, eval_indices) for one fold.
    Each class is shuffled once (deterministically via seed+class_i) then sliced
    into k equal folds, guaranteeing mutual exclusivity and class balance.
    """
    samples_per_class = calib_count // len(options)

    calib_indices = []
    for i, option in enumerate(options):
        indices = df[df[answer_col] == option].index.tolist()
        rng      = np.random.default_rng(seed + i)
        shuffled = rng.permutation(indices)

        fold_start = calib_run * samples_per_class
        fold_end   = fold_start + samples_per_class
        calib_indices.extend(shuffled[fold_start:fold_end].tolist())

    calib_set  = set(calib_indices)
    eval_indices = [i for i in df.index if i not in calib_set]
    return calib_indices, eval_indices

# ── Core correction logic ────────────────────────────────────────────────────

def rb_correct(df, task, answer_col, options):
    """
    Response-bias correction via k-fold class-balanced calibration.

    For each fold:
      - Draw a class-balanced calibration set from the ORIGINAL logprobs
      - Compute per-option mean logprob over that calibration set
      - Subtract those means from the eval items' logprobs (same option → same option)

    Each item accumulates one correction per fold in which it is an eval item (4 out of 5).
    Results stored in fold order; missing fold (where item was calibration) stored as NaN.

    Returns df with extra columns:
      corrected_{col}, avg_corrected_{col}  for each logprob col
      corrected_predicted_answer
      plain_predicted_answer  (rename of predicted_answer)
    """
    lp_cols = LOGPROB_COLS[task]
    raw     = df[lp_cols].values.astype(float)          # shape (200, n_options)
    n       = len(df)

    # Storage: (n_items, k_folds) — NaN where item was in calibration set
    corrections = {col: np.full((n, K_FOLDS), np.nan) for col in lp_cols}

    for fold in range(K_FOLDS):
        calib_idx, eval_idx = get_calibration_indices(
            df, answer_col, options, CALIB_COUNT, calib_run=fold
        )

        # Calibration means from ORIGINAL logprobs (use iloc-position-safe indexing)
        calib_pos = df.index.get_indexer(calib_idx)     # positional
        calib_means = raw[calib_pos, :].mean(axis=0)    # shape (n_options,)

        # Apply correction to eval items
        eval_pos = df.index.get_indexer(eval_idx)
        for j, col in enumerate(lp_cols):
            corrections[col][eval_pos, fold] = raw[eval_pos, j] - calib_means[j]

    # Build output dataframe
    out = df.copy()

    # Rename original predicted_answer
    if "predicted_answer" in out.columns:
        out.rename(columns={"predicted_answer": "plain_predicted_answer"}, inplace=True)

    # Drop is_correct
    if "is_correct" in out.columns:
        out.drop(columns=["is_correct"], inplace=True)

    # Add corrected columns and averages
    avg_corrected = {}
    for col in lp_cols:
        corr_col     = f"corrected_{col}"
        avg_corr_col = f"avg_corrected_{col}"
        # Store as list-in-cell (fold-ordered, NaN for calib fold)
        out[corr_col]     = [list(corrections[col][i]) for i in range(n)]
        out[avg_corr_col] = np.nanmean(corrections[col], axis=1)
        avg_corrected[col] = out[avg_corr_col].values

    # Corrected predicted answer: argmax over avg corrected logprobs
    avg_matrix = np.stack([avg_corrected[col] for col in lp_cols], axis=1)  # (n, n_options)
    best_idx   = np.argmax(avg_matrix, axis=1)

    if task == "yesno":
        label_map = {0: "Yes", 1: "No"}
    elif task == "nli":
        label_map = {0: 0, 1: 1, 2: 2}
    else:  # mcq
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

                    # NLI answer col is stored as int
                    if cfg["task"] == "nli":
                        df[cfg["answer_col"]] = df[cfg["answer_col"]].astype(int)

                    out_df = rb_correct(df, cfg["task"], cfg["answer_col"], cfg["options"])

                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    out_df.to_csv(out_path, index=False)
                    print(f"[OK]   {out_path}")
                    processed += 1

    print(f"\nDone. {processed} files corrected, {skipped} skipped.")

if __name__ == "__main__":
    run_correction_pipeline()