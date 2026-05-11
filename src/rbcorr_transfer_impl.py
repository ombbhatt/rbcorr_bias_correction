import argparse
import os
import sys
import numpy as np
import pandas as pd

# ── Dataset / path configuration ────────────────────────────────────────────

DATASET_CONFIG = {
    "ARITH":            dict(task="yesno", task_plain="yesnoplain", task_corr="yesnorbcorr", domain="arith",        answer_col="Correct Answer", options=["Yes","No"]),
    "BABI":             dict(task="yesno", task_plain="yesnoplain", task_corr="yesnorbcorr", domain="babi",         answer_col="Correct Answer", options=["Yes","No"]),
    "COMPS":            dict(task="yesno", task_plain="yesnoplain", task_corr="yesnorbcorr", domain="comps",        answer_col="Correct Answer", options=["Yes","No"]),
    "EWOK":             dict(task="yesno", task_plain="yesnoplain", task_corr="yesnorbcorr", domain="all_domains",  answer_col="Correct Answer", options=["Yes","No"]),
    "SNLI":             dict(task="nli",   task_plain="nliplain",   task_corr="nlirbcorr",   domain="snli",         answer_col="Correct Answer", options=[0,1,2]),
    "MNLI":             dict(task="nli",   task_plain="nliplain",   task_corr="nlirbcorr",   domain="mnli",         answer_col="Correct Answer", options=[0,1,2]),
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

PROMPT_LEVELS = ["zeroshot", "instronly", "fewshot"]
# PROMPT_LEVELS = ["instronly", "fewshot"]   # skip zeroshot since it has many missing files
NO_ZEROSHOT   = {"SNLI", "MNLI"}

# Reverse lookup: model name → family name
MODEL_TO_FAMILY = {m: f for f, ms in MODEL_FAMILIES.items() for m in ms}

LOGPROB_COLS = {
    "yesno": ["yes_logprob", "no_logprob"],
    "nli":   ["o0_logprob", "o1_logprob", "o2_logprob"],
    "mcq":   ["oa_logprob", "ob_logprob", "oc_logprob", "od_logprob"],
}

ROOT_IN  = "../outputs/Mar-23-2026"
ROOT_OUT = "../outputs/Mar-23-2026"

TOTAL       = 1200
SEED        = 75  
CALIB_SIZES = [240]
FIXED_K     = 5

# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="RB transfer correction across model/dataset/prompt.")
    parser.add_argument("--transfer_aspect", required=True, choices=["model", "dataset", "prompt"],
                        help="Aspect to transfer across.")
    parser.add_argument("--src_model",   required=True)
    parser.add_argument("--src_dataset", required=True)
    parser.add_argument("--src_prompt",  required=True)
    parser.add_argument("--tgt_model",   required=True)
    parser.add_argument("--tgt_dataset", required=True)
    parser.add_argument("--tgt_prompt",  required=True)
    return parser.parse_args()

# ── Transfer configuration ───────────────────────────────────────────────────

# ── Validation ───────────────────────────────────────────────────────────────

def validate_transfer(aspect, src_model, src_dataset, src_prompt,
                                tgt_model, tgt_dataset, tgt_prompt):
    aspects = {
        "model":   (src_model,   tgt_model),
        "dataset": (src_dataset, tgt_dataset),
        "prompt":  (src_prompt,  tgt_prompt),
    }

    # Exactly one aspect must differ
    differing = [a for a, (s, t) in aspects.items() if s != t]
    if differing != [aspect]:
        sys.exit(
            f"[ERROR] transfer_aspect='{aspect}' but differing aspects are {differing}. "
            f"Exactly one aspect must differ and it must match transfer_aspect."
        )

    # Cross-dataset: same task type required
    if aspect == "dataset":
        if DATASET_CONFIG[src_dataset]["task"] != DATASET_CONFIG[tgt_dataset]["task"]:
            sys.exit(
                f"[ERROR] Cross-dataset transfer requires the same task type. "
                f"{src_dataset} is '{DATASET_CONFIG[src_dataset]['task']}', "
                f"{tgt_dataset} is '{DATASET_CONFIG[tgt_dataset]['task']}'."
            )

    # Cross-model: same family required
    if aspect == "model":
        src_family = MODEL_TO_FAMILY.get(src_model)
        tgt_family = MODEL_TO_FAMILY.get(tgt_model)
        if src_family != tgt_family:
            sys.exit(
                f"[ERROR] Cross-model transfer requires the same model family. "
                f"{src_model} is '{src_family}', {tgt_model} is '{tgt_family}'."
            )

    print(f"[OK] Validation passed. Transferring across {aspect}: "
          f"'{aspects[aspect][0]}' → '{aspects[aspect][1]}'")

# ── K-fold class-balanced index generation ──────────────────────────────────


def get_calibration_indices(df, answer_col, options, calib_count, calib_run, seed=SEED):
    """
    Always samples with replacement since FIXED_K * calib_count > TOTAL for
    all calib sizes used here. Each class gets its own deterministic RNG.
    """
    samples_per_class = calib_count // len(options)
    calib_indices = []
    for i, option in enumerate(options):
        indices = df[df[answer_col] == option].index.tolist()
        rng     = np.random.default_rng(seed + i)
        pool    = rng.choice(indices, size=samples_per_class * FIXED_K, replace=True)
        fold_start = calib_run * samples_per_class
        fold_end   = fold_start + samples_per_class
        calib_indices.extend(pool[fold_start:fold_end].tolist())
    return calib_indices

# ── Core correction logic ────────────────────────────────────────────────────

def rb_transfer_correct(src_df, tgt_df, task, answer_col, options, calib_count):
    """
    Transfer RB correction.

    Draws FIXED_K independent class-balanced calibration sets (with replacement)
    from src_df. For each fold, computes per-option mean logprob from the source
    calibration set and subtracts it from ALL items in tgt_df.

    Each target item accumulates FIXED_K corrections (one per fold).
    avg_corrected_{col} is the mean across all FIXED_K corrections.

    Source and target must share the same logprob column structure (same task type).
    """
    k       = FIXED_K
    lp_cols = LOGPROB_COLS[task]

    src_raw = src_df[lp_cols].values.astype(float)
    tgt_raw = tgt_df[lp_cols].values.astype(float)
    n_tgt   = len(tgt_df)

    # Storage: (n_target_items, k_folds) — no NaNs since all target items are always eval
    corrections = {col: np.full((n_tgt, k), np.nan) for col in lp_cols}

    for fold in range(k):
        calib_idx   = get_calibration_indices(
            src_df, answer_col, options, calib_count, calib_run=fold
        )
        calib_pos   = src_df.index.get_indexer(calib_idx)
        calib_means = src_raw[calib_pos, :].mean(axis=0)   # (n_options,)

        for j, col in enumerate(lp_cols):
            corrections[col][:, fold] = tgt_raw[:, j] - calib_means[j]

    # Build output dataframe
    out = tgt_df.copy()

    if "predicted_answer" in out.columns:
        out.rename(columns={"predicted_answer": "plain_predicted_answer"}, inplace=True)
    if "is_correct" in out.columns:
        out.drop(columns=["is_correct"], inplace=True)

    avg_corrected = {}
    for col in lp_cols:
        out[f"corrected_{col}"]     = [list(corrections[col][i]) for i in range(n_tgt)]
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

def build_path(prompt_level, dataset_name, family, model, cfg):
    return os.path.join(
        ROOT_IN, prompt_level, dataset_name,
        cfg["task_plain"], family, cfg["domain"],
        f"{model}_results.csv"
    )

def build_output_path(tgt_prompt, tgt_dataset, tgt_model, src_value, calib_count):
    tgt_cfg    = DATASET_CONFIG[tgt_dataset]
    tgt_family = MODEL_TO_FAMILY[tgt_model]
    task_corr  = f"{tgt_cfg['task_corr']}_n{calib_count}_k{FIXED_K}"
    filename   = f"{tgt_model}_results_from_{src_value}.csv"
    return os.path.join(
        ROOT_OUT, tgt_prompt, tgt_dataset,
        task_corr, tgt_family, tgt_cfg["domain"],
        filename
    )

# ── Main pipeline ────────────────────────────────────────────────────────────

def run_transfer_pipeline():
    args = parse_args()

    TRANSFER_ASPECT = args.transfer_aspect
    SOURCE_MODEL    = args.src_model
    SOURCE_DATASET  = args.src_dataset
    SOURCE_PROMPT   = args.src_prompt
    TARGET_MODEL    = args.tgt_model
    TARGET_DATASET  = args.tgt_dataset
    TARGET_PROMPT   = args.tgt_prompt

    validate_transfer(
        TRANSFER_ASPECT,
        SOURCE_MODEL, SOURCE_DATASET, SOURCE_PROMPT,
        TARGET_MODEL, TARGET_DATASET, TARGET_PROMPT,
    )

    src_cfg    = DATASET_CONFIG[SOURCE_DATASET]
    tgt_cfg    = DATASET_CONFIG[TARGET_DATASET]
    src_family = MODEL_TO_FAMILY[SOURCE_MODEL]
    tgt_family = MODEL_TO_FAMILY[TARGET_MODEL]

    # The source and target share the same task type (validated above for cross-dataset)
    task       = tgt_cfg["task"]
    answer_col = tgt_cfg["answer_col"]
    options    = tgt_cfg["options"]

    # Source answer col / options (may differ in name for cross-dataset within same type)
    src_answer_col = src_cfg["answer_col"]
    src_options    = src_cfg["options"]

    src_value = {
        "model":   SOURCE_MODEL,
        "dataset": SOURCE_DATASET,
        "prompt":  SOURCE_PROMPT,
    }[TRANSFER_ASPECT]

    src_path = build_path(SOURCE_PROMPT, SOURCE_DATASET, src_family, SOURCE_MODEL, src_cfg)
    tgt_path = build_path(TARGET_PROMPT, TARGET_DATASET, tgt_family, TARGET_MODEL, tgt_cfg)

    if not os.path.exists(src_path):
        sys.exit(f"[ERROR] Source file not found: {src_path}")
    if not os.path.exists(tgt_path):
        sys.exit(f"[ERROR] Target file not found: {tgt_path}")

    src_df = pd.read_csv(src_path)
    tgt_df = pd.read_csv(tgt_path)

    if task == "nli":
        src_df[src_answer_col] = src_df[src_answer_col].astype(int)
        tgt_df[answer_col]     = tgt_df[answer_col].astype(int)

    processed = 0
    for calib_count in CALIB_SIZES:
        # k        = TOTAL // calib_count
        k       = FIXED_K
        out_path = build_output_path(TARGET_PROMPT, TARGET_DATASET, TARGET_MODEL, src_value, calib_count)

        out_df = rb_transfer_correct(
            src_df, tgt_df, task, src_answer_col, src_options, calib_count
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"  [OK] calib_count={calib_count}, k={k} → {out_path}")
        processed += 1

    print(f"\nDone. {processed} files written.")

if __name__ == "__main__":
    run_transfer_pipeline()