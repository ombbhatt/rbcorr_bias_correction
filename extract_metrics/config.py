# config.py

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT         = Path("../outputs/Mar-23-2026")
RESULTS_DIR  = Path("../results/Mar-23-2026")

# ── Prompt levels ────────────────────────────────────────────────────────────
PROMPT_LEVELS = ["zeroshot", "instronly", "fewshot"]
# PROMPT_LEVELS = ["instronly", "fewshot"]   # skip zeroshot since it has many missing files

# Datasets that have no zeroshot files
NO_ZEROSHOT = {"SNLI", "MNLI"}

# ── Datasets ─────────────────────────────────────────────────────────────────
# Maps dataset name → (task_type, domain_folder)
DATASETS = {
    "ARITH":          ("yesno", "arith"),
    "BABI":           ("yesno", "babi"),
    "COMPS":          ("yesno", "comps"),
    "EWOK":           ("yesno", "all_domains"),
    "SNLI":           ("nli",   "snli"),
    "MNLI":           ("nli",   "mnli"),
    "MMLU-HUMANITIES":("mcq",   "HUMANITIES"),
    "MMLU-OTHERS":    ("mcq",   "OTHERS"),
    "MMLU-SOCIAL_SCI":("mcq",   "SOCIAL_SCI"),
    "MMLU-STEM":      ("mcq",   "STEM"),
}

YESNO_DATASETS = {k for k, (t, _) in DATASETS.items() if t == "yesno"}
NLI_DATASETS   = {k for k, (t, _) in DATASETS.items() if t == "nli"}
MCQ_DATASETS   = {k for k, (t, _) in DATASETS.items() if t == "mcq"}

DATASETS_BY_TYPE = {
    "yesno": sorted(YESNO_DATASETS),
    "nli":   sorted(NLI_DATASETS),
    "mcq":   sorted(MCQ_DATASETS),
}

# ── Models ───────────────────────────────────────────────────────────────────
MODEL_FAMILIES = {
    # "GPT2": ["gpt2-medium", "gpt2-large"],
    "Falcon": [
        "Falcon3-3B-Base",
        "Falcon3-3B-Instruct",
        "Falcon3-10B-Base",
        "Falcon3-10B-Instruct",
    ],
    "Llama3": [
        "Llama-3.1-8B",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B",
        "Llama-3.1-70B-Instruct",
    ],
    "Gemma3": [
        "gemma-3-27b-pt",
        "gemma-3-27b-it",
        "gemma-3-12b-pt",
        "gemma-3-12b-it",
    ],
}

# Reverse map: model_name → family
MODEL_TO_FAMILY = {
    model: family
    for family, models in MODEL_FAMILIES.items()
    for model in models
}

# ── Task type → question type label (for JSON keys) ──────────────────────────
TASK_TO_QTYPE = {
    "yesno": "YESNO",
    "nli":   "NLI",
    "mcq":   "MCQ",
}

# ── Logprob columns by task type ─────────────────────────────────────────────
RAW_LOGPROB_COLS = {
    "yesno": ["yes_logprob", "no_logprob"],
    "nli":   ["o0_logprob", "o1_logprob", "o2_logprob"],
    "mcq":   ["oa_logprob", "ob_logprob", "oc_logprob", "od_logprob"],
}

# Answer option labels by task type
ANSWER_OPTIONS = {
    "yesno": ["Yes", "No"],
    "nli":   [0, 1, 2],
    "mcq":   ["A", "B", "C", "D"],
}

# Ground truth column by task type
GT_COL = {
    "yesno": "Correct Answer",
    "nli":   "Correct Answer",
    "mcq":   "answer",
}

# ── Method folder names ───────────────────────────────────────────────────────
PLAIN_FOLDERS = {
    "yesno": "yesnoplain",
    "nli":   "nliplain",
    "mcq":   "mcqplain",
}

CC_FOLDERS = {
    "yesno": "yesnocccorr",
    "nli":   "nlicccorr",
    "mcq":   "mcqcccorr",
}

BC_BASE_NAMES = {
    "yesno": "yesnobccorr",
    "nli":   "nlibccorr",
    "mcq":   "mcqbccorr",
}

RB_BASE_NAMES = {
    "yesno": "yesnorbcorr",
    "nli":   "nlirbcorr",
    "mcq":   "mcqrbcorr",
}

# Batch sizes and their derived k values
# BATCH_SIZES  = [24, 60, 120, 180, 240]  # Added smaller and larger batch sizes for more comprehensive analysis
BATCH_SIZES  = [60]
TOTAL_DATASET_SIZE = 1200
# FIXED_K = TOTAL_DATASET_SIZE // min(BATCH_SIZES)
# FIXED_K = 15  # Set a fixed k for all batch sizes to ensure consistency in results and avoid issues with small batch sizes
FIXED_K = 5  # Set a fixed k for all batch sizes to ensure consistency in results and avoid issues with small batch sizes
# BATCH_K = {n: TOTAL_DATASET_SIZE // n for n in BATCH_SIZES}
BATCH_K = {n: FIXED_K for n in BATCH_SIZES}

# Transfer correction is only run at this batch size
TRANSFER_BATCH_SIZE = 240
BATCH_K[TRANSFER_BATCH_SIZE] = 5 

# ── Output results directory structure ───────────────────────────────────────
# results/{method}_{qtype}/{promptlevel}_{family}_{dataset}.json
METHOD_QTYPES = [
    ("rb",  "yn"),
    ("rb",  "nli"),
    ("rb",  "mcq"),
    ("bc",  "yn"),
    ("bc",  "nli"),
    ("bc",  "mcq"),
    ("cc",  "yn"),
    ("cc",  "nli"),
    ("cc",  "mcq"),
]

TASK_TO_SHORT = {
    "yesno": "yn",
    "nli":   "nli",
    "mcq":   "mcq",
}