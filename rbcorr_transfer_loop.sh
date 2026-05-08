#!/bin/bash
set -e  # exit on first error

SCRIPT="rbcorr-transfer-tinyversion.py"

# ── Model / dataset / prompt definitions ────────────────────────────────────

FALCON_MODELS=("Falcon3-3B-Base" "Falcon3-3B-Instruct" "Falcon3-10B-Base" "Falcon3-10B-Instruct")
LLAMA3_MODELS=("Llama-3.1-8B" "Llama-3.1-8B-Instruct" "Llama-3.1-70B" "Llama-3.1-70B-Instruct")
GEMMA3_MODELS=("gemma-3-27b-pt" "gemma-3-27b-it" "gemma-3-12b-pt" "gemma-3-12b-it")

YESNO_DATASETS=("ARITH" "BABI" "COMPS" "EWOK")
NLI_DATASETS=("SNLI" "MNLI")
MCQ_DATASETS=("MMLU-HUMANITIES" "MMLU-OTHERS" "MMLU-SOCIAL_SCI" "MMLU-STEM")

ALL_PROMPTS=("zeroshot" "instronly" "fewshot")
NLI_PROMPTS=("instronly" "fewshot")   # no zeroshot for NLI

# ── Helper: run one transfer call ────────────────────────────────────────────

run_transfer() {
    local aspect=$1 src_model=$2 src_dataset=$3 src_prompt=$4 \
                    tgt_model=$5 tgt_dataset=$6 tgt_prompt=$7
    echo "  [RUN] $aspect | $src_model/$src_dataset/$src_prompt → $tgt_model/$tgt_dataset/$tgt_prompt"
    python "$SCRIPT" \
        --transfer_aspect "$aspect" \
        --src_model "$src_model" --src_dataset "$src_dataset" --src_prompt "$src_prompt" \
        --tgt_model "$tgt_model" --tgt_dataset "$tgt_dataset" --tgt_prompt "$tgt_prompt"
}

# ── 1. Cross-dataset ─────────────────────────────────────────────────────────
# Within same task type only. Model and prompt are held fixed.
# NLI datasets skip zeroshot.

echo "=== Cross-dataset ==="

for family_models in "FALCON_MODELS[@]" "LLAMA3_MODELS[@]" "GEMMA3_MODELS[@]"; do
# for family_models in "FALCON_MODELS[@]"; do
    for model in "${!family_models}"; do

        # # yes-no
        for prompt in "${ALL_PROMPTS[@]}"; do
            for src in "${YESNO_DATASETS[@]}"; do
                for tgt in "${YESNO_DATASETS[@]}"; do
                    [ "$src" == "$tgt" ] && continue
                    run_transfer "dataset" "$model" "$src" "$prompt" "$model" "$tgt" "$prompt"
                done
            done
        done

        # nli (no zeroshot)
        for prompt in "${NLI_PROMPTS[@]}"; do
            for src in "${NLI_DATASETS[@]}"; do
                for tgt in "${NLI_DATASETS[@]}"; do
                    [ "$src" == "$tgt" ] && continue
                    run_transfer "dataset" "$model" "$src" "$prompt" "$model" "$tgt" "$prompt"
                done
            done
        done

        # mcq
        for prompt in "${ALL_PROMPTS[@]}"; do
            for src in "${MCQ_DATASETS[@]}"; do
                for tgt in "${MCQ_DATASETS[@]}"; do
                    [ "$src" == "$tgt" ] && continue
                    run_transfer "dataset" "$model" "$src" "$prompt" "$model" "$tgt" "$prompt"
                done
            done
        done

    done
done

# ── 2. Cross-model ───────────────────────────────────────────────────────────
# Within same model family only. Dataset and prompt are held fixed.
# NLI datasets skip zeroshot.

echo "=== Cross-model ==="

for family_models in "FALCON_MODELS[@]" "LLAMA3_MODELS[@]" "GEMMA3_MODELS[@]"; do
# for family_models in "LLAMA3_MODELS[@]" "GEMMA3_MODELS[@]"; do
    for src_model in "${!family_models}"; do
        for tgt_model in "${!family_models}"; do
            [ "$src_model" == "$tgt_model" ] && continue

            # yes-no datasets
            for dataset in "${YESNO_DATASETS[@]}"; do
                for prompt in "${ALL_PROMPTS[@]}"; do
                    run_transfer "model" "$src_model" "$dataset" "$prompt" "$tgt_model" "$dataset" "$prompt"
                done
            done

            # nli datasets (no zeroshot)
            for dataset in "${NLI_DATASETS[@]}"; do
                for prompt in "${NLI_PROMPTS[@]}"; do
                    run_transfer "model" "$src_model" "$dataset" "$prompt" "$tgt_model" "$dataset" "$prompt"
                done
            done

            # mcq datasets
            for dataset in "${MCQ_DATASETS[@]}"; do
                for prompt in "${ALL_PROMPTS[@]}"; do
                    run_transfer "model" "$src_model" "$dataset" "$prompt" "$tgt_model" "$dataset" "$prompt"
                done
            done

        done
    done
done

# # ── 3. Cross-prompt ──────────────────────────────────────────────────────────
# All prompt pairs. Model and dataset are held fixed.
# NLI datasets skip zeroshot entirely (neither src nor tgt can be zeroshot).

echo "=== Cross-prompt ==="

for family_models in "FALCON_MODELS[@]" "LLAMA3_MODELS[@]" "GEMMA3_MODELS[@]"; do
# for family_models in "FALCON_MODELS[@]"; do
    for model in "${!family_models}"; do

        # yes-no and mcq: all prompt pairs
        for dataset in "${YESNO_DATASETS[@]}" "${MCQ_DATASETS[@]}"; do
            for src_prompt in "${ALL_PROMPTS[@]}"; do
                for tgt_prompt in "${ALL_PROMPTS[@]}"; do
                    [ "$src_prompt" == "$tgt_prompt" ] && continue
                    run_transfer "prompt" "$model" "$dataset" "$src_prompt" "$model" "$dataset" "$tgt_prompt"
                done
            done
        done

        # nli: only instronly <-> fewshot
        for dataset in "${NLI_DATASETS[@]}"; do
            for src_prompt in "${NLI_PROMPTS[@]}"; do
                for tgt_prompt in "${NLI_PROMPTS[@]}"; do
                    [ "$src_prompt" == "$tgt_prompt" ] && continue
                    run_transfer "prompt" "$model" "$dataset" "$src_prompt" "$model" "$dataset" "$tgt_prompt"
                done
            done
        done

    done
done

echo ""
echo "All transfer runs complete."