#!/bin/bash

# Specific Metrics Processor - Multiple Choice Questions
# Processes all configurations for MMLU datasets with progress tracking

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PROCESSOR_SCRIPT="${SCRIPT_DIR}/specific_metrics_processor.py"
PROCESSOR_SCRIPT="${SCRIPT_DIR}/process_specific_metrics.py"
DATE="Sep-16-2025"

# Check if processor script exists
if [ ! -f "$PROCESSOR_SCRIPT" ]; then
    echo -e "${RED}Error: Processor script not found at $PROCESSOR_SCRIPT${NC}"
    exit 1
fi

# Dataset configurations
DATASETS_PER=("MMLU-STEM" "MMLU-HUMANITIES" "MMLU-SOCIAL_SCI" "MMLU-OTHERS")
DATASETS_ALL=("MMLU")
MODEL_FAMILIES=("Falcon" "Gemma3" "Llama3")
PROMPTS=("zeroshot" "instronly" "fewshot")
AGGREGATION_LEVELS=("per")

# Calculate total configurations
TOTAL_CONFIGS=0

# Baseline configurations
for agg in "${AGGREGATION_LEVELS[@]}"; do
    if [ "$agg" == "per" ]; then
        TOTAL_CONFIGS=$((TOTAL_CONFIGS + ${#DATASETS_PER[@]} * ${#MODEL_FAMILIES[@]} * ${#PROMPTS[@]}))
    else
        TOTAL_CONFIGS=$((TOTAL_CONFIGS + ${#DATASETS_ALL[@]} * ${#MODEL_FAMILIES[@]} * ${#PROMPTS[@]}))
    fi
done

# Cross-dataset configurations (per only)
TOTAL_CONFIGS=$((TOTAL_CONFIGS + ${#DATASETS_PER[@]} * 3 * ${#MODEL_FAMILIES[@]} * ${#PROMPTS[@]}))
# Note: 3 = number of other MMLU domains (4-1)

# Cross-model configurations
for agg in "${AGGREGATION_LEVELS[@]}"; do
    if [ "$agg" == "per" ]; then
        TOTAL_CONFIGS=$((TOTAL_CONFIGS + ${#DATASETS_PER[@]} * ${#MODEL_FAMILIES[@]} * ${#PROMPTS[@]}))
    else
        TOTAL_CONFIGS=$((TOTAL_CONFIGS + ${#DATASETS_ALL[@]} * ${#MODEL_FAMILIES[@]} * ${#PROMPTS[@]}))
    fi
done

# Cross-prompt configurations
for agg in "${AGGREGATION_LEVELS[@]}"; do
    if [ "$agg" == "per" ]; then
        TOTAL_CONFIGS=$((TOTAL_CONFIGS + ${#DATASETS_PER[@]} * ${#MODEL_FAMILIES[@]} * ${#PROMPTS[@]} * 2))
    else
        TOTAL_CONFIGS=$((TOTAL_CONFIGS + ${#DATASETS_ALL[@]} * ${#MODEL_FAMILIES[@]} * ${#PROMPTS[@]} * 2))
    fi
done
# Note: 2 = number of other prompts (3-1)

CURRENT_CONFIG=0

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Specific Metrics Processor - MCQ${NC}"
echo -e "${BLUE}======================================${NC}"
echo -e "Total configurations to process: ${TOTAL_CONFIGS}"
echo -e "Date: ${DATE}"
echo -e "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Function to run processor and track progress
run_processor() {
    local question_type=$1
    local agg_level=$2
    local dataset=$3
    local model_family=$4
    local prompt=$5
    local cross_dataset_flag=$6
    local cross_model_flag=$7
    local cross_prompt_flag=$8
    local transfer_desc=$9
    
    CURRENT_CONFIG=$((CURRENT_CONFIG + 1))
    local percent=$((CURRENT_CONFIG * 100 / TOTAL_CONFIGS))
    
    echo -e "${YELLOW}[${CURRENT_CONFIG}/${TOTAL_CONFIGS} - ${percent}%]${NC} Processing: ${dataset} | ${model_family} | ${prompt} | ${agg_level} | ${transfer_desc}"
    
    python3 "$PROCESSOR_SCRIPT" \
        --question_type "$question_type" \
        --aggregation_level "$agg_level" \
        --dataset "$dataset" \
        --model_family "$model_family" \
        --prompt_type "$prompt" \
        --date "$DATE" \
        $cross_dataset_flag \
        $cross_model_flag \
        $cross_prompt_flag \
        2>&1 | grep -E "(Processing|Error|Processed|written)" || true
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✓ Completed${NC}"
    else
        echo -e "${RED}  ✗ Failed${NC}"
    fi
    echo ""
}

# echo -e "${BLUE}Phase 1: Baseline (same-dataset, same-model, same-prompt)${NC}"
# echo "================================================================"

# Baseline: per-dataset
for dataset in "${DATASETS_PER[@]}"; do
    for model_family in "${MODEL_FAMILIES[@]}"; do
        for prompt in "${PROMPTS[@]}"; do
            run_processor "mcq" "per" "$dataset" "$model_family" "$prompt" "" "" "" "baseline"
        done
    done
done

echo ""
echo -e "${BLUE}Phase 2: Cross-Dataset Transfer (MMLU Domains)${NC}"
echo "================================================================"

# Cross-dataset: per only (transfer between MMLU domains)
for dataset in "${DATASETS_PER[@]}"; do
    for model_family in "${MODEL_FAMILIES[@]}"; do
        for prompt in "${PROMPTS[@]}"; do
            run_processor "mcq" "per" "$dataset" "$model_family" "$prompt" "--include_cross_dataset" "" "" "cross-dataset"
        done
    done
done

echo ""
echo -e "${BLUE}Phase 3: Cross-Model Transfer${NC}"
echo "================================================================"

# Cross-model: per-dataset
for dataset in "${DATASETS_PER[@]}"; do
    for model_family in "${MODEL_FAMILIES[@]}"; do
        for prompt in "${PROMPTS[@]}"; do
            run_processor "mcq" "per" "$dataset" "$model_family" "$prompt" "" "--include_cross_model" "" "cross-model"
        done
    done
done

echo ""
echo -e "${BLUE}Phase 4: Cross-Prompt Transfer${NC}"
echo "================================================================"

# Cross-prompt: per-dataset
for dataset in "${DATASETS_PER[@]}"; do
    for model_family in "${MODEL_FAMILIES[@]}"; do
        for prompt in "${PROMPTS[@]}"; do
            run_processor "mcq" "per" "$dataset" "$model_family" "$prompt" "" "" "--include_cross_prompt" "cross-prompt"
        done
    done
done

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  All MCQ Configurations Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "Total processed: ${CURRENT_CONFIG}/${TOTAL_CONFIGS}"