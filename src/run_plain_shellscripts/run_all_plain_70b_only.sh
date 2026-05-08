#!/bin/bash

echo "=========================================="
echo "Running Plain for All Configurations"
echo "=========================================="
echo ""

# Define all models
LLAMA3_MODELS=("Meta-Llama-3.1-70B-bnb-4bit" "Meta-Llama-3.1-70B-Instruct-bnb-4bit")

# Combine all models into one array
ALL_MODELS=(
  "${LLAMA3_MODELS[@]}"
)

DATASETS=( "MNLI" "SNLI" "ARITH" "BABI" "COMPS" "EWOK" "MMLU-STEM" "MMLU-HUMANITIES" "MMLU-SOCIAL_SCI" "MMLU-OTHERS")

# PROMPTS=("zeroshot" "instronly" "fewshot")
# PROMPTS=("zeroshot")
# PROMPTS=("instronly")
PROMPTS=("fewshot")

# Counter for tracking progress
total_runs=$((${#ALL_MODELS[@]} * (${#DATASETS[@]}) * ${#PROMPTS[@]}))
current_run=0

echo "Total configurations to run: $total_runs"
echo ""

# ============================================
# ALL DATASETS
# ============================================
echo "Starting ALL Datasets..."
echo ""

for dataset in "${DATASETS[@]}"; do
  echo "----------------------------------------"
  echo "Dataset: $dataset"
  echo "----------------------------------------"
  
  for model in "${ALL_MODELS[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
      # Skip zeroshot for MNLI and SNLI
      if [[ "$prompt" == "zeroshot" && ( "$dataset" == "MNLI" || "$dataset" == "SNLI" ) ]]; then
        echo "Skipping zeroshot for $dataset"
        continue
      fi
      current_run=$((current_run + 1))
      echo ""
      echo "[$current_run/$total_runs] Running: $dataset | $model | $prompt"
      
      python ../unified_transprompt_transmodel_transdata.py \
        --target_dataset "$dataset" \
        --calib_dataset "$dataset" \
        --target_model "$model" \
        --calib_model "$model" \
        --target_prompt "$prompt" \
        --calib_prompt "$prompt" \
        --implementation plain \
        
      
      # Check if the command succeeded
      if [ $? -ne 0 ]; then
        echo "❌ ERROR: Failed for $dataset | $model | $prompt"
        echo "Continuing to next configuration..."
      else
        echo "✅ SUCCESS: $dataset | $model | $prompt"
      fi
    done
  done
done