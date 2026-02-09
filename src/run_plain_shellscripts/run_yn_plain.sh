#!/bin/bash

echo "=========================================="
echo "Running Contextual Calibration for All Configurations"
echo "=========================================="
echo ""

# Define all models
FALCON_MODELS=("Falcon3-3B-Base" "Falcon3-3B-Instruct" "Falcon3-10B-Base" "Falcon3-10B-Instruct")
GEMMA3_MODELS=("gemma-3-27b-pt" "gemma-3-27b-it" "gemma-3-12b-pt" "gemma-3-12b-it")
LLAMA3_MODELS=("Llama-3.1-8B" "Llama-3.1-8B-Instruct" "Llama-3.1-70B" "Llama-3.1-70B-Instruct")


# Combine all models into one array
ALL_MODELS=(
  "${FALCON_MODELS[@]}"
  "${GEMMA3_MODELS[@]}"
  "${LLAMA3_MODELS[@]}"
)

DATASETS=("ARITH" "BABI" "COMPS" "EWOK")

PROMPTS=("zeroshot" "instronly" "fewshot")

# Counter for tracking progress
total_runs=$((${#ALL_MODELS[@]} * (${#DATASETS[@]}) * ${#PROMPTS[@]}))
current_run=0

echo "Total configurations to run: $total_runs"
echo ""

# ============================================
# YES-NO DATASETS
# ============================================
echo "Starting Yes-No Datasets..."
echo ""

for dataset in "${DATASETS[@]}"; do
  echo "----------------------------------------"
  echo "Dataset: $dataset"
  echo "----------------------------------------"
  
  for model in "${ALL_MODELS[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
      current_run=$((current_run + 1))
      echo ""
      echo "[$current_run/$total_runs] Running: $dataset | $model | $prompt"
      
      python unified_transprompt_transmodel_transdata.py \
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