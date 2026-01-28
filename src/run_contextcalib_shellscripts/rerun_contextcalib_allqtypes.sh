#!/bin/bash
# rerun_contextcalib.sh - Re-run all contextual calibration with pre-computed content-free probs

echo "=========================================="
echo "Re-running Contextual Calibration for All Configurations"
echo "=========================================="
echo ""

FALCON_MODELS=("Falcon3-3B-Base" "Falcon3-3B-Instruct" "Falcon3-10B-Base" "Falcon3-10B-Instruct")
GEMMA3_MODELS=("gemma-3-27b-pt" "gemma-3-27b-it" "gemma-3-12b-pt" "gemma-3-12b-it")
LLAMA3_MODELS=("Llama-3.1-8B" "Llama-3.1-8B-Instruct" "Llama-3.1-70B" "Llama-3.1-70B-Instruct")

ALL_MODELS=(
  "${FALCON_MODELS[@]}"
  "${GEMMA3_MODELS[@]}"
  "${LLAMA3_MODELS[@]}"
)

YESNO_DATASETS=("COMPS" "EWOK" "BABI" "ARITH")
MCQ_DATASETS=("MMLU-STEM" "MMLU-HUMANITIES" "MMLU-SOCIAL_SCI" "MMLU-OTHERS")
NLI_DATASETS=("SNLI" "MNLI")

PROMPTS=("zeroshot" "fewshot" "instronly")

# Counter for tracking progress
total_runs=$((${#ALL_MODELS[@]} * (${#YESNO_DATASETS[@]} + ${#MCQ_DATASETS[@]} + ${#NLI_DATASETS[@]}) * ${#PROMPTS[@]}))
total_runs=$((${#ALL_MODELS[@]} * ${#MCQ_DATASETS[@]} * ${#PROMPTS[@]}))
total_runs=$((${#ALL_MODELS[@]} * ${#NLI_DATASETS[@]} * ${#PROMPTS[@]}))

current_run=0

echo "Total configurations to run: $total_runs"
echo ""

# ============================================
# YES-NO DATASETS
# ============================================
echo "Starting Yes-No Datasets..."
echo ""

for dataset in "${YESNO_DATASETS[@]}"; do
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
        --implementation contextcalib \
        # --force
      
      if [ $? -ne 0 ]; then
        echo "❌ ERROR: Failed for $dataset | $model | $prompt"
        echo "Continuing to next configuration..."
      else
        echo "✅ SUCCESS: $dataset | $model | $prompt"
      fi
    done
  done
done

# ============================================
# MCQ DATASETS (MMLU)
# ============================================
echo ""
echo "=========================================="
echo "Starting MCQ Datasets (MMLU)..."
echo "=========================================="
echo ""

for dataset in "${MCQ_DATASETS[@]}"; do
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
        --implementation contextcalib \
        --force
      
      if [ $? -ne 0 ]; then
        echo "❌ ERROR: Failed for $dataset | $model | $prompt"
        echo "Continuing to next configuration..."
      else
        echo "✅ SUCCESS: $dataset | $model | $prompt"
      fi
    done
  done
done

# ============================================
# NLI DATASETS
# ============================================
echo "Starting NLI Datasets..."
echo ""

for dataset in "${NLI_DATASETS[@]}"; do
  echo "----------------------------------------"
  echo "Dataset: $dataset"
  echo "----------------------------------------"
  
  for model in "${ALL_MODELS[@]}"; do
    for prompt in "fewshot" "instronly"; do
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
        --implementation contextcalib \
        --force
      
      if [ $? -ne 0 ]; then
        echo "❌ ERROR: Failed for $dataset | $model | $prompt"
        echo "Continuing to next configuration..."
      else
        echo "✅ SUCCESS: $dataset | $model | $prompt"
      fi
    done
  done
done

echo ""
echo "=========================================="
echo "✅ All contextual calibration runs complete!"
echo "Total configurations run: $current_run"
echo "=========================================="