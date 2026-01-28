#!/bin/bash
# rerun_specific_crossprompt_yesno.sh

echo "=========================================="
echo "CROSS-PROMPT TRANSFER: YES-NO DATASETS"
echo "=========================================="
echo ""

FALCON_MODELS=("Falcon3-3B-Base" "Falcon3-3B-Instruct" "Falcon3-10B-Base" "Falcon3-10B-Instruct")
GEMMA3_MODELS=("gemma-3-27b-pt" "gemma-3-27b-it" "gemma-3-12b-pt" "gemma-3-12b-it")
LLAMA3_MODELS=("Llama-3.1-8B" "Llama-3.1-8B-Instruct" "Llama-3.1-70B" "Llama-3.1-70B-Instruct")

YESNO_DATASETS=("COMPS" "EWOK" "BABI" "ARITH")
ALL_PROMPTS=("fewshot" "zeroshot" "instronly")

LOG_FILE="logs/crossprompt_yesno-median.log"
mkdir -p "$(dirname "$LOG_FILE")"

make_config_key() {
  echo "$1|$2|$3|$4|$5|$6"
}

is_completed() {
  local key="$1"
  [ -f "$LOG_FILE" ] && grep -q "^COMPLETED: $key$" "$LOG_FILE"
}

mark_completed() {
  local key="$1"
  echo "COMPLETED: $key" >> "$LOG_FILE"
}

count_total_configs() {
  local total=0
  for dataset in "${YESNO_DATASETS[@]}"; do
    for family_name in "FALCON" "GEMMA3" "LLAMA3"; do
      case $family_name in
        FALCON) family_models=("${FALCON_MODELS[@]}") ;;
        GEMMA3) family_models=("${GEMMA3_MODELS[@]}") ;;
        LLAMA3) family_models=("${LLAMA3_MODELS[@]}") ;;
      esac
      for model in "${family_models[@]}"; do
        for target_prompt in "${ALL_PROMPTS[@]}"; do
          for calib_prompt in "${ALL_PROMPTS[@]}"; do
            [ "$target_prompt" == "$calib_prompt" ] && continue
            total=$((total + 1))
          done
        done
      done
    done
  done
  echo $total
}

run_specific() {
  local dataset=$1
  local model=$2
  local target_prompt=$3
  local calib_prompt=$4
  local current=$5
  local total=$6
  
  local config_key=$(make_config_key "$dataset" "$dataset" "$model" "$model" "$target_prompt" "$calib_prompt")
  
  if is_completed "$config_key"; then
    echo "⏭️  (skipped - already completed) [$current/$total]"
    return 0
  fi
  
  echo "🔄 Running... [$current/$total]"
  
  python unified_transprompt_transmodel_transdata.py \
    --target_dataset "$dataset" --calib_dataset "$dataset" \
    --target_model "$model" --calib_model "$model" \
    --target_prompt "$target_prompt" --calib_prompt "$calib_prompt" \
    --implementation specific \
    # --force
  
  local exit_code=$?
  
  if [ $exit_code -eq 0 ]; then
    mark_completed "$config_key"
    echo "✅ Complete! [$current/$total]"
    return 0
  else
    echo "❌ Failed! [$current/$total]"
    return 1
  fi
}

echo "Calculating total configurations..."
TOTAL_CONFIGS=$(count_total_configs)
echo "Total configurations to process: $TOTAL_CONFIGS"
echo ""

total=0
success=0
skipped=0

if [ -f "$LOG_FILE" ]; then
  completed_count=$(grep -c "^COMPLETED:" "$LOG_FILE" 2>/dev/null || echo 0)
  echo "📋 Found existing progress log with $completed_count completed configurations"
  echo "   Resuming from where we left off..."
  echo "   Remaining: $((TOTAL_CONFIGS - completed_count)) configurations"
  echo ""
fi

for dataset in "${YESNO_DATASETS[@]}"; do
  echo "Dataset: $dataset"
  
  for family_name in "FALCON" "GEMMA3" "LLAMA3"; do
    case $family_name in
      FALCON) family_models=("${FALCON_MODELS[@]}") ;;
      GEMMA3) family_models=("${GEMMA3_MODELS[@]}") ;;
      LLAMA3) family_models=("${LLAMA3_MODELS[@]}") ;;
    esac
    
    for model in "${family_models[@]}"; do
      for target_prompt in "${ALL_PROMPTS[@]}"; do
        for calib_prompt in "${ALL_PROMPTS[@]}"; do
          [ "$target_prompt" == "$calib_prompt" ] && continue
          
          total=$((total + 1))
          echo -n "  $model | $target_prompt←$calib_prompt: "
          
          result=$(run_specific "$dataset" "$model" "$target_prompt" "$calib_prompt" "$total" "$TOTAL_CONFIGS")
          
          if [[ "$result" == *"skipped"* ]]; then
            skipped=$((skipped + 1))
            success=$((success + 1))
            echo "$result"
          elif [[ "$result" == *"✅"* ]]; then
            success=$((success + 1))
            echo "$result"
          else
            echo "$result"
          fi
        done
      done
    done
  done
  echo ""
done

echo "=========================================="
echo "COMPLETE: $success/$total successful"
if [ $skipped -gt 0 ]; then
  echo "  ($skipped were skipped from previous run)"
  echo "  ($(( success - skipped )) newly completed in this run)"
fi
echo "=========================================="
echo ""
echo "Progress log saved to: $LOG_FILE"
echo "To start fresh, delete: rm $LOG_FILE"