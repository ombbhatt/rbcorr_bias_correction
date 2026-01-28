#!/bin/bash
# rerun_sameconfigs_mcq.sh

echo "================================================================================="
echo "BATCH CALIBRATION CORRECTIONS RUNS ONLY (YN)"
echo "================================================================================="
echo ""

FALCON_MODELS=("Falcon3-3B-Base" "Falcon3-3B-Instruct" "Falcon3-10B-Base" "Falcon3-10B-Instruct")
GEMMA3_MODELS=("gemma-3-27b-pt" "gemma-3-27b-it" "gemma-3-12b-pt" "gemma-3-12b-it")
LLAMA3_MODELS=("Llama-3.1-8B" "Llama-3.1-8B-Instruct" "Llama-3.1-70B" "Llama-3.1-70B-Instruct")

ALL_DATASETS=("ARITH" "BABI" "COMPS" "EWOK")
ALL_PROMPTS=("zeroshot" "instronly" "fewshot")

LOG_FILE="logs/bc-yn.log"
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

echo "Counting total configs..."
TOTAL_CONFIGS=0
for family_name in "FALCON" "GEMMA3" "LLAMA3"; do
  case $family_name in
    FALCON) family_models=("${FALCON_MODELS[@]}") ;;
    GEMMA3) family_models=("${GEMMA3_MODELS[@]}") ;;
    LLAMA3) family_models=("${LLAMA3_MODELS[@]}") ;;
  esac
  for model in "${family_models[@]}"; do
    for dataset in "${ALL_DATASETS[@]}"; do
      for prompt in "${ALL_PROMPTS[@]}"; do
        TOTAL_CONFIGS=$((TOTAL_CONFIGS + 1))
      done
    done
  done
done

echo "Total: $TOTAL_CONFIGS configurations"
echo ""

count=0

for family_name in "FALCON" "GEMMA3" "LLAMA3"; do
  case $family_name in
    FALCON) family_models=("${FALCON_MODELS[@]}") ;;
    GEMMA3) family_models=("${GEMMA3_MODELS[@]}") ;;
    LLAMA3) family_models=("${LLAMA3_MODELS[@]}") ;;
  esac
  for model in "${family_models[@]}"; do
    for dataset in "${ALL_DATASETS[@]}"; do
      for prompt in "${ALL_PROMPTS[@]}"; do
        
        count=$((count + 1))
        key=$(make_config_key "$dataset" "$dataset" "$model" "$model" "$prompt" "$prompt")
        
        # if is_completed "$key"; then
        #   echo "⏭️  [$count/$TOTAL_CONFIGS] Already completed → $key"
        #   continue
        # fi

        echo "🔄  [$count/$TOTAL_CONFIGS] Running → $key"
        python unified_transprompt_transmodel_transdata.py \
          --target_dataset "$dataset" --calib_dataset "$dataset" \
          --target_model "$model" --calib_model "$model" \
          --target_prompt "$prompt" --calib_prompt "$prompt" \
          --implementation batchcalib

        if [ $? -eq 0 ]; then
          mark_completed "$key"
          echo "✅ Done → $key"
        else
          echo "❌ Failed → $key"
        fi

      done
    done
  done
done

echo "=========================================="
echo "ALL SAME-CONDITION RUNS COMPLETE"
echo "Progress stored in: $LOG_FILE"
echo "=========================================="
