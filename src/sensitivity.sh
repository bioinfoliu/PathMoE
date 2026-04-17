#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
PYTHON_EXE="/data/conda_envs/moe/bin/python"

CANCERS=("BRCA")

NOISE_STDS=(0.6 0.8)

# 5. 路径配置 - 指向你的 subtype 脚本
BASE_DIR="/data/zliu/Path_MoE"
SCRIPT_PATH="${BASE_DIR}/src/run_pathmoe_subtype.py"

# 为敏感性分析创建专属文件夹，避免覆盖历史数据
LOG_DIR="${BASE_DIR}/logs_subtype_sensitivity"
mkdir -p "$LOG_DIR"
mkdir -p "${BASE_DIR}/results_subtype_sensitivity"     
mkdir -p "${BASE_DIR}/predictions_subtype_sensitivity"  
mkdir -p "${BASE_DIR}/gating_subtype_sensitivity"     
mkdir -p "${BASE_DIR}/checkpoints_subtype_sensitivity" 

# 创建一个专门记录时间的汇总文件
TIME_LOG="${BASE_DIR}/results_subtype_sensitivity/timing_summary.txt"
echo "Cancer | Noise_Std | Duration_Seconds | Duration_Formatted" > "$TIME_LOG"

echo "========================================================="
echo "🧬 Path-MoE Subtype Classification: Sensitivity & Timing Analysis"
echo "🌐 Env: /data/conda_envs/moe"
echo "📍 GPU: $CUDA_VISIBLE_DEVICES | Cancer: BRCA"
echo "📊 Task: 5-Class Classification | Seeds: 20"
echo "🔍 Noise levels to test: ${NOISE_STDS[*]}"
echo "📂 Saving: Models, Predictions, Gating, Results, and Timing"
echo "========================================================="

for CANCER in "${CANCERS[@]}"
do
    for NOISE in "${NOISE_STDS[@]}"
    do
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] 🚀 Processing: $CANCER with noise_std = $NOISE ..."
        
        LOG_FILE="${LOG_DIR}/${CANCER}_noise_${NOISE}_run.log"
        RESULT_JSON="results_subtype_sensitivity/${CANCER}_noise_${NOISE}_summary.json"
        
        # 记录开始时间 (秒)
        START_TIME=$(date +%s)
        
        $PYTHON_EXE $SCRIPT_PATH \
            --cancer "$CANCER" \
            --seeds 20 \
            --optuna_trials 30 \
            --noise_std "$NOISE" \
            --num_classes 5 \
            --save_gating \
            --save_models \
            --save_results_json "$RESULT_JSON" \
            > "$LOG_FILE" 2>&1
        
        EXIT_CODE=$?
        
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        HOURS=$((DURATION / 3600))
        MINUTES=$(( (DURATION % 3600) / 60 ))
        SECONDS=$((DURATION % 60))
        FORMATTED_TIME=$(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ $CANCER (Noise $NOISE) Finished. Time Taken: $FORMATTED_TIME"
            # 写入时间日志
            echo "$CANCER | $NOISE | $DURATION | $FORMATTED_TIME" >> "$TIME_LOG"
        else
            echo "❌ $CANCER (Noise $NOISE) Task Failed. Check log: $LOG_FILE"
            echo "$CANCER | $NOISE | FAILED | FAILED" >> "$TIME_LOG"
        fi
        echo "---------------------------------------------------------"
    done
done

echo "🏆 All BRCA Sensitivity Analysis completed. Timing summary saved to $TIME_LOG."