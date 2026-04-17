#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
PYTHON_EXE="/data/conda_envs/moe/bin/python"
CANCERS=(  "BLCA" "BRCA" "HNSC" "KIRC" "LGG" "LUAD" "LUSC" "PRAD" "STAD" "THCA")

# 4. 路径配置
BASE_DIR="/data/zliu/Path_MoE"
SCRIPT_PATH="${BASE_DIR}/src/run_pathmoe_survival.py"
LOG_DIR="/data/zliu/Path_MoE/final_logs/logs_survival"

# 预先创建所有输出目录
mkdir -p $LOG_DIR
mkdir -p "${BASE_DIR}/results"
mkdir -p "${BASE_DIR}/predictions"
mkdir -p "${BASE_DIR}/gating"
mkdir -p "${BASE_DIR}/checkpoints"

echo "========================================================="
echo "🛰️  Path-MoE Pan-Cancer Mission Control"
echo "🌐 Env: /data/conda_envs/moe"
echo "📍 GPU: $CUDA_VISIBLE_DEVICES | Mode: Single-Model + Noisy Gating"
echo "📂 Saving: Models, Predictions, Gating, and JSON Results"
echo "========================================================="

for CANCER in "${CANCERS[@]}"
do
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 🚀 Processing: $CANCER ..."
    
    LOG_FILE="${LOG_DIR}/${CANCER}_run.log"
    
    # 使用绝对路径的 Python 运行
    $PYTHON_EXE $SCRIPT_PATH \
        --cancer "$CANCER" \
        --seeds 20 \
        --optuna_trials 30 \
        --noise_std 0.2 \
        --num_classes 1 \
        --save_gating \
        --save_models \
        --save_results_json "results/${CANCER}_final_results.json" \
        > "$LOG_FILE" 2>&1
    
    # 检查状态
    if [ $? -eq 0 ]; then
        echo "✅ $CANCER Finished Successfully."
    else
        echo "❌ $CANCER Failed. Check log: $LOG_FILE"
    fi
    echo "---------------------------------------------------------"
done

echo "🏆 Mission Accomplished. All 10 cohorts processed."