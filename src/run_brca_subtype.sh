#!/bin/bash

# 1. 锁定使用 GPU 1
export CUDA_VISIBLE_DEVICES=1

# 2. 指定 Conda 环境下的 Python 解释器
PYTHON_EXE="/data/conda_envs/moe/bin/python"

# 3. 任务目标：仅 BRCA (5-Class Subtype)
CANCERS=("BRCA")

# 4. 路径配置 - 必须指向你的 subtype 脚本
BASE_DIR="/data/zliu/Path_MoE"
SCRIPT_PATH="${BASE_DIR}/src/run_pathmoe_subtype.py"
LOG_DIR="${BASE_DIR}/logs_subtype"

mkdir -p $LOG_DIR
mkdir -p "${BASE_DIR}/results_subtype"      # 汇总 JSON
mkdir -p "${BASE_DIR}/predictions_subtype"  # 每一个病人的预测 CSV
mkdir -p "${BASE_DIR}/gating_subtype"       # 门控权重 CSV
mkdir -p "${BASE_DIR}/checkpoints_subtype"   # 模型 .pth 文件

echo "========================================================="
echo "🧬 Path-MoE Subtype Classification: BRCA ONLY"
echo "🌐 Env: /data/conda_envs/moe"
echo "📍 GPU: $CUDA_VISIBLE_DEVICES | Mode: Single-Model + Noisy Gating"
echo "📊 Task: 5-Class Classification | Seeds: 20"
echo "📂 Saving: Models, Predictions, Gating, and Results"
echo "========================================================="

for CANCER in "${CANCERS[@]}"
do
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 🚀 Processing: $CANCER ..."
    
    LOG_FILE="${LOG_DIR}/${CANCER}_subtype_run.log"
    
    $PYTHON_EXE $SCRIPT_PATH \
        --cancer "$CANCER" \
        --seeds  20\
        --optuna_trials 30 \
        --noise_std 0.5 \
        --use_tri_gating \
        --num_classes 5 \
        --save_gating \
        --save_models \
        --save_results_json "results_subtype/${CANCER}_subtype_summary.json" \
        > "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ $CANCER Subtype Task Finished Successfully."
    else
        echo "❌ $CANCER Subtype Task Failed. Check log: $LOG_FILE"
    fi
    echo "---------------------------------------------------------"
done

echo "🏆 All BRCA Subtype seed-experiments completed."