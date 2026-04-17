#!/bin/bash

# 1. 环境配置
export CUDA_VISIBLE_DEVICES=1
PYTHON_EXE="/data/conda_envs/moe/bin/python"

CANCERS=("BRCA" )

BASE_DIR="/data/zliu/Path_MoE"
SCRIPT_PATH="${BASE_DIR}/src/run_pathmoe_subtype.py" # 确保指向最新的 5-CV 脚本

mkdir -p results_ablation_subtype
mkdir -p logs_ablation_subtype

echo "========================================="
echo "🚀 Starting Softmax Ablation Study (5-Fold Ensemble + Noise)"
echo "📍 GPU: 1 | Noise Std: 0.2 | Total: ${#CANCERS[@]} cancers"
echo "========================================="

for cancer in "${CANCERS[@]}"; do
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] >>> Processing: $cancer"
    
    # 核心提交命令：
    $PYTHON_EXE $SCRIPT_PATH \
        --cancer "$cancer" \
        --seeds 20 \
        --optuna_trials 30 \
        --noise_std 0.2 \
        --use_softmax \
        --save_gating \
        --save_results_json "results_ablation/${cancer}_softmax_summary.json" \
        > "logs_ablation_subtype/${cancer}_softmax.log" 2>&1
        
    if [ $? -eq 0 ]; then
        echo "✅ $cancer Softmax ablation finished!"
    else
        echo "❌ Error in $cancer. Check logs_ablation/${cancer}_softmax.log"
    fi
    echo "-----------------------------------------"
done

echo "🎉 All Softmax ablation experiments completed!"