#!/bin/bash

# 1. 环境与显卡配置
export CUDA_VISIBLE_DEVICES=1
PYTHON_EXE="/data/conda_envs/moe/bin/python"
# "BLCA"  "BRCA" "HNSC" "KIRC" "LGG" "LUAD" "LUSC" "PRAD" "STAD" "THCA"
CANCERS=( "BLCA" ) 

BASE_DIR="/data/zliu/Path_MoE"
SCRIPT_PATH="${BASE_DIR}/src/run_pathmoe_survival.py" 


OUTPUT_DIR="${BASE_DIR}/results_ablation_survival"
LOG_DIR="${BASE_DIR}/logs_ablation_survival"


echo "========================================================="
echo "🔥 Path-MoE SURVIVAL Ablation: Softmax Mode"
echo "📍 GPU: 1 | Noise Std: 0.2 | Total: ${#CANCERS[@]} cancers"
echo "🌐 Results: $OUTPUT_DIR"
echo "========================================================="

for cancer in "${CANCERS[@]}"; do
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 🚀 Processing Survival: $cancer"
    
    LOG_FILE="${LOG_DIR}/${cancer}_survival_softmax.log"
    
    $PYTHON_EXE $SCRIPT_PATH \
        --cancer "$cancer" \
        --seeds 20 \
        --optuna_trials 30\
        --noise_std 0.2 \
        --use_softmax \
        --save_gating \
        --save_results_json "${OUTPUT_DIR}/${cancer}_softmax_summary.json" \
        > "$LOG_FILE" 2>&1
        
    if [ $? -eq 0 ]; then
        echo "✅ $cancer Survival Softmax finished!"
    else
        echo "❌ Error in $cancer. Check log: $LOG_FILE"
    fi
    echo "---------------------------------------------------------"
done

echo "🎉 All Survival Softmax ablation experiments completed!"