#!/bin/bash

USE_TRI=1
export CUDA_VISIBLE_DEVICES=1

PYTHON_EXE="/data/conda_envs/moe/bin/python"
BASE_DIR="/data/zliu/Path_MoE"
SCRIPT_PATH="${BASE_DIR}/src/run_pathmoe_subtype.py"
# ==========================================

if [ $USE_TRI -eq 1 ]; then
    SUFFIX="tri_gating"
    EXTRA_ARGS="--use_tri_gating"
    MODE_TEXT="Tri-omics Router (Ablation Study)"
else
    SUFFIX="standard"
    EXTRA_ARGS=""
    MODE_TEXT="RNA-only Router (Standard MoPE)"
fi

LOG_DIR="${BASE_DIR}/logs_subtype"
mkdir -p $LOG_DIR
mkdir -p "${BASE_DIR}/results_subtype"
mkdir -p "${BASE_DIR}/predictions_subtype"
mkdir -p "${BASE_DIR}/gating_subtype"
mkdir -p "${BASE_DIR}/checkpoints_subtype"

CANCERS=("BRCA")

echo "========================================================="
echo "🧬 Path-MoE Subtype Classification: BRCA"
echo "🌐 Env: /data/conda_envs/moe"
echo "📍 GPU: $CUDA_VISIBLE_DEVICES | 🚀 Mode: $MODE_TEXT"
echo "📂 Suffix: $SUFFIX"
echo "========================================================="

for CANCER in "${CANCERS[@]}"
do
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 🚀 Processing: $CANCER ..."
    
    LOG_FILE="${LOG_DIR}/${CANCER}_subtype_${SUFFIX}.log"
    JSON_OUT="results_subtype/${CANCER}_subtype_${SUFFIX}_summary.json"
    
    $PYTHON_EXE $SCRIPT_PATH \
        --cancer "$CANCER" \
        --seeds 20 \
        --optuna_trials 30 \
        --noise_std 0.5 \
        --num_classes 5 \
        --save_gating \
        --save_models \
        --save_results_json "$JSON_OUT" \
        $EXTRA_ARGS \
        > "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ $CANCER Task ($SUFFIX) Finished Successfully."
        echo "   Results saved to: $JSON_OUT"
    else
        echo "❌ $CANCER Task ($SUFFIX) Failed. Check log: $LOG_FILE"
    fi
    echo "---------------------------------------------------------"
done

echo "🏆 All experiments completed."