#!/bin/bash

# 1. 定义基本参数
CANCER="BRCA"
# 🔴 修改点 1：把 "cuda:0" 改成 "cuda"
DEVICE="cuda"
PYTHON_EXE="/data/conda_envs/moe/bin/python"

# 2. 定义 6 个组学组合消融实验 
OMICS_COMBS=("rna"  "cnv" "met" "rna_met" "cnv_met" "rna_cnv") 

# 3. 路径配置 
BASE_DIR="/data/zliu/Path_MoE"
SCRIPT_PATH="${BASE_DIR}/src/run_pathmoe_subtype_moi.py"

mkdir -p "${BASE_DIR}/results_subtype_moi"
mkdir -p "${BASE_DIR}/logs_subtype_moi"

echo "========================================================================"
echo "🔥 Task: MOI Ablation (5-Fold Ensemble + Noisy Gating)"
echo "📍 Cancer: $CANCER | Noise Std: 0.2 | Device: GPU 0"
echo "========================================================================"

# 🔴 修改点 2：在整个循环前导出环境变量，强制当前脚本只能看到第 1 张卡（索引为 0）
export CUDA_VISIBLE_DEVICES=0

for COMB in "${OMICS_COMBS[@]}"
do
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] >>> Testing Omics Combination: $COMB"

    LOG_FILE="${BASE_DIR}/logs_subtype_moi/${CANCER}_${COMB}.log"

    # 4. 核心执行命令 (严格对齐主实验参数)
    $PYTHON_EXE $SCRIPT_PATH \
        --cancer "$CANCER" \
        --omics_comb "$COMB" \
        --device "$DEVICE" \
        --seeds 20 \
        --optuna_trials 30 \
        --noise_std 0.2 \
        --num_classes 5 \
        --save_results_json "${BASE_DIR}/results_subtype_moi/ablation_${CANCER}_${COMB}.json" \
        --save_gating \
        > "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ Combination [$COMB] finished successfully!"
    else
        echo "❌ Error in [$COMB]. Check details in: $LOG_FILE"
    fi
    echo "------------------------------------------------------------------------"
done

echo "🎉 All Multi-Omics ablation experiments completed!"