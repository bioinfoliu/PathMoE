#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

DEVICE="cuda" 
PYTHON_EXE="/data/conda_envs/moe/bin/python" 


CANCERS=("BLCA" "BRCA" "HNSC" "KIRC" "LGG" "LUAD" "LUSC" "PRAD" "STAD" "THCA")

OMICS_COMBS=("rna" "cnv" "met" "rna_cnv" "rna_met" "cnv_met")

BASE_DIR="/data/zliu/Path_MoE"
SCRIPT_PATH="${BASE_DIR}/src/run_pathmoe_survival_moi.py"

mkdir -p "${BASE_DIR}/results_survival_moi"
mkdir -p "${BASE_DIR}/logs_survival_moi"

echo "========================================================================"
echo "🔥 准备开始 10 癌种 x 6 组合 = 60 组生存预测消融实验..."
echo "📍 核心配置: 5-Fold Ensemble | Noise Std: 0.2 | Num Classes: 1"
echo "========================================================================"

for CANCER in "${CANCERS[@]}"
do
    echo ""
    echo "🎯 当前进度: 开始跑 $CANCER"
    echo "------------------------------------------------------------------------"

    for COMB in "${OMICS_COMBS[@]}"
    do
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] 🚀 Task: Survival MOI | Cancer: $CANCER | Omics: $COMB"
        
        LOG_FILE="${BASE_DIR}/logs_survival_moi/${CANCER}_${COMB}.log"
        
        # 运行生存期消融脚本 (严格对齐主实验参数)
        $PYTHON_EXE $SCRIPT_PATH \
            --cancer "$CANCER" \
            --omics_comb "$COMB" \
            --device "$DEVICE" \
            --seeds 20 \
            --optuna_trials 30 \
            --noise_std 0.2 \
            --num_classes 1 \
            --save_results_json "${BASE_DIR}/results_survival_ablation/ablation_${CANCER}_${COMB}.json" \
            --save_gating \
            > "$LOG_FILE" 2>&1
        
        # 检查是否成功执行
        if [ $? -eq 0 ]; then
            echo "   ✅ [$COMB] 运行完毕!"
        else
            echo "   ❌ [$COMB] 发生错误，请查看日志: $LOG_FILE"
        fi
    done
done