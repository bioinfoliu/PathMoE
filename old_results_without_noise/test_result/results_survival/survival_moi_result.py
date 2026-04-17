import os
import json
import numpy as np
import pandas as pd

def generate_survival_ablation_summary():
    # 1. 基础配置
    base_dir = "/data/zliu/Path_MoE/results/test_result/results_survival"
    
    # 定义十种癌症类型
    cancer_types = [
        "BRCA", "HNSC", "LGG", "THCA", "PRAD", 
        "LUAD", "BLCA", "STAD", "LUSC", "KIRC"
    ]
    
    combinations = {
        "rna": "RNA",
        "cnv": "CNV",
        "met": "MET",
        "rna_cnv": "RNA+CNV",
        "rna_met": "RNA+MET",
        "cnv_met": "CNV+MET",
        "all": "Path-MoE"
    }
    
    # 用于存储所有结果的长列表
    all_rows = []

    print(f"🔍 正在目录 {base_dir} 中处理 {len(cancer_types)} 种癌症的消融实验结果...")

    for cancer in cancer_types:
        row_data = {"Cancer": cancer}
        
        for combo_key, display_name in combinations.items():
            file_name = f"ablation_{cancer}_{combo_key}.json"
            file_path = os.path.join(base_dir, file_name)
            
            res_str = "N/A" # 默认值
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # 智能提取列表数据
                    metrics = None
                    if isinstance(data, list):
                        metrics = data
                    elif isinstance(data, dict):
                        # 尝试寻找可能的 key
                        for key in ['metrics', 'auc', 'Path-MoE', cancer]:
                            if key in data and isinstance(data[key], list):
                                metrics = data[key]
                                break
                        # 如果还没找到，取第一个列表类型的 value
                        if not metrics:
                            for v in data.values():
                                if isinstance(v, list):
                                    metrics = v
                                    break

                    if metrics and len(metrics) > 0:
                        mean_val = np.mean(metrics)
                        std_val = np.std(metrics)
                        res_str = f"{mean_val:.3f}({std_val:.3f})"
                    else:
                        res_str = "Parse Error"
                        
                except Exception as e:
                    print(f"  [错误] 处理 {file_name} 失败: {e}")
                    res_str = "Error"
            else:
                # 记录缺失文件，但不中断程序
                pass
            
            # 将该组合的结果存入当前癌症的字典中
            row_data[display_name] = res_str
            
        all_rows.append(row_data)

    df = pd.DataFrame(all_rows)
    
    # 调整列顺序，确保 Cancer 在第一列，后面跟着 combinations 定义的顺序
    cols = ["Cancer"] + list(combinations.values())
    df = df[cols]

    # 4. 打印汇总结果
    print("\n" + "="*80)
    print("🏆 Multi-Omics Survival Ablation Summary (Mean Test AUC ± SD)")
    print("="*80)
    print(df.to_markdown(index=False))
    print("="*80)

    # 5. 导出文件
    out_csv = os.path.join(base_dir, "pan_cancer_ablation_summary.csv")
    df.to_csv(out_csv, index=False)
    
    # 同时生成一个适合直接粘贴到 LaTeX 的格式文件 (可选)
    out_tex = os.path.join(base_dir, "pan_cancer_ablation_summary.tex")
    df.to_latex(out_tex, index=False)
    
    print(f"\n✨ 汇总表已导出至: {out_csv}")
    print(f"✨ LaTeX 代码片段已生成至: {out_tex}")

if __name__ == "__main__":
    generate_survival_ablation_summary()