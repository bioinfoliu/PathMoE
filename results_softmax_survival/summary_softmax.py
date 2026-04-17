import json
import os
import numpy as np
import pandas as pd

# 1. 定义癌种和路径 
cancers = [ "BLCA", "BRCA", "HNSC", "KIRC", "LGG", "LUAD", "LUSC", "PRAD", "STAD", "THCA"]
base_dir = "/data/zliu/Path_MoE/results_ablation_survival"

results_list = []

# 2. 遍历读取 JSON 文件
for cancer in cancers:
    file_path = os.path.join(base_dir, f"{cancer}_softmax_summary.json")
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            
            # 提取 Mean 和 SD
            mean_auc = data.get("mean_test_auc", np.nan)
            std_auc = data.get("std_test_auc", np.nan)
            
            # 格式化为 Mean ± SD (保留三位小数)
            formatted_res = f"{mean_auc:.3f} ± {std_auc:.3f}"
            results_list.append({
                "Cancer Type": cancer, 
                "Softmax AUC (Mean ± SD)": formatted_res
            })
    else:
        print(f"⚠️ 找不到文件: {file_path}")
        results_list.append({
            "Cancer Type": cancer, 
            "Softmax AUC (Mean ± SD)": "N/A"
        })

# 3. 转换为 DataFrame 并打印
df = pd.DataFrame(results_list)

print("\n" + "="*50)
print("📊 终端查看格式 (Markdown):")
print("="*50)
print(df.to_markdown(index=False))

print("\n" + "="*50)
print("📝 论文/Beamer 填表格式 (LaTeX):")
print("="*50)
for index, row in df.iterrows():
    # 生成可以直接插入 LaTeX 表格的代码行
    print(f"{row['Cancer Type']} & {row['Softmax AUC (Mean ± SD)']} \\\\")
print("="*50)