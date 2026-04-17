import os
import glob
import json
import pandas as pd

results_dir = "/data/zliu/Path_MoE/results_survival_ablation"

data = []

print("🔍 正在扫描和解析 JSON 文件...")

for file_path in glob.glob(os.path.join(results_dir, "ablation_*.json")):
    with open(file_path, "r") as f:
        try:
            res = json.load(f)
            cancer = res.get("cancer")
            comb = res.get("comb")
            mean_auc = res.get("mean_test_auc")
            std_auc = res.get("std_test_auc")
            
            # 确保关键数据存在
            if None not in (cancer, comb, mean_auc, std_auc):
                # 核心要求：保留三位小数，并拼接成 mean ± sd
                formatted_auc = f"{mean_auc:.3f} ± {std_auc:.3f}"
                data.append({
                    "Cancer": cancer,
                    "Combination": comb,
                    "AUC_String": formatted_auc
                })
        except Exception as e:
            print(f"⚠️ 解析 {os.path.basename(file_path)} 时出错: {e}")

# 转换为 DataFrame
df = pd.DataFrame(data)

if not df.empty:
    # 制作透视表：行是癌种 (Cancer)，列是组合 (Combination)
    pivot_df = df.pivot(index="Cancer", columns="Combination", values="AUC_String")
    
    # 整理列的显示顺序：单组学 -> 双组学 -> 多组学
    ideal_order = ['rna', 'cnv', 'met', 'rna_cnv', 'rna_met', 'cnv_met', 'all']
    # 筛选出实际存在的列，并保持理想排序
    existing_cols = [col for col in ideal_order if col in pivot_df.columns]
    # 把其他不在预期内但也跑了的组合放在最后
    other_cols = [col for col in pivot_df.columns if col not in ideal_order]
    
    pivot_df = pivot_df[existing_cols + other_cols]

    # 打印到终端查看
    print("\n✅ 提取完成！整理后的消融实验 AUC 结果如下：\n")
    print(pivot_df.to_markdown())
    
    # 存为 CSV 文件，方便复制到 Excel / Word
    output_csv = os.path.join(results_dir, "ablation_summary_auc.csv")
    pivot_df.to_csv(output_csv)
    print(f"\n📁 汇总表格已成功保存至: {output_csv}")
    print("💡 提示: 你可以直接下载这个 CSV 文件用 Excel 打开，格式已经完全排好。")
else:
    print("❌ 未找到任何有效的数据，请检查 JSON 文件路径和格式。")