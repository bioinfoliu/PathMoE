### subgroup 的样本过滤文件

import pandas as pd
import os

df_expr_raw = pd.read_csv(expression_file)

df_expr = df_expr_raw.set_index(df_expr_raw.columns[0]).T
df_expr.index.name = 'sampleID'
df_expr = df_expr.reset_index()

# 获取转置后的样本列表
expr_samples = set(df_expr['sampleID'].unique())
print(f"表达谱转置完成，总样本数: {len(expr_samples)}")

# 4. 取交集 / Find intersection
common_samples = clinical_samples.intersection(expr_samples)
print(f"求得交集样本数: {len(common_samples)}")

# 5. 根据交集提取数据 / Extract data based on intersection
df_clinical_final = df_clinical_labeled[df_clinical_labeled['sampleID'].isin(common_samples)]
df_expr_final = df_expr[df_expr['sampleID'].isin(common_samples)]

# 按样本 ID 排序以确保完全对齐 / Sort by ID to ensure alignment
df_clinical_final = df_clinical_final.sort_values('sampleID')
df_expr_final = df_expr_final.sort_values('sampleID')

# 6. 保存提取结果 / Save results
output_dir = "extracted_data"
os.makedirs(output_dir, exist_ok=True)

df_clinical_final.to_csv(f"{output_dir}/BRCA_clinical_pam50_filtered.csv", index=False)
df_expr_final.to_csv(f"{output_dir}/BRCA_hallmark_tpm_pam50_filtered.csv", index=False)

print("\n✅ 提取与转置完成！")
print(f"样本 ID 列名已统一为 'sampleID'")
print(f"临床标签保存至: {output_dir}/BRCA_clinical_pam50_filtered.csv")
print(f"转置后的表达矩阵保存至: {output_dir}/BRCA_hallmark_tpm_pam50_filtered.csv")

# 打印亚型统计 / Print subtype stats
print("\n亚型分布统计:")
print(df_clinical_final['PAM50Call_RNAseq'].value_counts())