import os
import glob
import pandas as pd
import numpy as np
import torch
from collections import Counter

# ==========================================
# 1. 参数与路径设置
# ==========================================
data_dir = "/data/zliu/gating_survival/"  # 存放 gating.csv 的目录
ckpt_dir = "/data/zliu/checkpoints/" # 存放 .pth 权重的目录

print(f"🔍 正在扫描 {data_dir} 目录下的门控权重文件...")

all_files = glob.glob(os.path.join(data_dir, "*_s*_gating.csv"))
target_files = sorted([f for f in all_files if "softmax" not in f])

if not target_files:
    raise FileNotFoundError("未找到符合条件的 _gating.csv 文件，请检查路径。")

print(f"📂 成功匹配到 {len(target_files)} 个数据文件，开始计算异质性...")

# ==========================================
# 2. 核心分析逻辑
# ==========================================
results = []

for file_path in target_files:
    filename = os.path.basename(file_path)
    parts = filename.replace('_gating.csv', '').split('_')
    cancer_type = parts[0]
    seed_str = parts[1]
    
    # 动态获取当前实验的 k 值
    k = 2 # 默认值
    try:
        seed_num = seed_str.replace('s', '')
        ckpt_path = os.path.join(ckpt_dir, f"{cancer_type}_seed{seed_num}_fold1.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            k = checkpoint.get('top_k', 2) 
    except Exception:
        pass 
        
    # 读取门控权重
    df = pd.read_csv(file_path, index_col=0)
    weights = df.values # [病人数, 50 个通路]
    pathway_names = df.columns.tolist()
    
    # 提取每个病人的 Top-k 通路索引，并横向排序
    # 排序是为了确保 [通路A, 通路B] 和 [通路B, 通路A] 被视作同一种组合
    top_k_indices = np.sort(np.argsort(weights, axis=1)[:, -k:], axis=1)
    
    # 将索引转化为实际的通路名称组合 (Tuple 格式方便哈希和统计)
    combinations = []
    for row in top_k_indices:
        combo = tuple([pathway_names[idx] for idx in row])
        combinations.append(combo)
        
    # 统计独立组合的数量
    unique_combinations = set(combinations)
    num_samples = len(combinations)
    num_unique = len(unique_combinations)
    
    # 计算“最大共识率” (即最常见的那一种组合，占了所有病人的百分之几？)
    combo_counts = Counter(combinations)
    most_common_combo, most_common_count = combo_counts.most_common(1)[0]
    consensus_rate = most_common_count / num_samples
    
    # 记录结果
    results.append({
        'Cancer': cancer_type,
        'Seed': seed_str,
        'k_value': k,
        'Total_Patients': num_samples,
        'Unique_Combos': num_unique,
        'Consensus_Rate': f"{consensus_rate:.2%}",
        'Most_Common_Combo': " + ".join(most_common_combo)
    })

# ==========================================
# 3. 打印分析结果
# ==========================================
results_df = pd.DataFrame(results)

# 打印美化后的表格
print("\n" + "="*100)
print(f"{'Cancer':<8} | {'Seed':<5} | {'k':<2} | {'Patients':<8} | {'Unique Combos':<13} | {'Consensus':<10} | {'Most Common Pathway Combo'}")
print("-" * 100)

for _, row in results_df.iterrows():
    # 为了防止路径名字太长导致换行，截断显示组合名字
    combo_display = row['Most_Common_Combo'][:40] + "..." if len(row['Most_Common_Combo']) > 40 else row['Most_Common_Combo']
    print(f"{row['Cancer']:<8} | {row['Seed']:<5} | {row['k_value']:<2} | {row['Total_Patients']:<8} | {row['Unique_Combos']:<13} | {row['Consensus_Rate']:<10} | {combo_display}")

print("="*100)

# 如果你想保存这个结果：
results_df.to_csv(os.path.join(data_dir, "Patient_Heterogeneity_Report.csv"), index=False)


df = results_df.copy()

print("🏆 各癌种跨 20 次实验的 Top 1 核心驱动通路：\n" + "="*60)

top_pathways_dict = {}

# 按癌种分组统计
for cancer, group in df.groupby('Cancer'):
    all_pathways_in_cancer = []
    
    # 遍历该癌种的 20 个 Seed
    for combo_str in group['Most_Common_Combo']:
        # 将 "PATHWAY_A + PATHWAY_B" 拆分成独立通路
        pathways = [p.strip() for p in combo_str.split('+')]
        all_pathways_in_cancer.extend(pathways)
        
    # 统计出现次数最多的那 1 个
    pathway_counts = Counter(all_pathways_in_cancer)
    top_1_pathway, count = pathway_counts.most_common(1)[0]
    
    # 计算在 20 个 Seed 中的出现概率 (注意：这里的概率是指在 20 次实验中露脸的比例)
    appearance_rate = count / 20.0 
    top_pathways_dict[cancer] = top_1_pathway
    
    # 清理名字，方便阅读
    clean_name = top_1_pathway.replace('HALLMARK_', '')
    print(f"[{cancer:<4}] {clean_name:<30} (命中率: {appearance_rate:.0%}, {count}/20)")

print("="*60)



import pandas as pd

print("🏆 各癌种跨 20 次实验的真实患者级 Top 1 通路 (与热图完全一致)：\n" + "="*65)
