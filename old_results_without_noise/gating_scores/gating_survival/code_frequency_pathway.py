import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. 参数与路径设置
# ==========================================
data_dir = "/data/zliu/Path_MoE/results/gating_scores/gating_survival/"  

all_files = glob.glob(os.path.join(data_dir, "*_s*_gating.csv"))
target_files = [f for f in all_files if "softmax" not in f]
print(f"📂 成功匹配到 {len(target_files)} 个数据文件。")

# ==========================================
# 2. 遍历文件并聚合 (直接求均值，绝不二次截断)
# ==========================================
cancer_pathway_dict = {}

for file_path in target_files:
    filename = os.path.basename(file_path)
    cancer_type = filename.split('_')[0]
    
    df = pd.read_csv(file_path, index_col=0)
    mean_weights = df.mean(axis=0)
    
    if cancer_type not in cancer_pathway_dict:
        cancer_pathway_dict[cancer_type] = []
    cancer_pathway_dict[cancer_type].append(mean_weights)

# ==========================================
# 3. 跨 Seed 求集成共识 (Ensemble Consensus)
# ==========================================
final_heatmap_data = {}
for cancer, weight_list in cancer_pathway_dict.items():
    avg_df = pd.DataFrame(weight_list)
    final_heatmap_data[cancer] = avg_df.mean(axis=0)

df_heatmap = pd.DataFrame(final_heatmap_data)

# ==========================================
# 4. 数据清洗与完美制图
# ==========================================
initial_pathway_count = df_heatmap.shape[0]

df_heatmap = df_heatmap.loc[(df_heatmap != 0).any(axis=1)]

df_heatmap.index = df_heatmap.index.str.replace('HALLMARK_', '', regex=False).str.replace('_', ' ')

removed_count = initial_pathway_count - df_heatmap.shape[0]
print(f"🧹 数据清洗: 移除了 {removed_count} 个绝对为 0 的闲置通路。")
print(f"✅ 最终制图数据矩阵: {df_heatmap.shape[1]} 个癌种, {df_heatmap.shape[0]} 个核心通路。")

# 开始绘图
plt.figure(figsize=(12, 16))

# 使用 Reds 色系代表“激活热度”，保留 standard_scale=1 以突出通路在不同癌种间的特异性
g = sns.clustermap(df_heatmap, 
                   cmap="Reds", 
                   standard_scale=1, 
                   linewidths=0.5, 
                   figsize=(12, max(8, df_heatmap.shape[0] * 0.25)), # 动态调整高度
                   cbar_pos=(0.02, 0.8, 0.03, 0.15), 
                   tree_kws=dict(linewidths=1.5),
                   xticklabels=True,
                   yticklabels=True)

# 调整字体和标签排版
g.ax_heatmap.set_xlabel("TCGA Cancer Cohorts", fontsize=14, fontweight='bold')
g.ax_heatmap.set_ylabel("MsigDB Hallmark Pathways", fontsize=14, fontweight='bold')
g.ax_heatmap.tick_params(axis='y', labelsize=9)
g.ax_heatmap.tick_params(axis='x', labelsize=11, rotation=45)

# 极其严谨的学术命名
g.ax_cbar.set_title("Relative\nExpected\nActivation", fontsize=10, fontweight='bold')
g.fig.suptitle("Pathway Dependencies (Ensemble Consensus)", fontsize=16, fontweight='bold', y=1.02)

# ==========================================
# 5. 保存结果与验证输出
# ==========================================
pdf_path = os.path.join(data_dir, "Pan_Cancer_Expected_Activation_Heatmap.pdf")
png_path = os.path.join(data_dir, "Pan_Cancer_Expected_Activation_Heatmap.png")
plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
plt.savefig(png_path, dpi=300, bbox_inches='tight')

print(f"🎉 绘图成功！文件已保存至：\n {pdf_path}\n {png_path}")

print("\n🏆 各癌种跨 20 次实验的核心驱动通路 (Top-1 期望激活)：\n" + "="*65)
for cancer in df_heatmap.columns:
    top_pathway = df_heatmap[cancer].idxmax()
    top_score = df_heatmap[cancer].max()
    print(f"[{cancer:<4}] {top_pathway:<35} (绝对集成激活分数: {top_score:.4f})")
print("="*65)