import os
import re
import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. 路径配置 (根据实际路径调整)
# ==========================================
BASE_DIR = "/data/zliu/Path_MoE"
LOG_FILE = os.path.join(BASE_DIR, "logs_subtype/BRCA_subtype_run.log")
GATING_DIR = os.path.join(BASE_DIR, "gating_subtype")
PRED_DIR = os.path.join(BASE_DIR, "predictions_subtype")

CANCER = "BRCA"
SEEDS = 20

# 映射字典: 将数字标签转为真实亚型名称
SUBTYPE_MAPPING = {
    0: 'Basal',
    1: 'Her2',
    2: 'LumA',
    3: 'LumB',
    4: 'Normal'
}

# ==========================================
# 2. 从 Log 解析每个 Seed 的最优 Top-K
# ==========================================
print("🔍 解析 Log 文件获取 K 值...")
seed_k_map = {}
with open(LOG_FILE, "r") as f:
    for line in f:
        match = re.search(r"Best Params for Seed (\d+): (\{.*\})", line)
        if match:
            seed_idx = int(match.group(1))
            params_dict = ast.literal_eval(match.group(2))
            seed_k_map[seed_idx] = params_dict["top_k"]

print(f"✅ 成功解析 {len(seed_k_map)} 个 Seed 的 K 值。\n")

# ==========================================
# 3. 计算 Selection Frequency
# ==========================================
all_seeds_freq = []

print("⚙️ 计算 20 个 Seed 的 Pathway 激活频率...")
for seed in range(SEEDS):
    k = seed_k_map.get(seed)
    if k is None: continue

    gating_csv = os.path.join(GATING_DIR, f"{CANCER}_s{seed}_gating.csv")
    detailed_csv = os.path.join(PRED_DIR, f"{CANCER}_seed{seed}_detailed.csv")
    
    if not os.path.exists(gating_csv) or not os.path.exists(detailed_csv): continue
        
    gating_df = pd.read_csv(gating_csv)
    detailed_df = pd.read_csv(detailed_csv)
    
    pathways = [col for col in gating_df.columns if col != "sample_id"]
    gating_mat = gating_df[pathways].values
    
    cutoffs = np.partition(gating_mat, -k, axis=1)[:, -k]
    activation_mask = (gating_mat >= cutoffs[:, None]).astype(float)
    
    mask_df = pd.DataFrame(activation_mask, columns=pathways)
    
    # 替换数字标签为真实的文字标签
    mask_df["Subtype"] = detailed_df["true_label"].map(SUBTYPE_MAPPING)
    
    subtype_freq = mask_df.groupby("Subtype").mean()
    all_seeds_freq.append(subtype_freq)

# ==========================================
# 4. 集成与可视化 (Ensemble & Plotting)
# ==========================================
print("📊 生成 Ensemble 热图...")

ensemble_freq_df = sum(all_seeds_freq) / len(all_seeds_freq)
ensemble_freq_df = ensemble_freq_df.T

# 整理通路名称：去掉 "HALLMARK_" 前缀，下划线替换为空格
clean_pathways = []
for p in ensemble_freq_df.index:
    clean_name = p.replace("HALLMARK_", "").replace("_", " ").title()
    clean_pathways.append(clean_name)
ensemble_freq_df.index = clean_pathways

# 🚀 核心修改：保留所有被选择过的通路（剔除总频率为0的绝对死通路）
plot_df = ensemble_freq_df[ensemble_freq_df.sum(axis=1) > 0]

# --- 绘制热图 ---
sns.set_theme(style="white")

# 🚀 核心修改：删掉 square=True，适当调低整体高度
cg = sns.clustermap(
    plot_df, 
    cmap="YlOrRd",           
    annot=False,             
    col_cluster=True,       
    row_cluster=True,        
    linewidths=.5, 
    figsize=(8, 9),      
    dendrogram_ratio=(0.2, 0.04),    # 🚀 杀手锏：(左侧树宽度比例, 顶部树高度比例)。强行把顶部的空白占
    cbar_pos=(1.05, 0.3, 0.03, 0.4), # 如果图例和右边文字重叠，把 0.95 改大一点比如 1.05
    cbar_kws={'label': 'Ensemble Selection Frequency'}
)

# 调整标题和标签字体
cg.ax_heatmap.set_ylabel("Hallmark Pathways", fontsize=12, fontweight='bold')
cg.ax_heatmap.set_xlabel("PAM50 Subtypes", fontsize=12, fontweight='bold')
plt.setp(cg.ax_heatmap.get_yticklabels(), rotation=0, fontsize=10) 
plt.setp(cg.ax_heatmap.get_xticklabels(), fontsize=11, rotation=45, ha='right')

plt.suptitle("Subtype-Specific Pathway Selection Frequency", y=1.05, fontsize=14, fontweight='bold') # y稍微调大一点防重叠

# 后面保存图表的代码不变...

output_pdf = os.path.join(BASE_DIR, f"{CANCER}_pathway_heatmap_full.pdf")
output_png = os.path.join(BASE_DIR, f"{CANCER}_pathway_heatmap_full.png")
# 使用 bbox_inches='tight' 确保移到右边的 Colorbar 不会被裁剪掉
cg.savefig(output_pdf, bbox_inches='tight', dpi=300)
cg.savefig(output_png, bbox_inches='tight', dpi=300)

print(f"🎉 大功告成！全景热图已保存至:")
print(f"   -> {output_pdf}")
print(f"   -> {output_png}")



# 1. 提取全局最热 Top 3 (Global Top Pathways)
global_mean = ensemble_freq_df.mean(axis=1)
global_top3 = global_mean.nlargest(3)

print("\n🌍 [Global Top 3] (证明模型捕捉到了乳腺癌的通用机制):")
for pathway, score in global_top3.items():
    print(f"  ➤ {pathway} (平均频率: {score:.3f})")
    print(f"    🔍 搜索建议: Breast cancer AND \"{pathway}\"")

# 2. 提取每个亚型专属的 Top 3 (Subtype-specific Top Pathways)
print("\n🎯 [Subtype-Specific Top 3] (你的核心 Story 卖点):")
for subtype in ensemble_freq_df.columns:
    print(f"\n  [{subtype} 亚型]:")
    
    # 找到在这个亚型中频率最高的 Top 1
    subtype_top3 = ensemble_freq_df[subtype].nlargest(1)
    
    for pathway, score in subtype_top3.items():
        # 顺便对比一下全局平均，看看是不是真的“特异”
        g_mean = global_mean[pathway]
        enrichment = score / (g_mean + 1e-6) # 相对全局的富集倍数
        
        # 标记出那些在当前亚型特别高，但在全局不算特别高的“真正特异性通路”
        marker = "🔥 (高度特异)" if enrichment > 1.5 else "✅"
        
        print(f"    {marker} {pathway} (当前频率: {score:.3f} | 全局: {g_mean:.3f})")
        print(f"        🔍 搜索建议: BRCA \"{subtype} subtype\" AND \"{pathway}\"")

