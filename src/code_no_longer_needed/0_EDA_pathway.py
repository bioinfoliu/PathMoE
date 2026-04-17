import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings

# 忽略 Seaborn 的一些常规警告以保持终端整洁
warnings.filterwarnings("ignore")

# ================= 配置区 =================
# 建议使用绝对路径
BASE_DIR = "/data/zliu/Path_MoE"
GMT_FILE = os.path.join(BASE_DIR, "data", "h.all.v2023.1.Hs.symbols.gmt")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "eda_pathway")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 取消了单独的 SIZE 图片，合并为 DIST_COMBINED
OUTPUT_IMG_DIST_COMBINED = os.path.join(OUTPUT_DIR, "pathway_and_overlap_distributions.png")
OUTPUT_IMG_HEATMAP = os.path.join(OUTPUT_DIR, "overlap_heatmap.png")

def read_gmt(gmt_path):
    """读取 GMT 文件并返回 {pathway: set(genes)} 字典"""
    pathway_dict = {}
    if not os.path.exists(gmt_path):
        print(f"❌ Error: GMT file not found at {gmt_path}")
        return {}
    
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            name = parts[0].replace("HALLMARK_", "")
            genes = set(parts[2:]) 
            pathway_dict[name] = genes
            
    print(f"📘 Successfully loaded {len(pathway_dict)} pathways from GMT.")
    return pathway_dict

def analyze_pathway_sizes(pathway_dict):
    """分析通路自身的大小（基因数量）并打印，返回数据用于后续绘图"""
    data = [{"Pathway": name, "Gene_Count": len(genes)} for name, genes in pathway_dict.items()]
    df = pd.DataFrame(data)
    
    print("\n" + "="*50)
    print(f"📊 Pathway Size Statistics (Total: {len(df)})")
    print("="*50)
    print(df["Gene_Count"].describe().to_string())
    print("-" * 50)
    
    max_row = df.loc[df['Gene_Count'].idxmax()]
    min_row = df.loc[df['Gene_Count'].idxmin()]
    print(f"🔴 Largest Pathway : {max_row['Pathway']} ({max_row['Gene_Count']} genes)")
    print(f"🔵 Smallest Pathway: {min_row['Pathway']} ({min_row['Gene_Count']} genes)")
    print("="*50)

    # 不再这里画图，直接返回数据给主绘图函数
    return df["Gene_Count"]

def analyze_overlaps(pathway_dict):
    """分析通路之间的基因重叠矩阵"""
    pathways = sorted(list(pathway_dict.keys()))
    n = len(pathways)
    
    overlap_matrix = np.zeros((n, n))
    overlap_pairs = [] 
    
    print(f"\n🔄 Calculating overlaps for {n} pathways ({n*(n-1)//2} combinations)...")
    
    for i in range(n):
        for j in range(i+1, n):
            name_a, name_b = pathways[i], pathways[j]
            intersection = len(pathway_dict[name_a] & pathway_dict[name_b])
            
            overlap_matrix[i, j] = intersection
            overlap_matrix[j, i] = intersection
            overlap_pairs.append(intersection)
            
        # 对角线填充为自身长度，用于热图展示
        overlap_matrix[i, i] = len(pathway_dict[pathways[i]])

    df_matrix = pd.DataFrame(overlap_matrix, index=pathways, columns=pathways)
    return df_matrix, overlap_pairs

def plot_distributions_and_heatmap(df_matrix, overlap_pairs, gene_counts):
    """绘制组合分布图与聚类热图"""
    
    # ================= 图 1: 大小分布与重叠分布的组合图 =================
    plt.figure(figsize=(14, 6))
    
    # 左子图：通路自身大小分布 (Histogram)
    plt.subplot(1, 2, 1)
    sns.histplot(gene_counts, bins=15, kde=True, color="skyblue", edgecolor="black")
    mean_val_size = gene_counts.mean()
    median_val_size = gene_counts.median()
    plt.axvline(mean_val_size, color='red', linestyle='--', label=f"Mean: {mean_val_size:.1f}")
    plt.axvline(median_val_size, color='green', linestyle='--', label=f"Median: {median_val_size:.1f}")
    plt.title("Distribution of Pathway Sizes", fontweight='bold')
    plt.xlabel("Number of Genes in Pathway")
    plt.ylabel("Frequency")
    plt.legend()

    # 右子图：重叠对分布 (Barplot)
    plt.subplot(1, 2, 2)
    overlap_counts = pd.Series(overlap_pairs).value_counts().sort_index()
    sns.barplot(x=overlap_counts.index, y=overlap_counts.values, color="royalblue", edgecolor="black")
    plt.title(f"Distribution of Gene Overlaps\n(Total Pairs: {len(overlap_pairs)})", fontweight='bold')
    plt.xlabel("Number of Overlapping Genes")
    plt.ylabel("Count of Pairs (Log Scale)")
    plt.yscale("log")
    
    ax = plt.gca()
    for ind, label in enumerate(ax.get_xticklabels()):
        if ind % 5 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)

    mean_val_ov = np.mean(overlap_pairs)
    max_val_ov = np.max(overlap_pairs)
    
    if mean_val_ov in overlap_counts.index:
         mean_idx = overlap_counts.index.get_loc(mean_val_ov)
    else:
         mean_idx = np.searchsorted(overlap_counts.index, mean_val_ov)
    plt.axvline(mean_idx, color='blue', linestyle='--', label=f"Mean Overlap: {mean_val_ov:.2f}")
    
    if max_val_ov in overlap_counts.index:
         max_idx = overlap_counts.index.get_loc(max_val_ov)
         plt.axvline(max_idx, color='red', linestyle='--', label=f"Max Overlap: {max_val_ov}")
         
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_DIST_COMBINED, dpi=300)
    print(f"✅ Combined distributions plot saved to: {OUTPUT_IMG_DIST_COMBINED}")
    plt.close()
    
    # ================= 图 2: 聚类热图 =================
    mask_upper = np.triu(np.ones_like(df_matrix, dtype=bool), k=1)
    mask_diag = np.eye(df_matrix.shape[0], dtype=bool)
    mask_lower = np.tril(np.ones_like(df_matrix, dtype=bool), k=-1)

    fig, ax = plt.subplots(figsize=(16, 14))
    cmap_diag = sns.color_palette("Blues", as_cmap=True)
    cmap_offdiag = sns.color_palette("OrRd", as_cmap=True)
    
    # 绘制非对角线
    mask_offdiag = mask_upper | mask_diag
    sns.heatmap(df_matrix, mask=mask_offdiag, cmap=cmap_offdiag, 
                ax=ax, cbar=False, xticklabels=True, yticklabels=True)
                
    # 绘制对角线
    mask_only_diag = ~mask_diag
    sns.heatmap(df_matrix, mask=mask_only_diag, cmap=cmap_diag, 
                ax=ax, cbar=False, xticklabels=True, yticklabels=True)

    cbar_ax_offdiag = fig.add_axes([0.92, 0.55, 0.02, 0.3])
    cbar_ax_diag = fig.add_axes([0.92, 0.15, 0.02, 0.3])
    
    vmin_offdiag = df_matrix.values[mask_lower].min() if np.any(mask_lower) else 0
    vmax_offdiag = df_matrix.values[mask_lower].max() if np.any(mask_lower) else 1
    vmin_diag = df_matrix.values[mask_diag].min()
    vmax_diag = df_matrix.values[mask_diag].max()

    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_offdiag, norm=plt.Normalize(vmin=vmin_offdiag, vmax=vmax_offdiag)),
                 cax=cbar_ax_offdiag, label='Number of Shared Genes (Overlaps)')
                 
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_diag, norm=plt.Normalize(vmin=vmin_diag, vmax=vmax_diag)),
                 cax=cbar_ax_diag, label='Total Genes in Pathway (Diagonal)')

    ax.set_title("Pathway-Pathway Overlap Matrix", fontsize=22, pad=20, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.savefig(OUTPUT_IMG_HEATMAP, dpi=300, bbox_inches='tight')
    print(f"✅ Overlap heatmap saved to: {OUTPUT_IMG_HEATMAP}")
    plt.close()

def print_top_overlaps(pathway_dict, df_matrix):
    """计算并打印 Jaccard 相似度最高的前 10 个通路对"""
    df_matrix_tri = df_matrix.where(np.triu(np.ones(df_matrix.shape), k=1).astype(bool))
    stack = df_matrix_tri.stack().reset_index()
    stack.columns = ['Pathway_A', 'Pathway_B', 'Overlap_Count']
    
    top_candidates = stack.sort_values('Overlap_Count', ascending=False).head(15)
    
    print("\n" + "="*60)
    print("🔥 TOP 10 Most Overlapping Pathway Pairs (By Absolute Count)")
    print("="*60)
    
    results = []
    for _, row in top_candidates.iterrows():
        p1, p2 = row['Pathway_A'], row['Pathway_B']
        count = int(row['Overlap_Count'])
        total_p1, total_p2 = len(pathway_dict[p1]), len(pathway_dict[p2])
        
        union = total_p1 + total_p2 - count
        jaccard = count / union
        results.append({'p1': p1, 'p2': p2, 'count': count, 'jaccard': jaccard, 't1': total_p1, 't2': total_p2})
    
    results = sorted(results, key=lambda x: x['count'], reverse=True)[:10]
    for r in results:
        print(f"{r['count']:3d} shared genes | Jaccard: {r['jaccard']:.2f} | {r['p1']} ({r['t1']}) <--> {r['p2']} ({r['t2']})")
    print("="*60 + "\n")

if __name__ == "__main__":
    print("🚀 Starting Pathway EDA Pipeline...")
    pathways = read_gmt(GMT_FILE)
    if pathways:
        # 1. 提取通路基因数数据
        gene_counts_data = analyze_pathway_sizes(pathways)
        
        # 2. 生成矩阵和重叠数据
        df_mat, pairs = analyze_overlaps(pathways)
        # 3. 统一绘制两张终图 (组合分布图 + 热图)
        plot_distributions_and_heatmap(df_mat, pairs, gene_counts_data)
        
        # 4. 打印 Top 10
        print_top_overlaps(pathways, df_mat)