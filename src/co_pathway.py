import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

# ================= Nature 级别全局图表设置 =================
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 8
# ==========================================================

# ================= 参数设置 =================
base_file_path = '/data/zliu/gating_pm50/BRCA_s{}_gating.csv'
num_seeds = 20
min_edge_weight = 0.016 # 共激活频率阈值
# ============================================

pathway_names = None
total_co_freq = None
total_node_freq = None
valid_seeds = 0

print(f"Starting Nature-style ensemble analysis for {num_seeds} seeds...")

for seed in range(num_seeds):
    file_path = base_file_path.format(seed)
    if not os.path.exists(file_path):
        continue
        
    df = pd.read_csv(file_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    pathway_data = df.iloc[:, 1:] if isinstance(df.iloc[0, 0], str) else df
    
    if pathway_names is None:
        pathway_names = pathway_data.columns.tolist()
        num_pathways = len(pathway_names)
        total_co_freq = np.zeros((num_pathways, num_pathways))
        total_node_freq = np.zeros(num_pathways)
        
    # 向量化提取 Top-k
    binary_matrix = (pathway_data.values > 0).astype(float)
        
    co_occurrence = np.dot(binary_matrix.T, binary_matrix)
    total_co_freq += (co_occurrence / len(df))
    total_node_freq += binary_matrix.mean(axis=0)
    valid_seeds += 1

if valid_seeds == 0:
    raise ValueError("No valid CSV files found!")

avg_co_freq = total_co_freq / valid_seeds
avg_node_freq = total_node_freq / valid_seeds
co_freq_df = pd.DataFrame(avg_co_freq, index=pathway_names, columns=pathway_names)

# ================= 构建网络图 =================
G = nx.Graph()
for i in range(len(pathway_names)):
    for j in range(i + 1, len(pathway_names)):
        weight = co_freq_df.iloc[i, j]
        if weight > min_edge_weight:
            name_u = pathway_names[i].replace("HALLMARK_", "")
            name_v = pathway_names[j].replace("HALLMARK_", "")
            G.add_edge(name_u, name_v, weight=weight)

# ================= Nature 风格可视化 =================
fig, ax = plt.subplots(figsize=(7.2, 6), dpi=300)
pos = nx.spring_layout(G, k=1.2, iterations=100, seed=42)

node_color = "#4DBBD5" 
node_edge_color = "#3C5488"
edge_color = "#CCCCCC"

# 节点大小动态计算
node_sizes = []
for node in G.nodes():
    original_name = f"HALLMARK_{node}" if f"HALLMARK_{node}" in pathway_names else node
    idx = pathway_names.index(original_name)
    node_sizes.append(avg_node_freq[idx] * 2000) 

# ================= 核心修改：动态映射连线粗细 =================
edges = G.edges(data=True)
weights_list = [d['weight'] for u, v, d in edges]

if weights_list:
    max_weight = max(weights_list)
    # 按照比例映射：让最大共激活频率的线宽达到 4.5 磅 (非常醒目)，其他的按比例缩小
    # 加上 0.2 作为基础线宽，防止最小的频率线宽变成 0 看不见
    edge_widths = [(w / max_weight) * 4.5 + 0.2 for w in weights_list]
else:
    edge_widths = []
# ==========================================================

nx.draw_networkx_nodes(G, pos, ax=ax, 
                       node_size=node_sizes, 
                       node_color=node_color, 
                       edgecolors=node_edge_color, 
                       linewidths=1.2, 
                       alpha=0.9)

# 使用新的 edge_widths
nx.draw_networkx_edges(G, pos, ax=ax, 
                       width=edge_widths, 
                       edge_color=edge_color, 
                       alpha=0.7)

nx.draw_networkx_labels(G, pos, ax=ax, 
                        font_size=7, 
                        font_family='Arial', 
                        font_color='#333333', 
                        font_weight='bold')

ax.axis('off')
plt.title("BRCA Pathway Co-activation Synergy", fontsize=10, fontweight='bold', fontfamily='Arial', pad=10)
plt.tight_layout()

output_pdf = 'BRCA_Nature_Synergy.pdf'
plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight', transparent=True)
print(f"🎉 Success! Nature-style publication figure saved as {output_pdf}")