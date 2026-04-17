import pandas as pd
import numpy as np
import os
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

def get_genes_from_gmt(gmt_path, pathway_name):
    """解析 GMT 文件并提取指定通路的基因列表"""
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts[0] == pathway_name:
                # GMT 格式：通路名 \t 描述 \t 基因1 \t 基因2 ...
                return parts[2:]
    return []

def run_pc_on_tcga():
    # 1. 路径设置 (根据你提供的 ls 结果)
    data_dir = "/data/zliu/Path_MoE/data/BRCA/filtered_subtype_data/"
    gmt_path = "/data/zliu/Path_MoE/data/h.all.v2023.1.Hs.symbols.gmt"
    
    tpm_file = os.path.join(data_dir, "TCGA-BRCA.hallmark_tpm_filtered.csv")
    output_dir = "/data/zliu/Path_MoE/data/causal_analysis/TCGA_BRCA/"
    os.makedirs(output_dir, exist_ok=True)

    # 2. 从 GMT 提取基因
    target_pathway = "HALLMARK_CHOLESTEROL_HOMEOSTASIS"
    pathway_genes = get_genes_from_gmt(gmt_path, target_pathway)
    print(f"从 GMT 中提取到 {len(pathway_genes)} 个属于 {target_pathway} 的基因。")

    # 3. 读取 TPM 数据 (RNA-seq)
    print("正在加载 TPM 表达矩阵...")
    # 假设第一列是 Sample ID 或 Gene ID，请根据实际 csv 调整 index_col
    df_tpm = pd.read_csv(tpm_file, index_col=0)
    
    # 4. 提取子集并清洗
    # 找出数据集中存在的基因
    available_genes = [g for g in pathway_genes if g in df_tpm.columns]
    df_sub = df_tpm[available_genes]
    
    # 过滤掉方差为 0 的基因（Fisher-Z 测试要求有变异）
    valid_genes = df_sub.columns[df_sub.var() > 1e-6].tolist()
    df_final = df_sub[valid_genes]
    print(f"最终用于 PC 算法的基因数: {len(valid_genes)} (已排除不存在或无变异的基因)")

    # 5. 运行 PC 算法
    print(f"开始运行 PC 算法 (Alpha=0.05)...")
    data_matrix = df_final.values
    # 使用 Fisher-Z 进行独立性测试
    cg = pc(data_matrix, alpha=0.05, indep_test=fisherz)
    
    # 6. 解析结果并保存为 Edge List (方便导入 Cytoscape)
    adj = cg.G.graph
    edges = []
    nodes = valid_genes
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            # causal-learn 邻接矩阵规则: i->j 为 -1,1; i--j 为 -1,-1
            if adj[i, j] == -1 and adj[j, i] == 1:
                edges.append({'Source': nodes[i], 'Target': nodes[j], 'Type': 'Directed'})
            elif adj[i, j] == 1 and adj[j, i] == -1:
                edges.append({'Source': nodes[j], 'Target': nodes[i], 'Type': 'Directed'})
            elif adj[i, j] == -1 and adj[j, i] == -1:
                edges.append({'Source': nodes[i], 'Target': nodes[j], 'Type': 'Undirected'})

    df_edges = pd.DataFrame(edges)
    output_path = os.path.join(output_dir, "TCGA_Cholesterol_Causal_Edges.csv")
    df_edges.to_csv(output_path, index=False)
    
    print(f"分析完成！共发现 {len(df_edges)} 条边。")
    print(f"结果已保存至: {output_path}")

if __name__ == "__main__":
    run_pc_on_tcga()


    