import torch
import numpy as np

def create_pathway_mask(gmt_path, gene_list):
    """
    生成 [Genes, 50] 的掩码矩阵。
    如果基因 i 属于通路 j，则 mask[i, j] = 1，否则为 0。
    """
    # 1. 读取 GMT 文件
    pathway_dict = {}
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            p_name = parts[0]
            p_genes = set(parts[2:]) # 跳过描述列
            pathway_dict[p_name] = p_genes
            
    # 排序通路名称，保证每次运行顺序一致
    pathway_names = sorted(list(pathway_dict.keys()))
    num_pathways = len(pathway_names)
    num_genes = len(gene_list)
    
    print(f"🧬 [Prior Knowledge] Loaded {num_pathways} Hallmark Pathways.")
    
    # 2. 构建 Mask
    mask = torch.zeros((num_genes, num_pathways), dtype=torch.float32)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    
    match_count = 0
    for j, p_name in enumerate(pathway_names):
        p_genes = pathway_dict[p_name]
        for g in p_genes:
            if g in gene_to_idx:
                mask[gene_to_idx[g], j] = 1.0
                match_count += 1
                
    print(f"   ✅ Mask generated. Total gene-pathway connections: {match_count}")
    return mask, pathway_names