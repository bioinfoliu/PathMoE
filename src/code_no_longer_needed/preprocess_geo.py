#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import GEOparse

# 设置路径
GSE_ID = "GSE25066"
GEO_DIR = "./geo_data"
TCGA_FEAT_PATH = "/data/zliu/Path_MoE/data/BRCA/filtered_subtype_data/TCGA-BRCA.hallmark_tpm_filtered.csv"
OUT_DIR = "/data/zliu/Path_MoE/data/GSE25066/processed"

def run_mapping_and_preprocess():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. 重新加载并利用 GEOparse 获取映射表
    print(f"正在从 {GSE_ID} 提取探针映射关系...")
    gse = GEOparse.get_GEO(geo=GSE_ID, destdir=GEO_DIR)
    
    # 获取平台信息 (GPL)
    gpl = list(gse.gpls.values())[0]
    mapping = gpl.table[['ID', 'Gene Symbol']].dropna()
    mapping_dict = dict(zip(mapping['ID'], mapping['Gene Symbol']))
    
    # 2. 读取之前保存的原始数据 (包含探针 ID)
    raw_data_path = os.path.join(GEO_DIR, f"{GSE_ID}_processed_dataset.csv")
    df_raw = pd.read_csv(raw_data_path, index_col=0)
    
    # 分离标签
    labels = df_raw['PAM50_Subtype']
    expr = df_raw.drop(columns=['PAM50_Subtype'])
    
    # 3. 执行探针到基因名的转换
    print("正在将探针 ID 转换为 Gene Symbols...")
    expr_t = expr.T
    expr_t['symbol'] = expr_t.index.map(mapping_dict)
    
    # 丢弃没有对应基因名的探针，并对相同基因名取平均值
    expr_mapped = expr_t.dropna(subset=['symbol']).groupby('symbol').mean().T
    print(f"转换后有效基因数: {expr_mapped.shape[1]}")

    # 4. 读取 TCGA 目标基因列表并对齐
    tcga_genes = pd.read_csv(TCGA_FEAT_PATH, nrows=0).columns[1:].tolist()
    
    # 关键点：对齐并检查非零比例
    expr_aligned = expr_mapped.reindex(columns=tcga_genes, fill_value=0.0)
    
    zero_ratio = (expr_aligned == 0).sum().sum() / expr_aligned.size
    print(f"⚠️ 警告: 对齐后数据中 0 的比例为 {zero_ratio:.2%}")
    if zero_ratio > 0.9:
        print("❌ 错误: 0 的比例过高！说明绝大部分基因名依然没对上。")
    
    # 5. 保存为 npy (适配 PathMoE)
    sample_ids = expr_aligned.index.tolist()
    x_rna = expr_aligned.values.astype(np.float32)
    
    # 填充缺失模态
    zero_fill = np.zeros_like(x_rna)
    
    # 标签处理
    PAM50_TO_INT = {"Basal": 0, "Her2": 1, "LumA": 2, "LumB": 3, "Normal": 4}
    def clean_p(x):
        s = str(x).lower()
        if 'basal' in s: return 0
        if 'her2' in s: return 1
        if 'luma' in s or 'luminal a' in s: return 2
        if 'lumb' in s or 'luminal b' in s: return 3
        if 'normal' in s: return 4
        return -1

    y = labels.apply(clean_p).values
    
    # 过滤掉标签不全的样本
    valid_idx = y != -1
    x_rna, y = x_rna[valid_idx], y[valid_idx]
    zero_fill = zero_fill[valid_idx]
    final_samples = np.array(sample_ids)[valid_idx]

    # 保存
    np.save(os.path.join(OUT_DIR, "mrna.npy"), x_rna)
    np.save(os.path.join(OUT_DIR, "cnv.npy"), zero_fill)
    np.save(os.path.join(OUT_DIR, "met.npy"), zero_fill)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    pd.Series(final_samples).to_csv(os.path.join(OUT_DIR, "sample_ids.csv"), index=False)
    pd.Series(tcga_genes).to_csv(os.path.join(OUT_DIR, "tcga_gene_list.csv"), index=False)
    
    print(f"✅ 处理完成。有效样本: {len(y)}")

if __name__ == "__main__":
    run_mapping_and_preprocess()