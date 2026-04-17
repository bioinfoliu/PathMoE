import cptac
import pandas as pd
import numpy as np
import os

def prepare_cptac_for_pathmoe(out_dir="/data/zliu/Path_MoE/data/CPTAC/processed"):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. 下载并加载数据
    print("正在加载 CPTAC BRCA 数据 (首次运行需下载)...")
    br = cptac.Brca()
    
    # 2. 提取各组学
    rna = br.get_transcriptomics() # RNA-Seq
    prot = br.get_proteomics()     # 蛋白质组
    # 临床标签 (包含 PAM50)
    clinical = br.get_clinical()
    
    # 3. 标签映射
    pam50_map = {"Basal": 0, "Her2": 1, "LumA": 2, "LumB": 3, "Normal": 4}
    # CPTAC 中标签列通常叫 'PAM50'
    y_raw = clinical['PAM50']
    
    # 4. 三方对齐 (样本取交集)
    common_samples = rna.index.intersection(prot.index).intersection(y_raw.dropna().index)
    print(f"对齐后的样本数: {len(common_samples)}")
    
    rna_aligned = rna.loc[common_samples]
    prot_aligned = prot.loc[common_samples]
    y_aligned = y_raw.loc[common_samples].map(pam50_map).astype(int)
    
    # 5. 特征对齐 (对齐到你的 TCGA 基因列表)
    tcga_feat_path = "/data/zliu/Path_MoE/data/BRCA/filtered_subtype_data/TCGA-BRCA.hallmark_tpm_filtered.csv"
    tcga_genes = pd.read_csv(tcga_feat_path, nrows=0).columns[1:].tolist()
    
    # 蛋白质组列名处理 (去掉 MultiIndex)
    if isinstance(prot_aligned.columns, pd.MultiIndex):
        prot_aligned.columns = prot_aligned.columns.get_level_values(0)
        
    rna_final = rna_aligned.reindex(columns=tcga_genes, fill_value=0.0)
    prot_final = prot_aligned.reindex(columns=tcga_genes, fill_value=0.0)
    
    # 这里我们用蛋白质组代替缺失的 CNV，或者依然置零，但 RNA 的质量比芯片高得多
    np.save(os.path.join(out_dir, "mrna.npy"), rna_final.values.astype(np.float32))
    np.save(os.path.join(out_dir, "cnv.npy"), prot_final.values.astype(np.float32)) # 暂时借用 CNV 通道跑蛋白质
    np.save(os.path.join(out_dir, "met.npy"), np.zeros_like(rna_final.values))
    np.save(os.path.join(out_dir, "y.npy"), y_aligned.values)
    
    print(f"✅ CPTAC 数据准备就绪！路径: {out_dir}")

if __name__ == "__main__":
    prepare_cptac_for_pathmoe()