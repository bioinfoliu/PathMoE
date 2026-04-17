import os
import pandas as pd

# ================= 配置区 =================
SUBTYPE_COL_MAP = {
    "BRCA": "PAM50Call_RNAseq",
    # "BLCA": "RNA_cluster", # 示例：膀胱癌可能的亚型列名
    # "LUAD": "expression_subtype", # 示例：肺腺癌可能的亚型列名
}

CANCER_LIST = list(SUBTYPE_COL_MAP.keys()) # 目前只跑我们在字典里定义好标签的癌种
BASE_DIR = "/data/zliu/Path_MoE/data"

def log(msg):
    print(f"[*] {msg}", flush=True)

def clean_sample_id(sample_id):
    """
    统一截取前 15 位 (Sample ID)，保留 -01 (Tumor) 等信息。
    """
    if isinstance(sample_id, str) and len(sample_id) >= 15:
        return sample_id[:15]
    return str(sample_id)

def load_and_fix_omics(filepath):
    """读取组学矩阵，自动处理转置，并清洗样本 ID"""
    if not os.path.exists(filepath):
        return None
        
    df = pd.read_csv(filepath, index_col=0)
    
    if df.shape[0] > df.shape[1] and df.shape[0] > 2000:
        df = df.T
        
    df.index = [clean_sample_id(c) for c in df.index]
    if df.index.duplicated().any():
        df = df.groupby(level=0).mean()
        
    return df

def process_subtype_cancer(cancer, target_col):
    cancer_dir = os.path.join(BASE_DIR, cancer)
    output_dir = os.path.join(cancer_dir, "filtered_subtype_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载临床数据并提取亚型标签
    log(f"   📋 Extracting labels from {target_col}...")
    clinical_file = os.path.join(cancer_dir, f"TCGA-{cancer}.clinicalMatrix.tsv")
    if not os.path.exists(clinical_file):
        log(f"   ❌ Missing clinical file for {cancer}.")
        return

    df_clinical = pd.read_csv(clinical_file, sep='\t')
    
    if target_col not in df_clinical.columns:
        log(f"   ❌ Column '{target_col}' not found in {cancer} clinical data.")
        return
        
    # 剔除没有亚型标签的样本
    df_clinical_labeled = df_clinical.dropna(subset=[target_col]).copy()
    df_clinical_labeled['sampleID'] = df_clinical_labeled['sampleID'].apply(clean_sample_id)
    
    # 将 sampleID 设为 index，并提取 target_col 作为 Label
    df_clinical_labeled.set_index('sampleID', inplace=True)
    # 处理可能的重复临床样本记录
    df_clinical_labeled = df_clinical_labeled[~df_clinical_labeled.index.duplicated(keep='first')]
    valid_label_samples = set(df_clinical_labeled.index)

    # 2. 加载三组学数据
    log(f"   📖 Loading Omics: RNA, CNV, MET...")
    df_rna = load_and_fix_omics(os.path.join(cancer_dir, f"TCGA-{cancer}.hallmark_tpm.csv"))
    df_cnv = load_and_fix_omics(os.path.join(cancer_dir, f"TCGA-{cancer}.hallmark_cnv.csv"))
    df_met = load_and_fix_omics(os.path.join(cancer_dir, f"TCGA-{cancer}.hallmark_met.csv"))

    if df_rna is None or df_cnv is None or df_met is None:
        log("   ❌ Missing hallmark csv files.")
        return

    # 3. 计算最终样本大交集 (RNA + CNV + MET + 拥有亚型标签的样本)
    common_samples = (
        set(df_rna.index) & set(df_cnv.index) & set(df_met.index) & valid_label_samples
    )
    common_samples = sorted(list(common_samples))
    
    log(f"   🎯 Final Aligned Samples (Complete Data + Labels): {len(common_samples)} patients")
    
    if len(common_samples) < 50:
        log("   ❌ Too few samples! Skipping.")
        return

    # 4. 裁剪并保存
    df_rna.loc[common_samples].to_csv(os.path.join(output_dir, f"TCGA-{cancer}.hallmark_tpm_filtered.csv"))
    df_cnv.loc[common_samples].to_csv(os.path.join(output_dir, f"TCGA-{cancer}.hallmark_cnv_filtered.csv"))
    df_met.loc[common_samples].to_csv(os.path.join(output_dir, f"TCGA-{cancer}.hallmark_met_filtered.csv"))
    
    # 提取并保存 Label 数据
    df_labels = df_clinical_labeled.loc[common_samples, [target_col]]
    df_labels.columns = ['Subtype'] # 统一改名为 Subtype 方便下游 Dataloader 读取
    df_labels.to_csv(os.path.join(output_dir, f"TCGA-{cancer}.subtype_filtered.csv"))
    
    # 打印一下类别分布
    log(f"   📊 Subtype Distribution:\n{df_labels['Subtype'].value_counts().to_string()}")
    log(f"   ✅ Success: Saved to filtered_subtype_data/")

def main():
    log("🚀 Starting Multi-omics Alignment for Subtype Classification")
    
    for cancer, target_col in SUBTYPE_COL_MAP.items():
        log(f"\n{'-'*40}\n   Processing: {cancer}\n{'-'*40}")
        try:
            process_subtype_cancer(cancer, target_col)
        except Exception as e:
            log(f"❌ Error in {cancer}: {e}")

if __name__ == "__main__":
    main()