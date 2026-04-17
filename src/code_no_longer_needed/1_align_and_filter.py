import os
import pandas as pd
import requests

# ================= 配置区 =================
CANCER_LIST = [
    "BLCA", "BRCA",  "HNSC", "KIRC", "LGG",
    "LUAD", "LUSC", "PRAD", "STAD", "THCA", 
]
BASE_DIR = "/data/zliu/Path_MoE/data"

# 最权威的 TCGA 2017 泛癌生存时间金标准表 (用于提取 Label)
PANCAN_SURV_URL = "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/Survival_SupplementalTable_S1_20171025_xena_sp"
PANCAN_SURV_FILE = os.path.join(BASE_DIR, "TCGA_PANCAN_Survival_Master.tsv")

def log(msg):
    print(f"[*] {msg}", flush=True)

def clean_sample_id(sample_id):
    """
    统一截取前 15 位 (Sample ID)，保留 -01 (Tumor) 等信息。
    这是实现组学与生存数据对齐的关键。
    """
    if isinstance(sample_id, str) and len(sample_id) >= 15:
        return sample_id[:15]
    return str(sample_id)

def load_and_fix_omics(filepath):
    """读取组学矩阵，自动处理转置 (确保为 Samples x Genes)，并清洗样本 ID"""
    if not os.path.exists(filepath):
        return None
        
    df = pd.read_csv(filepath, index_col=0)
    
    # 自动修复维度：如果行数(基因)远大于列数(样本)，执行转置
    if df.shape[0] > df.shape[1] and df.shape[0] > 2000:
        df = df.T
        
    # 归一化 Index 并处理重复样本（取均值）
    df.index = [clean_sample_id(c) for c in df.index]
    if df.index.duplicated().any():
        df = df.groupby(level=0).mean()
        
    return df

def get_pancan_survival():
    """下载并加载 PanCan Atlas 官方生存主表"""
    if not os.path.exists(PANCAN_SURV_FILE):
        log("⬇️ Downloading PanCan Survival Master File...")
        response = requests.get(PANCAN_SURV_URL, stream=True)
        response.raise_for_status()
        with open(PANCAN_SURV_FILE, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
    df = pd.read_csv(PANCAN_SURV_FILE, sep='\t')
    df['sample'] = df['sample'].apply(clean_sample_id)
    df = df.set_index('sample')
    df = df[~df.index.duplicated(keep='first')]
    return df

def process_cancer(cancer, master_surv_df):
    cancer_dir = os.path.join(BASE_DIR, cancer)
    output_dir = os.path.join(cancer_dir, "filtered_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载三组学数据 (纯组学模式，不再下载 clinicalMatrix)
    log(f"   📖 Loading Omics: RNA, CNV, MET...")
    df_rna = load_and_fix_omics(os.path.join(cancer_dir, f"TCGA-{cancer}.hallmark_tpm.csv"))
    df_cnv = load_and_fix_omics(os.path.join(cancer_dir, f"TCGA-{cancer}.hallmark_cnv.csv"))
    df_met = load_and_fix_omics(os.path.join(cancer_dir, f"TCGA-{cancer}.hallmark_met.csv"))

    if df_rna is None or df_cnv is None or df_met is None:
        log("   ❌ Missing hallmark csv files. Please run extraction script first.")
        return

    # 2. 匹配生存数据 (获取训练 Label)
    # 取组学样本与生存总表的交集
    surv_samples = set(df_rna.index).intersection(master_surv_df.index)
    df_surv = master_surv_df.loc[list(surv_samples), ['OS', 'OS.time']].copy()
    df_surv.columns = ['OS', 'OS_time']
    
    # 清洗：移除空值和异常的生存时间 (<= 0)
    df_surv = df_surv.dropna(subset=['OS', 'OS_time'])
    df_surv = df_surv[df_surv['OS_time'] > 0]
    valid_surv_samples = set(df_surv.index)

    # 3. 计算最终样本大交集
    common_samples = (
        set(df_rna.index) & set(df_cnv.index) & set(df_met.index) & valid_surv_samples
    )
    common_samples = sorted(list(common_samples))
    
    log(f"   🎯 Final Aligned Samples: {len(common_samples)} patients")
    
    # 过滤掉样本量过少的癌种（纯组学模型建议保留 50 例以上）
    if len(common_samples) < 200:
        log("   ❌ Too few samples! Skipping.")
        return

    # 4. 裁剪并保存
    df_rna.loc[common_samples].to_csv(os.path.join(output_dir, f"TCGA-{cancer}.hallmark_tpm_filtered.csv"))
    df_cnv.loc[common_samples].to_csv(os.path.join(output_dir, f"TCGA-{cancer}.hallmark_cnv_filtered.csv"))
    df_met.loc[common_samples].to_csv(os.path.join(output_dir, f"TCGA-{cancer}.hallmark_met_filtered.csv"))
    df_surv.loc[common_samples].to_csv(os.path.join(output_dir, f"TCGA-{cancer}.survival_filtered.csv"))
    
    # 【兼容性处理】保存一个空的临床文件，防止 DataLoader 报错找不到文件
    pd.DataFrame(index=common_samples).to_csv(os.path.join(output_dir, f"TCGA-{cancer}.clinical_filtered.csv"))
    
    log("   ✅ Success: Files saved to filtered_data/ (Pure-Omics Mode)")

def main():
    log("🚀 Starting Pure-Omics Data Alignment (No Clinical Dependency)")
    master_surv_df = get_pancan_survival()
    
    for cancer in CANCER_LIST:
        log(f"\n{'-'*40}\n   Processing: {cancer}\n{'-'*40}")
        try:
            process_cancer(cancer, master_surv_df)
        except Exception as e:
            log(f"❌ Error in {cancer}: {e}")

if __name__ == "__main__":
    main()