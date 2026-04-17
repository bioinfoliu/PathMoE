import os
import subprocess
import pandas as pd
import numpy as np
import requests
import gc
import sys

# ================= 配置区 =================
CANCER_LIST = [
"BRCA", "HNSC", "LGG", "THCA", "PRAD", "LUAD", "BLCA", "STAD", "LUSC", "KIRC"
]

BASE_DIR = "/data/zliu/Path_MoE/data"
PROBE_MAP_URL = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/probeMap%2FilluminaMethyl450_hg19_GPL16304_TCGAlegacy"
PANCAN_SURV_URL = "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/Survival_SupplementalTable_S1_20171025_xena_sp"
PANCAN_SURV_FILE = os.path.join(BASE_DIR, "TCGA_PANCAN_Survival_Master.tsv")
TOP_K = 2000  # 提取方差最大的前 K 个特征

def log(msg):
    """日志打印函数"""
    print(f"[*] {msg}", flush=True)

def download_file(url, out_path):
    """通用的安全下载函数"""
    if not os.path.exists(out_path):
        log(f"📥 Downloading: {os.path.basename(out_path)}...")
        cmd = ["wget", "--user-agent='Mozilla/5.0'", "-q", "--show-progress", "-O", out_path, url]
        try:
            subprocess.run(cmd, check=True)
            return True
        except Exception as e:
            log(f"❌ Download failed for {url}: {e}")
            if os.path.exists(out_path):
                os.remove(out_path)
            return False
    return True

# ================= PHASE 1: 特征提取函数 =================

def get_probe_to_gene_map():
    """获取甲基化探针到基因的映射字典"""
    probe_map_path = os.path.join(BASE_DIR, "probeMap_legacy.tsv")
    download_file(PROBE_MAP_URL, probe_map_path)
    
    log("📖 Loading Probe Map...")
    header = pd.read_csv(probe_map_path, sep='\t', nrows=0).columns.tolist()
    
    id_col = header[0] 
    gene_col = next((c for c in header if 'gene' in c.lower()), None)
    if not gene_col:
        gene_col = header[1]

    map_df = pd.read_csv(probe_map_path, sep='\t', usecols=[id_col, gene_col])
    map_df = map_df.dropna(subset=[gene_col])
    
    return dict(zip(map_df[id_col], map_df[gene_col]))

def process_rna(cancer, cancer_dir):
    url = f"https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.{cancer}.sampleMap%2FHiSeqV2.gz"
    raw_file = os.path.join(cancer_dir, f"TCGA-{cancer}.HiSeqV2.gz")
    output_file = os.path.join(cancer_dir, f"TCGA-{cancer}.top2k_var_tpm.csv")
    
    if os.path.exists(output_file):
        log("✅ RNA (Top2k) already processed. Skipping.")
        return

    if download_file(url, raw_file):
        log("🔄 Processing RNA...")
        try:
            df = pd.read_csv(raw_file, sep='\t', index_col=0, compression='gzip')
            if df.index.duplicated().any():
                df = df.groupby(df.index).mean()
            
            variances = df.var(axis=1)
            top_genes = variances.nlargest(TOP_K).index
            final_df = df.loc[top_genes]
            
            final_df.to_csv(output_file)
            log(f"🎉 RNA extracted! Shape: {final_df.shape}")
            del df, variances, final_df
        except Exception as e:
            log(f"❌ Error processing RNA: {e}")

def process_cnv(cancer, cancer_dir):
    url = f"https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.{cancer}.sampleMap%2FGistic2_CopyNumber_Gistic2_all_data_by_genes.gz"
    raw_file = os.path.join(cancer_dir, f"TCGA-{cancer}.Gistic2_all_data.gz")
    output_file = os.path.join(cancer_dir, f"TCGA-{cancer}.top2k_var_cnv.csv")
    
    if os.path.exists(output_file):
        log("✅ CNV (Top2k) already processed. Skipping.")
        return

    if download_file(url, raw_file):
        log("🔄 Processing CNV...")
        try:
            df = pd.read_csv(raw_file, sep='\t', compression='gzip', index_col=0)
            if df.index.duplicated().any():
                df = df.groupby(df.index).mean()
            
            variances = df.var(axis=1)
            top_genes = variances.nlargest(TOP_K).index
            final_df = df.loc[top_genes]
            
            final_df.to_csv(output_file)
            log(f"🎉 CNV extracted! Shape: {final_df.shape}")
            del df, variances, final_df
        except Exception as e:
            log(f"❌ Error processing CNV: {e}")

def process_met(cancer, cancer_dir, probe_to_gene):
    url = f"https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.{cancer}.sampleMap%2FHumanMethylation450.gz"
    raw_file = os.path.join(cancer_dir, f"TCGA-{cancer}.HumanMethylation450.gz")
    output_file = os.path.join(cancer_dir, f"TCGA-{cancer}.top2k_var_met.csv")
    
    if os.path.exists(output_file):
        log("✅ Methylation (Top2k) already processed. Skipping.")
        return

    if download_file(url, raw_file):
        log("🔄 Processing Methylation (This might take a while)...")
        try:
            df = pd.read_csv(raw_file, sep='\t', compression='gzip', index_col=0)
            df.index = df.index.map(probe_to_gene)
            df = df[df.index.notna()]
            
            gene_df = df.groupby(level=0).mean()
            variances = gene_df.var(axis=1)
            top_genes = variances.nlargest(TOP_K).index
            final_df = gene_df.loc[top_genes]
            
            final_df.to_csv(output_file)
            log(f"🎉 Methylation extracted! Shape: {final_df.shape}")
            del df, gene_df, variances, final_df
        except Exception as e:
            log(f"❌ Error processing Methylation: {e}")


# ================= PHASE 2: 对齐与缺失值清洗函数 =================

def clean_sample_id(sample_id):
    """统一截取前 15 位 Sample ID，保留 -01 (Tumor) 等信息"""
    if isinstance(sample_id, str) and len(sample_id) >= 15:
        return sample_id[:15]
    return str(sample_id)

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

def load_and_clean_omics(filepath, name):
    """读取组学文件，转置(样本为行)，清理ID，处理重复，并填补NaN"""
    if not os.path.exists(filepath):
        return None
    
    df = pd.read_csv(filepath, index_col=0)
    
    # 之前提取的中间文件是 (Genes x Samples)，现在转置为 (Samples x Genes)
    df = df.T
    
    # 清洗样本 ID
    df.index = [clean_sample_id(c) for c in df.index]
    
    # 处理同一个病人有多个样本的情况 (取均值)
    if df.index.duplicated().any():
        df = df.groupby(level=0).mean()
        
    # 【缺失值处理核心】检查并补 0
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        log(f"      [NaN Check] {name} has {nan_count} NaNs. Filling with 0.0...")
        df = df.fillna(0.0)
    else:
        log(f"      [NaN Check] {name} is clean (0 NaNs).")
        
    return df

def align_and_save_cancer(cancer, cancer_dir, master_surv_df):
    """对齐多组学与生存数据，并输出最终的 Filtered 矩阵"""
    output_dir = os.path.join(cancer_dir, "filtered_data")
    os.makedirs(output_dir, exist_ok=True)
    
    log(f"   📖 Loading and cleaning extracted Top2k Omics...")
    df_rna = load_and_clean_omics(os.path.join(cancer_dir, f"TCGA-{cancer}.top2k_var_tpm.csv"), "RNA")
    df_cnv = load_and_clean_omics(os.path.join(cancer_dir, f"TCGA-{cancer}.top2k_var_cnv.csv"), "CNV")
    df_met = load_and_clean_omics(os.path.join(cancer_dir, f"TCGA-{cancer}.top2k_var_met.csv"), "MET")

    if df_rna is None or df_cnv is None or df_met is None:
        log("   ❌ Missing intermediate csv files. Skipping alignment.")
        return

    # 匹配生存数据
    surv_samples = set(df_rna.index).intersection(master_surv_df.index)
    df_surv = master_surv_df.loc[list(surv_samples), ['OS', 'OS.time']].copy()
    df_surv.columns = ['OS', 'OS_time']
    
    # 清洗异常的生存时间 (<= 0)
    df_surv = df_surv.dropna(subset=['OS', 'OS_time'])
    df_surv = df_surv[df_surv['OS_time'] > 0]
    valid_surv_samples = set(df_surv.index)

    # 计算最终完备的大交集
    common_samples = (
        set(df_rna.index) & set(df_cnv.index) & set(df_met.index) & valid_surv_samples
    )
    common_samples = sorted(list(common_samples))
    
    log(f"   🎯 Final Aligned Samples: {len(common_samples)} patients")
    
    if len(common_samples) < 50:
        log("   ❌ Too few aligned samples! Skipping save.")
        return

    # 裁剪并保存最终文件
    df_rna.loc[common_samples].to_csv(os.path.join(output_dir, f"TCGA-{cancer}.top2k_var_tpm_filtered.csv"))
    df_cnv.loc[common_samples].to_csv(os.path.join(output_dir, f"TCGA-{cancer}.top2k_var_cnv_filtered.csv"))
    df_met.loc[common_samples].to_csv(os.path.join(output_dir, f"TCGA-{cancer}.top2k_var_met_filtered.csv"))
    df_surv.loc[common_samples].to_csv(os.path.join(output_dir, f"TCGA-{cancer}.top2k_var_survival_filtered.csv"))
    
    log("   ✅ Saved final aligned & cleaned matrices to filtered_data/ !")


# ================= 主流程 =================

def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    
    log("🚀======================================================🚀")
    log(f"   PHASE 1: High Variance (Top {TOP_K}) Extraction")
    log("🚀======================================================🚀\n")
    probe_to_gene = get_probe_to_gene_map()
    
    for i, cancer in enumerate(CANCER_LIST):
        log(f"\n--- [{i+1}/{len(CANCER_LIST)}] Extracting features for {cancer} ---")
        cancer_dir = os.path.join(BASE_DIR, cancer)
        os.makedirs(cancer_dir, exist_ok=True)
        
        process_rna(cancer, cancer_dir)
        process_cnv(cancer, cancer_dir)
        process_met(cancer, cancer_dir, probe_to_gene)
        gc.collect()

    log("\n🚀======================================================🚀")
    log("   PHASE 2: Data Alignment & NaN Cleanup")
    log("🚀======================================================🚀\n")
    master_surv_df = get_pancan_survival()
    
    for i, cancer in enumerate(CANCER_LIST):
        log(f"\n--- [{i+1}/{len(CANCER_LIST)}] Aligning {cancer} ---")
        cancer_dir = os.path.join(BASE_DIR, cancer)
        try:
            align_and_save_cancer(cancer, cancer_dir, master_surv_df)
        except Exception as e:
            log(f"❌ Alignment error for {cancer}: {e}")

    log("\n✨ Pipeline Complete! Your Top2K variance datasets are ready for training.")

if __name__ == "__main__":
    main()