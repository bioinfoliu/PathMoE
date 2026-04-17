import os
import subprocess
import pandas as pd
import gc
import sys


# "BLCA", "BRCA", "COAD","HNSC", "KIRC", "LGG", "LUAD", "LUSC", "PRAD", "STAD", "THCA", "UCEC"
CANCER_LIST = ["COAD"]

BASE_DIR = "/data/zliu/Path_MoE/data"
GMT_FILE = os.path.join(BASE_DIR, "h.all.v2023.1.Hs.symbols.gmt")
PROBE_MAP_URL = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/probeMap%2FilluminaMethyl450_hg19_GPL16304_TCGAlegacy"

def log(msg):
    print(f"[*] {msg}", flush=True)

def download_file(url, out_path):
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

def load_hallmark_genes(filepath):
    unique_genes = set()
    if not os.path.exists(filepath):
        log(f"❌ Error: Source file not found: {filepath}")
        sys.exit(1)

    log(f"📖 Reading gene definitions from: {os.path.basename(filepath)}")
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            genes = parts[2:]
            unique_genes.update(genes)

    sorted_genes = sorted(list(unique_genes))
    log(f"   🎯 Total unique Hallmark genes: {len(sorted_genes)}")
    return sorted_genes

def get_probe_to_gene_map():
    probe_map_path = os.path.join(BASE_DIR, "probeMap_legacy.tsv")
    download_file(PROBE_MAP_URL, probe_map_path)
    
    log("📖 Loading Probe Map...")
    header = pd.read_csv(probe_map_path, sep='\t', nrows=0).columns.tolist()
    
    id_col = header[0] 
    gene_col = next((c for c in header if 'gene' in c.lower()), None)
    if not gene_col:
        log("⚠️ Warning: 'gene' column not found directly, defaulting to the second column.")
        gene_col = header[1]

    map_df = pd.read_csv(probe_map_path, sep='\t', usecols=[id_col, gene_col])
    map_df = map_df.dropna(subset=[gene_col])
    
    return dict(zip(map_df[id_col], map_df[gene_col]))

def process_rna(cancer, cancer_dir, target_genes):
    """处理转录组 RNA-Seq (HiSeqV2) - 强制基因维度对齐"""
    url = f"https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.{cancer}.sampleMap%2FHiSeqV2.gz"
    raw_file = os.path.join(cancer_dir, f"TCGA-{cancer}.HiSeqV2.gz")
    output_file = os.path.join(cancer_dir, f"TCGA-{cancer}.hallmark_tpm.csv")
    
    if os.path.exists(output_file):
        log("✅ RNA (TPM) already processed. Skipping.")
        return

    if download_file(url, raw_file):
        log("🔄 Processing RNA...")
        try:
            df = pd.read_csv(raw_file, sep='\t', index_col=0, compression='gzip')
            # 【修复 Bug】先处理可能存在的重复基因名，防止 reindex 报错
            if df.index.duplicated().any():
                df = df.groupby(df.index).mean()
            # 【核心对齐】使用 reindex 保留白名单基因，缺失的填充为 0.0
            aligned_df = df.reindex(target_genes, fill_value=0.0)
            aligned_df.to_csv(output_file)
            log(f"🎉 RNA processed! Shape perfectly aligned: {aligned_df.shape}")
            del df, aligned_df
        except Exception as e:
            log(f"❌ Error processing RNA: {e}")

def process_cnv(cancer, cancer_dir, target_genes):
    """处理拷贝数变异 (CNV) - 强制基因维度对齐"""
    url = f"https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.{cancer}.sampleMap%2FGistic2_CopyNumber_Gistic2_all_data_by_genes.gz"
    raw_file = os.path.join(cancer_dir, f"TCGA-{cancer}.Gistic2_all_data.gz")
    output_file = os.path.join(cancer_dir, f"TCGA-{cancer}.hallmark_cnv.csv")
    
    if os.path.exists(output_file):
        log("✅ CNV already processed. Skipping.")
        return

    if download_file(url, raw_file):
        log("🔄 Processing CNV...")
        try:
            df = pd.read_csv(raw_file, sep='\t', compression='gzip', index_col=0)
            # 【修复 Bug】处理重复基因名
            if df.index.duplicated().any():
                df = df.groupby(df.index).mean()
            # 【核心对齐】强制对齐到白名单基因，缺失补 0.0
            final_df = df.reindex(target_genes, fill_value=0.0)
            final_df.to_csv(output_file)
            log(f"🎉 CNV processed! Shape perfectly aligned: {final_df.shape}")
            del df, final_df
        except Exception as e:
            log(f"❌ Error processing CNV: {e}")

def process_met(cancer, cancer_dir, target_genes, probe_to_gene):
    """处理 DNA 甲基化 (Methylation450k) - 强制基因维度对齐"""
    url = f"https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.{cancer}.sampleMap%2FHumanMethylation450.gz"
    raw_file = os.path.join(cancer_dir, f"TCGA-{cancer}.HumanMethylation450.gz")
    output_file = os.path.join(cancer_dir, f"TCGA-{cancer}.hallmark_met.csv")
    
    if os.path.exists(output_file):
        log("✅ Methylation already processed. Skipping.")
        return

    if download_file(url, raw_file):
        log("🔄 Processing Methylation (This might take a while)...")
        try:
            df = pd.read_csv(raw_file, sep='\t', compression='gzip', index_col=0)
            
            # 探针映射到基因
            df.index = df.index.map(probe_to_gene)
            # 丢弃没有映射到基因的探针
            df = df[df.index.notna()]
            
            # 先按基因名合并探针取均值
            gene_df = df.groupby(level=0).mean()
            # 【核心对齐】强制对齐到白名单基因，缺失补 0.0
            final_df = gene_df.reindex(target_genes, fill_value=0.0)
            
            final_df.to_csv(output_file)
            log(f"🎉 Methylation processed! Shape perfectly aligned: {final_df.shape}")
            del df, gene_df, final_df
        except Exception as e:
            log(f"❌ Error processing Methylation: {e}")

def main():
    log("🚀======================================================🚀")
    log("   Starting Pan-Cancer Multi-Omics Pipeline (N > 400)")
    log("🚀======================================================🚀\n")
    
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # 1. 准备全局变量 (Hallmark 基因集 & 甲基化探针映射)
    target_genes = load_hallmark_genes(GMT_FILE)
    probe_to_gene = get_probe_to_gene_map()
    
    # 2. 遍历大样本癌种
    for i, cancer in enumerate(CANCER_LIST):
        log(f"\n==========================================")
        log(f"   [{i+1}/{len(CANCER_LIST)}] Processing {cancer}")
        log(f"==========================================")
        
        cancer_dir = os.path.join(BASE_DIR, cancer)
        os.makedirs(cancer_dir, exist_ok=True)
        
        # 依次处理三种组学
        process_rna(cancer, cancer_dir, target_genes)
        process_cnv(cancer, cancer_dir, target_genes)
        process_met(cancer, cancer_dir, target_genes, probe_to_gene)
        
        # 显式回收内存，防止内存泄漏导致服务器卡死
        gc.collect()

if __name__ == "__main__":
    main()