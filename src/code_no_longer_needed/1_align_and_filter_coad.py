import pandas as pd
import os
import argparse

# ==========================================
# 🌟 终极破局神器：患者级 (Patient-level) 强制对齐
# ==========================================
def clean_to_patient_id(barcode: str) -> str:
    """
    统一格式化为 12 位的患者 ID (Patient Barcode)：
    1. 替换所有的点 '.' 为横杠 '-'
    2. 严格截取前 12 位字符 (例如：'TCGA-A6-2670')
    无视后面是 -01 (肿瘤), -11 (癌旁), 还是 A/B。
    """
    barcode = str(barcode).strip()
    barcode = barcode.replace('.', '-')
    # 保证不越界的情况下，取前12个字符
    return barcode[:12] if len(barcode) >= 12 else barcode

def align_and_filter_coad(data_dir: str):
    print(f"🚀 开始处理 COAD 数据集，目录: {data_dir}")
    
    # 1. 配置文件路径
    rna_file = os.path.join(data_dir, "TCGA-COAD.hallmark_tpm.csv")
    cnv_file = os.path.join(data_dir, "TCGA-COAD.hallmark_cnv.csv")
    met_file = os.path.join(data_dir, "TCGA-COAD.hallmark_met.csv")
    label_file = os.path.join(data_dir, "COAD_Labels.csv")
    
    out_dir = os.path.join(data_dir, "filtered_subtype_data")
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. 读取数据
    print("📥 正在读取原始组学数据和标签文件...")
    try:
        df_rna = pd.read_csv(rna_file, index_col=0)
        df_cnv = pd.read_csv(cnv_file, index_col=0)
        df_met = pd.read_csv(met_file, index_col=0)
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到组学文件。\n{e}")
        return

    try:
        df_labels = pd.read_csv(label_file)
    except FileNotFoundError:
         print(f"❌ 错误: 找不到标签文件 {label_file}。")
         return
         
    # 3. 处理临床标签
    print("🔍 正在处理 COAD 临床标签...")
    id_col = "pan.samplesID"
    subtype_col = "Subtype_Selected"
    
    if id_col not in df_labels.columns or subtype_col not in df_labels.columns:
         print(f"❌ 错误: 找不到列名 '{id_col}' 或 '{subtype_col}'。")
         return
         
    df_labels_clean = df_labels[[id_col, subtype_col]].dropna().copy()
    
    # 🌟 将临床标签的 ID 强制洗成 12 位的患者 ID
    df_labels_clean[id_col] = df_labels_clean[id_col].apply(clean_to_patient_id)
    df_labels_clean = df_labels_clean.drop_duplicates(subset=[id_col], keep='first')
    df_labels_clean.set_index(id_col, inplace=True)
    
    unique_subtypes = df_labels_clean[subtype_col].unique()
    print(f"   ➤ 发现 {len(unique_subtypes)} 个亚型类别: {unique_subtypes}")
    subtype_mapping = {subtype: idx for idx, subtype in enumerate(sorted(unique_subtypes))}
    print(f"   ➤ 映射关系: {subtype_mapping}")
    df_labels_clean['label'] = df_labels_clean[subtype_col].map(subtype_mapping)
    
    # 4. 🌟 强制将所有组学数据的列名洗成 12 位的患者 ID！
    print("🔄 正在执行患者级 (Patient-Level) 样本对齐...")
    
    # 重命名列
    df_rna.columns = [clean_to_patient_id(c) for c in df_rna.columns]
    df_cnv.columns = [clean_to_patient_id(c) for c in df_cnv.columns]
    df_met.columns = [clean_to_patient_id(c) for c in df_met.columns]
    
    # 极少数患者会有多个肿瘤样本，我们只保留 DataFrame 中出现的第一个样本
    df_rna = df_rna.loc[:, ~df_rna.columns.duplicated(keep='first')]
    df_cnv = df_cnv.loc[:, ~df_cnv.columns.duplicated(keep='first')]
    df_met = df_met.loc[:, ~df_met.columns.duplicated(keep='first')]

    samples_rna = set(df_rna.columns)
    samples_cnv = set(df_cnv.columns)
    samples_met = set(df_met.columns)
    samples_label = set(df_labels_clean.index)
    
    common_samples = samples_rna & samples_cnv & samples_met & samples_label
    common_samples = sorted(list(common_samples))
    
    print(f"   ➤ 清洗后(患者级)数量 -> RNA: {len(samples_rna)}, CNV: {len(samples_cnv)}, MET: {len(samples_met)}, 标签: {len(samples_label)}")
    print(f"   ➤ 🎉 共同患者数 (交集): {len(common_samples)}")
    
    if len(common_samples) == 0:
        print("❌ 错误: 依然没有共同样本！(打印前3个排查)")
        print(f"RNA: {list(samples_rna)[:3]}")
        print(f"Label: {list(samples_label)[:3]}")
        return

    # 5. 过滤并保存数据
    print("💾 正在保存过滤后的对齐数据...")
    df_rna_filtered = df_rna[common_samples]
    df_cnv_filtered = df_cnv[common_samples]
    df_met_filtered = df_met[common_samples]
    df_y_filtered = df_labels_clean.loc[common_samples, ['label']]
    
    df_rna_filtered.to_csv(os.path.join(out_dir, "TCGA-COAD.hallmark_tpm_filtered.csv"))
    df_cnv_filtered.to_csv(os.path.join(out_dir, "TCGA-COAD.hallmark_cnv_filtered.csv"))
    df_met_filtered.to_csv(os.path.join(out_dir, "TCGA-COAD.hallmark_met_filtered.csv"))
    df_y_filtered.to_csv(os.path.join(out_dir, "TCGA-COAD.y_filtered.csv"))
    
    print("✅ 处理完成！你可以去跑 COAD 的模型了！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/zliu/Path_MoE/data/COAD")
    args = parser.parse_args()
    align_and_filter_coad(args.data_dir)