#!/usr/bin/env python3
import argparse
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


PAM50_ALLOWED = ["Basal", "Her2", "LumA", "LumB", "Normal"]
PAM50_TO_INT: Dict[str, int] = {
    "Basal": 0,
    "Her2": 1,
    "LumA": 2,
    "LumB": 3,
    "Normal": 4,
}


def _canonicalize_pam50(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    s_low = s.lower()
    if s_low == "her2" or s_low == "her2-enriched" or s_low == "her2_enriched":
        return "Her2"
    if s_low == "basal" or s_low == "basal-like" or s_low == "basal_like":
        return "Basal"
    if s_low == "luma" or s_low == "luminal a" or s_low == "luminal_a":
        return "LumA"
    if s_low == "lumb" or s_low == "luminal b" or s_low == "luminal_b":
        return "LumB"
    if s_low == "normal" or s_low == "normal-like" or s_low == "normal_like":
        return "Normal"
    return s  # keep original for debugging (will be filtered if not allowed)


def read_tcga_gene_list(tcga_csv_path: str) -> List[str]:
    df0 = pd.read_csv(tcga_csv_path, nrows=0)
    cols = list(df0.columns)
    if len(cols) < 2:
        raise ValueError(f"TCGA feature file has too few columns: {tcga_csv_path}")
    genes = [c for c in cols[1:] if isinstance(c, str) and c.strip()]
    if len(genes) == 0:
        raise ValueError(f"Failed to parse gene list from TCGA file header: {tcga_csv_path}")
    return genes


def read_omics_matrix_tsv(
    path: str,
    gene_col: str = "Hugo_Symbol",
    drop_cols: Iterable[str] = ("Entrez_Gene_Id",),
    groupby_gene_mean: bool = True,
) -> pd.DataFrame:
    """
    Return (samples x genes) matrix.
    Input is (genes x samples) with a gene symbol column.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if gene_col not in df.columns:
        raise ValueError(f"Missing `{gene_col}` column in {path}")

    keep = [c for c in df.columns if c not in set(drop_cols)]
    df = df[keep]

    df[gene_col] = df[gene_col].astype(str).str.strip()
    df = df[df[gene_col].notna() & (df[gene_col] != "") & (df[gene_col] != "nan")]

    df = df.set_index(gene_col)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if groupby_gene_mean:
        df = df.groupby(df.index).mean(numeric_only=True)

    df_t = df.T
    df_t.index = df_t.index.astype(str).str.strip()
    df_t.index.name = "sampleID"
    return df_t


def read_metabric_clinical_pam50(
    clinical_path: str,
    sample_id_col: str = "PATIENT_ID",
    pam50_col: str = "CLAUDIN_SUBTYPE",
) -> pd.DataFrame:
    df = pd.read_csv(clinical_path, sep="\t", comment="#", low_memory=False)
    if sample_id_col not in df.columns:
        raise ValueError(f"Missing `{sample_id_col}` column in {clinical_path}")
    if pam50_col not in df.columns:
        raise ValueError(f"Missing `{pam50_col}` column in {clinical_path}")

    df = df[[sample_id_col, pam50_col]].copy()
    df = df.rename(columns={sample_id_col: "sampleID", pam50_col: "PAM50"})
    df["sampleID"] = df["sampleID"].astype(str).str.strip()
    df["PAM50"] = df["PAM50"].apply(_canonicalize_pam50)

    df = df[df["PAM50"].isin(PAM50_ALLOWED)].copy()
    df = df.sort_values("sampleID").reset_index(drop=True)
    return df


def align_features_to_gene_list(mat: pd.DataFrame, genes: List[str]) -> pd.DataFrame:
    # 一次性对齐列，缺失列用 0 填充，避免逐列 insert 的性能问题
    mat2 = mat.replace([np.inf, -np.inf], np.nan)
    mat2 = mat2.reindex(columns=genes, fill_value=0.0)
    mat2 = mat2.fillna(0.0)
    return mat2


def intersect_and_sort_samples(
    mats: List[Tuple[str, pd.DataFrame]],
    clinical: pd.DataFrame,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, List[str]]:
    sample_sets = [set(df.index) for _, df in mats]
    sample_sets.append(set(clinical["sampleID"]))
    common = set.intersection(*sample_sets) if sample_sets else set()
    common_sorted = sorted(common)

    clinical2 = clinical[clinical["sampleID"].isin(common)].copy()
    clinical2 = clinical2.sort_values("sampleID").reset_index(drop=True)

    out = {}
    for name, df in mats:
        out[name] = df.loc[common_sorted].copy()
    return out, clinical2, common_sorted


def encode_pam50(labels: pd.Series) -> np.ndarray:
    y = labels.map(PAM50_TO_INT)
    if y.isna().any():
        bad = labels[y.isna()].unique().tolist()
        raise ValueError(f"Found unmapped PAM50 labels after filtering: {bad}")
    return y.astype(np.int64).to_numpy()


def save_outputs(
    out_dir: str,
    genes: List[str],
    sample_ids: List[str],
    mrna: pd.DataFrame,
    cnv: pd.DataFrame,
    met: pd.DataFrame,
    y: np.ndarray,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    pd.Series(sample_ids, name="sampleID").to_csv(os.path.join(out_dir, "sample_ids.csv"), index=False)
    pd.Series(genes, name="gene").to_csv(os.path.join(out_dir, "tcga_gene_list.csv"), index=False)
    pd.Series(PAM50_ALLOWED, name="PAM50").to_csv(os.path.join(out_dir, "pam50_classes_order.csv"), index=False)

    np.save(os.path.join(out_dir, "mrna.npy"), mrna.to_numpy(dtype=np.float32, copy=False))
    np.save(os.path.join(out_dir, "cnv.npy"), cnv.to_numpy(dtype=np.float32, copy=False))
    np.save(os.path.join(out_dir, "met.npy"), met.to_numpy(dtype=np.float32, copy=False))
    np.save(os.path.join(out_dir, "y.npy"), y)

    mrna.to_csv(os.path.join(out_dir, "mrna_aligned.csv.gz"), index=True, compression="gzip")
    cnv.to_csv(os.path.join(out_dir, "cnv_aligned.csv.gz"), index=True, compression="gzip")
    met.to_csv(os.path.join(out_dir, "met_aligned.csv.gz"), index=True, compression="gzip")
    pd.DataFrame({"sampleID": sample_ids, "y": y}).to_csv(
        os.path.join(out_dir, "labels_pam50_encoded.csv"), index=False
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess METABRIC BRCA multi-omics for external validation.")
    ap.add_argument(
        "--metabric_dir",
        type=str,
        default="/data/zliu/Path_MoE/data/brca_metabric",
        help="METABRIC source dir containing data_*.txt files.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/data/zliu/Path_MoE/data/brca_metabric/processed",
        help="Output directory.",
    )
    ap.add_argument(
        "--tcga_feature_csv",
        type=str,
        default="/data/zliu/Path_MoE/data/BRCA/filtered_subtype_data/TCGA-BRCA.hallmark_tpm_filtered.csv",
        help="TCGA training features (read header only).",
    )
    args = ap.parse_args()

    mrna_path = os.path.join(args.metabric_dir, "data_mrna_illumina_microarray.txt")
    cnv_path = os.path.join(args.metabric_dir, "data_cna.txt")
    met_path = os.path.join(args.metabric_dir, "data_methylation_promoters_rrbs.txt")
    clinical_path = os.path.join(args.metabric_dir, "data_clinical_patient.txt")

    print(f"[1/6] 读取 TCGA 特征列表: {args.tcga_feature_csv}")
    genes = read_tcga_gene_list(args.tcga_feature_csv)
    print(f"TCGA 基因数: {len(genes)}")

    print(f"[2/6] 读取 METABRIC 组学矩阵并映射 Gene Symbol: {mrna_path}")
    mrna = read_omics_matrix_tsv(mrna_path, gene_col="Hugo_Symbol", drop_cols=("Entrez_Gene_Id",), groupby_gene_mean=True)
    print(f"mRNA shape (samples x genes): {mrna.shape}")

    print(f"[2/6] 读取 METABRIC CNV 矩阵: {cnv_path}")
    cnv = read_omics_matrix_tsv(cnv_path, gene_col="Hugo_Symbol", drop_cols=("Entrez_Gene_Id",), groupby_gene_mean=True)
    print(f"CNV shape (samples x genes): {cnv.shape}")

    print(f"[2/6] 读取 METABRIC 甲基化矩阵并按基因聚合(mean): {met_path}")
    met = read_omics_matrix_tsv(met_path, gene_col="Hugo_Symbol", drop_cols=(), groupby_gene_mean=True)
    print(f"MET shape (samples x genes): {met.shape}")

    print(f"[3/6] 读取临床 PAM50 并过滤为 5 类: {clinical_path}")
    clinical = read_metabric_clinical_pam50(clinical_path, sample_id_col="PATIENT_ID", pam50_col="CLAUDIN_SUBTYPE")
    print(f"临床样本数(过滤后): {clinical.shape[0]}")
    print("PAM50 分布:")
    print(clinical["PAM50"].value_counts())

    mats, clinical_aligned, common_samples = intersect_and_sort_samples(
        [("mrna", mrna), ("cnv", cnv), ("met", met)], clinical
    )
    print(f"[4/6] mRNA/CNV/MET/Clinical 交集样本数: {len(common_samples)}")

    mrna2 = mats["mrna"]
    cnv2 = mats["cnv"]
    met2 = mats["met"]

    print("[5/6] 严格按 TCGA 基因列表对齐特征，缺失列/缺失值均补 0")
    mrna3 = align_features_to_gene_list(mrna2, genes)
    cnv3 = align_features_to_gene_list(cnv2, genes)
    met3 = align_features_to_gene_list(met2, genes)

    if not (list(mrna3.index) == list(cnv3.index) == list(met3.index) == list(clinical_aligned["sampleID"])):
        raise RuntimeError("样本顺序未完全对齐，请检查 sampleID 处理逻辑。")

    y = encode_pam50(clinical_aligned["PAM50"])
    print(f"[5/6] y shape: {y.shape}, classes: {PAM50_ALLOWED}")

    # ========================================================
    # 新增统计打印逻辑 / Added statistics print block
    # ========================================================
    print("\n[5.5/6] 📊 对齐后的最终数据统计:")
    print(f"最终 mRNA shape: {mrna3.shape}")
    print(f"最终 CNV shape:  {cnv3.shape}")
    print(f"最终 MET shape:  {met3.shape}")
    print("\n最终 PAM50 样本分布:")
    print(clinical_aligned["PAM50"].value_counts().to_string())
    print("-" * 50 + "\n")
    # ========================================================

    print(f"[6/6] 保存到: {args.out_dir}")
    save_outputs(
        out_dir=args.out_dir,
        genes=genes,
        sample_ids=common_samples,
        mrna=mrna3,
        cnv=cnv3,
        met=met3,
        y=y,
    )
    print("✅ 完成。输出包含 mrna/cnv/met 的 aligned CSV 与 npy，以及 y.npy/labels_csv。")


if __name__ == "__main__":
    main()