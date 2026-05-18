import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class SurvivalDataset(Dataset):
    def __init__(self, cancer: str, base_dir: str):
        """
        专门为纯组学模式设计的加载器 (原始数据版)
        - 自动处理 NaN
        - 自动计算生存时间中位数并生成 0/1 标签
        - 不包含全局标准化，保留原始数据特征
        """
        self.cancer = cancer
        # 路径指向你的 filtered_data 目录
        self.data_dir = os.path.join(base_dir, cancer, "filtered_data")
        
        # 1. 加载组学数据 
        try:
            rna_df = pd.read_csv(os.path.join(self.data_dir, f"TCGA-{cancer}.hallmark_tpm_filtered.csv"), index_col=0)
            cnv_df = pd.read_csv(os.path.join(self.data_dir, f"TCGA-{cancer}.hallmark_cnv_filtered.csv"), index_col=0)
            met_df = pd.read_csv(os.path.join(self.data_dir, f"TCGA-{cancer}.hallmark_met_filtered.csv"), index_col=0)
            surv_df = pd.read_csv(os.path.join(self.data_dir, f"TCGA-{cancer}.survival_filtered.csv"), index_col=0)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"❌ 找不到 {cancer} 的对齐文件，请检查 filtered_data 目录。") from e

        # 2. 确保样本顺序完全一致
        common_samples = surv_df.index
        self.rna = rna_df.loc[common_samples]
        self.cnv = cnv_df.loc[common_samples]
        self.met = met_df.loc[common_samples]

        # 3. 核心标签逻辑：Median Split
        # 这里的标签 1 代表“高风险/短生存期”，0 代表“低风险/长生存期”
        time_col = 'OS_time'
        median_time = surv_df[time_col].median()
        self.labels = (surv_df[time_col] < median_time).astype(np.int64)
        
        # 4. 转换为 Tensor (为了兼容 PyTorch 模型)
        # 这里直接使用原始数据的值，不进行任何缩放
        self.x_rna = torch.tensor(self.rna.values, dtype=torch.float32)
        self.x_cnv = torch.tensor(self.cnv.values, dtype=torch.float32)
        self.x_met = torch.tensor(self.met.values, dtype=torch.float32)
        self.y = torch.tensor(self.labels.values, dtype=torch.float32)
        
        self.sample_ids = common_samples.tolist()
        self.feature_names = self.rna.columns.tolist() # Hallmark Pathways 名字
        self.gene_list = self.rna.columns.tolist()    

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "x_rna": self.x_rna[idx],
            "x_cnv": self.x_cnv[idx],
            "x_met": self.x_met[idx],
            "y": self.y[idx],
            "id": self.sample_ids[idx]
        }

    def get_labels(self):
        """供 sklearn Baseline 模型直接调用的方法"""
        return self.labels.values