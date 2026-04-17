import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

class SubtypeDataset(Dataset):
    def __init__(self, cancer_type: str, base_dir: str):
        super().__init__()
        self.cancer = cancer_type
        
        data_dir = os.path.join(base_dir, cancer_type, "filtered_subtype_data")
        
        df_rna = pd.read_csv(os.path.join(data_dir, f"TCGA-{cancer_type}.hallmark_tpm_filtered.csv"), index_col=0)
        df_cnv = pd.read_csv(os.path.join(data_dir, f"TCGA-{cancer_type}.hallmark_cnv_filtered.csv"), index_col=0)
        df_met = pd.read_csv(os.path.join(data_dir, f"TCGA-{cancer_type}.hallmark_met_filtered.csv"), index_col=0)
        
        # 2. 加载亚型标签
        df_labels = pd.read_csv(os.path.join(data_dir, f"TCGA-{cancer_type}.subtype_filtered.csv"), index_col=0)
        
        # 确保顺序完全一致
        self.sample_ids = df_labels.index.tolist()
        df_rna = df_rna.loc[self.sample_ids]
        df_cnv = df_cnv.loc[self.sample_ids]
        df_met = df_met.loc[self.sample_ids]
        
        # 3. 转换为 Tensor
        self.x_rna = torch.tensor(df_rna.values, dtype=torch.float32)
        self.x_cnv = torch.tensor(df_cnv.values, dtype=torch.float32)
        self.x_met = torch.tensor(df_met.values, dtype=torch.float32)
        
        # 4. 标签编码 (将 'LumA', 'Basal' 等转为 0, 1, 2...)
        self.label_encoder = LabelEncoder()
        labels_str = df_labels['Subtype'].values
        self.y = self.label_encoder.fit_transform(labels_str)
        self.y_tensor = torch.tensor(self.y, dtype=torch.long) # 分类任务必须是 torch.long
        
        self.num_classes = len(self.label_encoder.classes_)
        print(f"[{cancer_type}] 发现 {self.num_classes} 个亚型类别: {self.label_encoder.classes_}")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        return {
            "id": self.sample_ids[idx],
            "x_rna": self.x_rna[idx],
            "x_cnv": self.x_cnv[idx],
            "x_met": self.x_met[idx],
            "y": self.y_tensor[idx]
        }
    
    def get_class_weights(self) -> torch.Tensor:
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.y),
            y=self.y
        )
        return torch.tensor(weights, dtype=torch.float32)