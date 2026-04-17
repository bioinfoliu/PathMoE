#!/usr/bin/env python3
import json
import os
import sys
import argparse
import traceback
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 0. 自定义 PyTorch MLP (兼容 sklearn 接口并支持 GPU)
# ==========================================
class TorchMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_sizes=(64, 32), lr=1e-3, epochs=50, batch_size=128):
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.num_classes_ = len(self.classes_)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device) 

        layers = []
        input_dim = X.shape[1]
        for h in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3)) 
            input_dim = h
        layers.append(nn.Linear(input_dim, self.num_classes_)) 
        
        self.model_ = nn.Sequential(*layers).to(self.device)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        
        class_weights = compute_class_weight(class_weight='balanced', classes=self.classes_, y=y)
        weight_tensor = torch.FloatTensor(class_weights).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                out = self.model_(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model_(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# ==========================================
# 1. 数据加载器 (Dataset Definition)
# ==========================================
class SubtypeDataset(Dataset):
    def __init__(self, cancer: str, base_dir: str):
        self.cancer = cancer
        self.data_dir = os.path.join(base_dir, cancer, "filtered_subtype_data")
        
        try:
            rna_df = pd.read_csv(os.path.join(self.data_dir, f"TCGA-{cancer}.hallmark_tpm_filtered.csv"), index_col=0)
            cnv_df = pd.read_csv(os.path.join(self.data_dir, f"TCGA-{cancer}.hallmark_cnv_filtered.csv"), index_col=0)
            met_df = pd.read_csv(os.path.join(self.data_dir, f"TCGA-{cancer}.hallmark_met_filtered.csv"), index_col=0)
            label_df = pd.read_csv(os.path.join(self.data_dir, f"TCGA-{cancer}.subtype_filtered.csv"), index_col=0)
        except Exception as e:
            raise FileNotFoundError(f"数据加载失败: {e}")

        common_samples = label_df.index
        self.rna = rna_df.loc[common_samples]
        self.cnv = cnv_df.loc[common_samples]
        self.met = met_df.loc[common_samples]

        self.label_encoder = LabelEncoder()
        labels_int = self.label_encoder.fit_transform(label_df['Subtype'].values)
        
        self.x_rna = torch.tensor(self.rna.values, dtype=torch.float32)
        self.x_cnv = torch.tensor(self.cnv.values, dtype=torch.float32)
        self.x_met = torch.tensor(self.met.values, dtype=torch.float32)
        self.y = torch.tensor(labels_int, dtype=torch.long)
        
        self.classes_ = self.label_encoder.classes_
        # 🚀 记录 RNA 特征维度
        self.n_rna = self.x_rna.shape[1]

    def get_X_y(self):
        X = np.concatenate([self.x_rna.numpy(), self.x_cnv.numpy(), self.x_met.numpy()], axis=1)
        y = self.y.numpy()
        return X, y, self.n_rna

# ==========================================
# 2. 模型网格配置
# ==========================================
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

PARAM_GRIDS = {
    "Logistic Regression": {
        "model": LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced'),
        "params": {
            "C": [0.01, 0.1, 1, 10], 
            "penalty": ["l1", "l2"]
        },
        "is_gpu": False 
    },
    "Random Forest": {
        "model": RandomForestClassifier(n_jobs=1, class_weight='balanced'),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 10]
        },
        "is_gpu": False 
    },
    "XGBoost": {
        "model": xgb.XGBClassifier(eval_metric="mlogloss", n_jobs=4, tree_method='gpu_hist') if XGB_AVAILABLE else None,
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 6],
            "subsample": [0.8],
            "reg_alpha": [0.1, 0.5, 1] 
        },
        "is_gpu": True 
    },
    "MLP": {
        "model": TorchMLP(epochs=50),
        "params": {
            "hidden_sizes": [(64, 32), (128, 64)],
            "lr": [1e-3, 5e-3]
        },
        "is_gpu": True 
    }
}

# ==========================================
# 3. 核心评估函数
# ==========================================
def tune_and_evaluate(name: str, X_train, y_train, X_test, y_test, args_n_jobs: int) -> Tuple[float, float]:
    cfg = PARAM_GRIDS.get(name)
    if not cfg or cfg["model"] is None: return np.nan, np.nan
    try:
        actual_n_jobs = 1 if cfg["is_gpu"] else args_n_jobs

        search = GridSearchCV(
            estimator=cfg["model"],
            param_grid=cfg["params"],
            scoring="f1_macro",
            cv=5,
            n_jobs=actual_n_jobs,
            verbose=0
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        
        preds = best_model.predict(X_test)
        f1 = float(f1_score(y_test, preds, average="macro"))
        bacc = float(balanced_accuracy_score(y_test, preds))
        return f1, bacc
    except Exception as e:
        print(f"      [WARN] {name} Grid Search 失败: {e}")
        return np.nan, np.nan

# ==========================================
# 4. 主运行逻辑
# ==========================================
def run_experiment(args):
    results = {}
    cancers = args.cancers
    
    for cancer in cancers:
        print(f"\n{'='*60}\n📌 癌症: {cancer} (Subtype Classification)\n{'='*60}")
        
        try:
            ds = SubtypeDataset(cancer, args.base_dir)
            X, y, n_rna = ds.get_X_y()
        except Exception as e:
            print(f"❌ 数据跳过: {e}")
            continue

        print(f"   样本数: {len(y)} | 特征维数: {X.shape[1]} (RNA 维度: {n_rna}) | 类别数: {len(ds.classes_)}")
        
        model_metrics = {k: {"f1": [], "bacc": []} for k in PARAM_GRIDS.keys()}
        
        for seed in range(args.n_seeds):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            train_idx, test_idx = next(sss.split(X, y))
            X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
            y_train, y_test = y[train_idx], y[test_idx]

            # ==========================================
            # 🚀 核心修复：只对 RNA 部分进行 Z-score
            # ==========================================
            scaler = StandardScaler()
            X_train[:, :n_rna] = scaler.fit_transform(X_train[:, :n_rna])
            X_test[:, :n_rna] = scaler.transform(X_test[:, :n_rna])
            # ==========================================

            seed_res = []
            for name in PARAM_GRIDS.keys():
                f1, bacc = tune_and_evaluate(name, X_train, y_train, X_test, y_test, args.n_jobs)
                model_metrics[name]["f1"].append(f1)
                model_metrics[name]["bacc"].append(bacc)
                seed_res.append(f"{name[:3]} F1:{f1:.3f}")
            
            print(f"   Seed {seed+1:02d}/{args.n_seeds}")
        
        results[cancer] = model_metrics

    os.makedirs("results_pm50", exist_ok=True)
    out_file = "results_pm50/baseline_subtype_summary.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("🏆 Subtype Baseline 结果汇总 (Mean ± SD)")
    models = list(PARAM_GRIDS.keys())
    for cancer, m_data in results.items():
        print(f"Cancer: {cancer}")
        for m in models:
            f1s = [a for a in m_data[m]["f1"] if not np.isnan(a)]
            print(f"  {m}: F1 {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/data/zliu/Path_MoE/data")
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--cancers", type=str, nargs="*", default=["BRCA"])
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    run_experiment(args)