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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import roc_auc_score

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 将数据转换为张量并放到 GPU 上
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).view(-1, 1).to(self.device)

        # 动态构建神经网络层
        layers = []
        input_dim = X.shape[1]
        for h in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3)) # Dropout 防过拟合
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        
        self.model_ = nn.Sequential(*layers).to(self.device)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

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
            probs = torch.sigmoid(logits).cpu().numpy()
        # sklearn API 要求二分类返回 [p(y=0), p(y=1)]
        return np.hstack([1 - probs, probs])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)

# ==========================================
# 1. 数据加载器 (Dataset Definition - Top2K Version)
# ==========================================
class SurvivalDataset(Dataset):
    def __init__(self, cancer: str, base_dir: str):
        self.cancer = cancer
        self.data_dir = os.path.join(base_dir, cancer, "filtered_data")
        
        try:
            # 修改为加载 top2k_var 数据文件
            rna_df = pd.read_csv(os.path.join(self.data_dir, f"TCGA-{cancer}.top2k_var_tpm_filtered.csv"), index_col=0)
            cnv_df = pd.read_csv(os.path.join(self.data_dir, f"TCGA-{cancer}.top2k_var_cnv_filtered.csv"), index_col=0)
            met_df = pd.read_csv(os.path.join(self.data_dir, f"TCGA-{cancer}.top2k_var_met_filtered.csv"), index_col=0)
            surv_df = pd.read_csv(os.path.join(self.data_dir, f"TCGA-{cancer}.top2k_var_survival_filtered.csv"), index_col=0)
        except Exception as e:
            raise FileNotFoundError(f"数据加载失败: {e}")

        common_samples = surv_df.index
        self.rna = rna_df.loc[common_samples]
        self.cnv = cnv_df.loc[common_samples]
        self.met = met_df.loc[common_samples]

        # 生存时间中位数切分标签 (0: 长生存, 1: 短生存)
        median_time = surv_df['OS_time'].median()
        self.labels = (surv_df['OS_time'] < median_time).astype(np.int64)
        
        self.x_rna = torch.tensor(self.rna.values, dtype=torch.float32)
        self.x_cnv = torch.tensor(self.cnv.values, dtype=torch.float32)
        self.x_met = torch.tensor(self.met.values, dtype=torch.float32)
        self.y = torch.tensor(self.labels.values, dtype=torch.float32)
        
        # 记录各组学维度
        self.n_rna = self.x_rna.shape[1]
        self.n_cnv = self.x_cnv.shape[1]
        self.n_met = self.x_met.shape[1]

    def get_X_y(self) -> Tuple[np.ndarray, np.ndarray, int]:
        # 拼接特征并返回 Numpy 格式给 sklearn，同时返回 RNA 维度
        X = np.concatenate([self.x_rna.numpy(), self.x_cnv.numpy(), self.x_met.numpy()], axis=1)
        y = self.y.numpy()
        return X, y, self.n_rna

# ==========================================
# 2. 模型网格配置 ( 区分 CPU 和 GPU 的并行策略 )
# ==========================================
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

PARAM_GRIDS = {
    "Logistic Regression": {
        "model": LogisticRegression(solver='liblinear', max_iter=5000),
        "params": {
            "C": [0.01, 0.1, 1, 10, 100], 
            "penalty": ["l1", "l2"]
        },
        "is_gpu": False 
    },
    "Random Forest": {
        "model": RandomForestClassifier(n_jobs=1),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 10]
        },
        "is_gpu": False 
    },
    "XGBoost": {
        "model": xgb.XGBClassifier(eval_metric="logloss", n_jobs=4, tree_method='hist', device='cuda') if XGB_AVAILABLE else None,
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
def tune_and_evaluate(name: str, X_train, y_train, X_test, y_test, args_n_jobs: int) -> float:
    cfg = PARAM_GRIDS.get(name)
    if not cfg or cfg["model"] is None: return np.nan
    try:
        actual_n_jobs = 1 if cfg["is_gpu"] else args_n_jobs

        search = GridSearchCV(
            estimator=cfg["model"],
            param_grid=cfg["params"],
            scoring="roc_auc",
            cv=5,
            n_jobs=actual_n_jobs,
            verbose=0
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        proba = best_model.predict_proba(X_test)[:, 1]
        return float(roc_auc_score(y_test, proba))
    except Exception as e:
        print(f"      [WARN] {name} Grid Search 失败: {e}")
        return np.nan

def run_experiment(args):
    results = {}
    cancers = args.cancers
    
    for cancer in cancers:
        print(f"\n{'='*60}\n📌 癌症: {cancer} (Top 2000 Variance Mode)\n{'='*60}")
        
        try:
            ds = SurvivalDataset(cancer, args.base_dir)
            X, y, n_rna = ds.get_X_y()
        except Exception as e:
            print(f"❌ 数据跳过: {e}")
            continue

        print(f"   样本数: {len(y)} | 特征维数: {X.shape[1]} (RNA cols: 0-{n_rna-1})")
        
        model_aucs = {k: [] for k in PARAM_GRIDS.keys()}
        
        for seed in range(args.n_seeds):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            train_idx, test_idx = next(sss.split(X, y))
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # ==========================================
            # 🚀 核心修复：只对矩阵的前 n_rna 列（即 RNA 数据）做 Z-score
            # ==========================================
            scaler = StandardScaler()
            # 仅拟合训练集的 RNA 部分，并转换训练/测试集的对应列
            X_train[:, :n_rna] = scaler.fit_transform(X_train[:, :n_rna])
            X_test[:, :n_rna] = scaler.transform(X_test[:, :n_rna])
            # ==========================================

            seed_res = []
            for name in PARAM_GRIDS.keys():
                auc = tune_and_evaluate(name, X_train, y_train, X_test, y_test, args.n_jobs)
                model_aucs[name].append(auc)
                seed_res.append(f"{name[:3]}:{auc:.3f}")
            
            print(f"   Seed {seed+1:02d}/{args.n_seeds} | {' | '.join(seed_res)}")
        
        results[cancer] = model_aucs

    # 保存结果
    os.makedirs("results", exist_ok=True)
    out_file = "results/baseline_top2k_summary.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # 打印最终表格汇总
    print("\n" + "="*80)
    print("🏆 Top 2000 Baseline 结果汇总 (Mean AUC ± SD)")
    models = list(PARAM_GRIDS.keys())
    header = "| Cancer | " + " | ".join(models) + " |"
    print(header)
    print("|" + "---|" * (len(models) + 1))
    for cancer, m_data in results.items():
        row = [cancer]
        for m in models:
            aucs = [a for a in m_data[m] if not np.isnan(a)]
            row.append(f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}" if aucs else "—")
        print("| " + " | ".join(row) + " |")
    print("="*80)

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/data/zliu/Path_MoE/data")
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--n_jobs", type=int, default=8, help="并行进程数(仅针对CPU模型)")
    parser.add_argument("--cancers", type=str, nargs="*", 
                        default=["BLCA", "BRCA", "HNSC", "KIRC", "LGG", "LUAD", "LUSC", "PRAD", "STAD", "THCA"])
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    run_experiment(args)