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
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 0. 自定义 PyTorch MLP (兼容 sklearn 接口)
# ==========================================
class TorchMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_sizes=(128, 64), lr=1e-3, epochs=50, batch_size=32):
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
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3)) 
            input_dim = h
        layers.append(nn.Linear(input_dim, self.num_classes_)) 
        
        self.model_ = nn.Sequential(*layers).to(self.device)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        
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
# 1. METABRIC 数据加载器
# ==========================================
class MetabricNpyDataset:
    def __init__(self, data_dir: str):
        print(f"📥 Loading METABRIC npy files from {data_dir}...")
        self.x_rna = np.load(os.path.join(data_dir, "mrna.npy"))
        self.x_cnv = np.load(os.path.join(data_dir, "cnv.npy"))
        self.x_met = np.load(os.path.join(data_dir, "met.npy"))
        self.y = np.load(os.path.join(data_dir, "y.npy"))
        self.n_rna = self.x_rna.shape[1]

    def get_X_y(self):
        X = np.concatenate([self.x_rna, self.x_cnv, self.x_met], axis=1)
        return X, self.y, self.n_rna

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
        "params": {"C": [0.1, 1, 10]},
        "is_gpu": False 
    },
    "Random Forest": {
        "model": RandomForestClassifier(class_weight='balanced'),
        "params": {"n_estimators": [100, 200], "max_depth": [10, None]},
        "is_gpu": False 
    },
    "XGBoost": {
        "model": xgb.XGBClassifier(tree_method='gpu_hist', eval_metric="mlogloss") if XGB_AVAILABLE else None,
        "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
        "is_gpu": True 
    },
    "MLP": {
        "model": TorchMLP(epochs=50),
        "params": {"hidden_sizes": [(128, 64), (256, 128)], "lr": [1e-3]},
        "is_gpu": True 
    }
}

# ==========================================
# 3. 核心运行与保存逻辑
# ==========================================
def run_metabric_experiment(args):
    try:
        ds = MetabricNpyDataset(args.data_dir)
        X, y, n_rna = ds.get_X_y()
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    print(f"✅ Samples: {len(y)} | RNA Dim: {n_rna} | Total Dim: {X.shape[1]}")
    model_metrics = {k: {"f1": [], "bacc": []} for k in PARAM_GRIDS.keys()}
    
    for seed in range(args.n_seeds):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(sss.split(X, y))
        X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train[:, :n_rna] = scaler.fit_transform(X_train[:, :n_rna])
        X_test[:, :n_rna] = scaler.transform(X_test[:, :n_rna])

        for name, cfg in PARAM_GRIDS.items():
            if cfg["model"] is None: continue
            search = GridSearchCV(cfg["model"], cfg["params"], scoring="f1_macro", cv=5, n_jobs=1 if cfg["is_gpu"] else args.n_jobs)
            search.fit(X_train, y_train)
            preds = search.best_estimator_.predict(X_test)
            model_metrics[name]["f1"].append(f1_score(y_test, preds, average="macro"))
            model_metrics[name]["bacc"].append(balanced_accuracy_score(y_test, preds))
        
        print(f"   [Seed {seed:02d}] Progressing...")

    # --- 保存结果逻辑 ---
    final_results = []
    print("\n" + "="*60)
    print("🏆 METABRIC Baseline Summary (Mean ± SD)")
    print("="*60)

    for m in PARAM_GRIDS.keys():
        f1s = [v for v in model_metrics[m]["f1"] if not np.isnan(v)]
        baccs = [v for v in model_metrics[m]["bacc"] if not np.isnan(v)]
        if f1s:
            res = {
                "Method": m,
                "F1_Mean": np.mean(f1s),
                "F1_Std": np.std(f1s),
                "BAcc_Mean": np.mean(baccs),
                "BAcc_Std": np.std(baccs)
            }
            final_results.append(res)
            print(f"{m:20s}: F1 {res['F1_Mean']:.4f} ± {res['F1_Std']:.4f}")

    # 保存为 CSV
    os.makedirs(args.out_dir, exist_ok=True)
    df_res = pd.DataFrame(final_results)
    csv_path = os.path.join(args.out_dir, "metabric_baseline_results.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\n📂 结果已保存至: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/zliu/Path_MoE/data/brca_metabric/processed")
    parser.add_argument("--out_dir", type=str, default="/data/zliu/Path_MoE/results_baseline")
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--n_jobs", type=int, default=16)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run_metabric_experiment(args)