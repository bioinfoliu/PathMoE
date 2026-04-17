#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
严格对齐点：
- 数据来源：只用 Path-MoE 的 `SubtypeDataset`（样本对齐、标签编码、三组学张量一致）
- 特征空间：MoGCN 输入 features = concat([x_rna, x_cnv, x_met])（与你模型看到的 x_full 对齐）

- 外层：20 seeds（0..seeds-1），每个 seed 用 StratifiedShuffleSplit 做 80/20(dev/test)
- 内层：dev(80%) 上 StratifiedKFold 5-fold 训练 5 个模型，test 上做平均概率集成
- 标准化：默认严格复用 run_pathmoe_subtype.py 的实现（只对 RNA 用 dev 统计做 StandardScaler）
         可选开启对 x_full 做 dev-only z-score（更“严格”的 train/dev 统计口径）
         
- 不平衡：class-weighted CrossEntropy（balanced weights，等价于 Path-MoE 的 weighted CE 思路）
- 早停：val macro-F1 早停（与 Path-MoE 对齐）
- 指标：macro-F1 与 balanced accuracy（与 Path-MoE 对齐）
- 输出：
  - JSON: /data/zliu/Path_MoE/results_subtype/MoGCN_BRCA_summary.json
  - ckpt: /data/zliu/Path_MoE/checkpoints_subtype/MoGCN_BRCA_s{seed}_fold{fold}.pth

注意：
- 图邻接使用 SNF（无监督），由三组学特征构建；为对齐“同一输入空间”，邻接用与该 seed 相同标准化口径的三组学特征构建。
"""

import argparse
import copy
import json
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------
# 1) 你的数据管线（严格复用）
# -----------------------------
from dataset_subtype import SubtypeDataset

# -----------------------------
# 2) MoGCN（保持架构不变）
# -----------------------------
MOGCN_ROOT = "/data/zliu/MoGCN-master"
if MOGCN_ROOT not in sys.path:
    sys.path.insert(0, MOGCN_ROOT)

from gcn_model import GCN  # noqa: E402
from utils import setup_seed_strict  # noqa: E402


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _require_snf():
    try:
        import snf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("缺少依赖 snf（通常来自 snfpy）。请先安装：pip install snfpy") from e
    return snf


def build_adj_from_omics_snf(
    x_rna: np.ndarray,
    x_cnv: np.ndarray,
    x_met: np.ndarray,
    threshold: float,
    metric: str,
    K: int,
    mu: float,
) -> np.ndarray:
    """
    SNF 融合 -> 阈值过滤 -> 取 0/1 邻接 -> D^{-1}A 行归一化。
    该“行归一化”实现对齐 MoGCN-master/utils.py: load_data()
    """
    snf = _require_snf()

    affinity_nets = snf.make_affinity([x_rna, x_cnv, x_met], metric=metric, K=K, mu=mu)
    fused = snf.snf(affinity_nets, K=K)

    adj_m = fused.copy()
    np.fill_diagonal(adj_m, 0.0)
    adj_m[adj_m < threshold] = 0.0

    exist = (adj_m != 0) * 1.0
    deg = exist.dot(np.ones(exist.shape[1], dtype=np.float32))
    deg_safe = deg.copy()
    deg_safe[deg_safe == 0] = 1.0
    adj_hat = exist / deg_safe[:, None]
    return adj_hat.astype(np.float32)


@torch.no_grad()
def predict_proba(model: GCN, features: torch.Tensor, adj: torch.Tensor) -> np.ndarray:
    model.eval()
    logits = model(features, adj)
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    return probs


def train_one_fold(
    features: torch.Tensor,
    adj: torch.Tensor,
    labels: torch.Tensor,
    idx_train: torch.Tensor,
    idx_val: torch.Tensor,
    nclass: int,
    hidden: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    device: torch.device,
) -> Tuple[GCN, float]:
    """
    Full-batch GCN 单 fold 训练；val macro-F1 早停；class-weighted CE。
    """
    model = GCN(n_in=int(features.shape[1]), n_hid=int(hidden), n_out=int(nclass), dropout=float(dropout)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    y_train_np = labels[idx_train].detach().cpu().numpy()
    cls_w = compute_class_weight(class_weight="balanced", classes=np.arange(nclass), y=y_train_np)
    cls_w_t = torch.tensor(cls_w, dtype=torch.float32, device=device)

    best_f1 = -1.0
    best_state = None
    bad = 0

    for _epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(features, adj)
        loss = F.cross_entropy(out[idx_train], labels[idx_train], weight=cls_w_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(features, adj)[idx_val]
            val_pred = torch.argmax(val_logits, dim=1).detach().cpu().numpy()
            val_true = labels[idx_val].detach().cpu().numpy()
        val_f1 = float(f1_score(val_true, val_pred, average="macro"))

        if val_f1 > best_f1 + 1e-12:
            best_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_f1)


@dataclass
class Summary:
    cancer: str
    seeds: int
    mean_test_bacc: float
    std_test_bacc: float
    mean_test_f1: float
    std_test_f1: float
    seed_test_baccs: List[float]
    seed_test_f1s: List[float]
    config: Dict[str, object]


def evaluate_one_seed(seed: int, ds: SubtypeDataset, args: argparse.Namespace, device: torch.device) -> Tuple[float, float]:
    """
    严格对齐 run_pathmoe_subtype.py 的 seed 协议：
    outer(80/20) + dev 5-fold ensemble + test 集成评估
    """
    setup_seed_strict(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    labels_np = ds.y
    nclass = int(ds.num_classes)

    sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=float(args.test_size), random_state=seed)
    dev_idx, test_idx = next(sss_outer.split(np.zeros(len(labels_np)), labels_np))

    # (1) 标准化：严格复用 run_pathmoe_subtype.py（默认仅标准化 RNA，统计来自 dev）
    current_ds = copy.copy(ds)
    if bool(args.standardize_rna):
        scaler = StandardScaler()
        scaler.fit(ds.x_rna[dev_idx].numpy())
        current_ds.x_rna = torch.tensor(scaler.transform(ds.x_rna.numpy()), dtype=torch.float32)

    # (2) 特征空间：拼接为 x_full（与 MoPE-MOI/Path-MoE 对齐）
    x_full = torch.cat([current_ds.x_rna, current_ds.x_cnv, current_ds.x_met], dim=1).numpy().astype(np.float32)

    # 可选：对 x_full 做 dev-only z-score（更严格的“仅用训练/开发统计量”）
    if bool(args.feature_zscore):
        mu = x_full[dev_idx].mean(axis=0)
        sd = x_full[dev_idx].std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        x_full = (x_full - mu) / sd

    # (3) 邻接：SNF 构图（无监督）。使用与该 seed 同口径的三组学特征。
    adj_hat = build_adj_from_omics_snf(
        x_rna=current_ds.x_rna.numpy().astype(np.float32),
        x_cnv=current_ds.x_cnv.numpy().astype(np.float32),
        x_met=current_ds.x_met.numpy().astype(np.float32),
        threshold=float(args.threshold),
        metric=str(args.snf_metric),
        K=int(args.snf_k),
        mu=float(args.snf_mu),
    )

    features = torch.tensor(x_full, dtype=torch.float32, device=device)
    labels = torch.tensor(labels_np, dtype=torch.long, device=device)
    adj = torch.tensor(adj_hat, dtype=torch.float32, device=device)

    # (4) inner 5-fold on dev
    dev_labels = labels_np[dev_idx]
    skf = StratifiedKFold(n_splits=int(args.k_folds), shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(len(dev_labels)), dev_labels))

    # (5) nested：仅用 fold0 做超参搜索（与 run_pathmoe_subtype.py 的 Optuna 风格对齐）
    hidden = int(args.hidden)
    dropout = float(args.dropout)
    lr = float(args.lr)
    weight_decay = float(args.weight_decay)
    patience = int(args.patience)

    if int(args.optuna_trials) > 0:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        f0_tr_sub, f0_val_sub = folds[0]
        f0_tr_idx = dev_idx[f0_tr_sub]
        f0_val_idx = dev_idx[f0_val_sub]
        idx_train0 = torch.tensor(f0_tr_idx, dtype=torch.long, device=device)
        idx_val0 = torch.tensor(f0_val_idx, dtype=torch.long, device=device)

        def objective(trial: "optuna.Trial") -> float:
            t_hidden = trial.suggest_categorical("hidden", [32, 64, 128])
            t_dropout = trial.suggest_float("dropout", 0.1, 0.8)
            t_lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
            t_wd = trial.suggest_float("weight_decay", 1e-5, 5e-2, log=True)
            t_pat = trial.suggest_int("patience", 10, 40)
            _m, best_f1 = train_one_fold(
                features=features,
                adj=adj,
                labels=labels,
                idx_train=idx_train0,
                idx_val=idx_val0,
                nclass=nclass,
                hidden=t_hidden,
                dropout=t_dropout,
                lr=t_lr,
                weight_decay=t_wd,
                epochs=int(args.epochs),
                patience=t_pat,
                device=device,
            )
            return float(best_f1)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=int(args.optuna_trials))
        best = study.best_params
        hidden = int(best.get("hidden", hidden))
        dropout = float(best.get("dropout", dropout))
        lr = float(best.get("lr", lr))
        weight_decay = float(best.get("weight_decay", weight_decay))
        patience = int(best.get("patience", patience))

    # (6) 训练 5 个 fold 模型并保存 checkpoint
    ckpt_dir = "/data/zliu/Path_MoE/checkpoints_subtype"
    os.makedirs(ckpt_dir, exist_ok=True)

    fold_models: List[GCN] = []
    for fold_i, (tr_sub, val_sub) in enumerate(folds):
        tr_idx = dev_idx[tr_sub]
        val_idx = dev_idx[val_sub]
        idx_train = torch.tensor(tr_idx, dtype=torch.long, device=device)
        idx_val = torch.tensor(val_idx, dtype=torch.long, device=device)

        m, _ = train_one_fold(
            features=features,
            adj=adj,
            labels=labels,
            idx_train=idx_train,
            idx_val=idx_val,
            nclass=nclass,
            hidden=hidden,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=int(args.epochs),
            patience=patience,
            device=device,
        )
        fold_models.append(m)

        if bool(args.save_models):
            ckpt_path = os.path.join(ckpt_dir, f"MoGCN_{args.cancer}_s{seed}_fold{fold_i}.pth")
            torch.save(
                {
                    "state_dict": m.state_dict(),
                    "seed": int(seed),
                    "fold": int(fold_i),
                    "cancer": str(args.cancer),
                    "hidden": int(hidden),
                    "dropout": float(dropout),
                    "lr": float(lr),
                    "weight_decay": float(weight_decay),
                    "patience": int(patience),
                    "threshold": float(args.threshold),
                    "snf_metric": str(args.snf_metric),
                    "snf_k": int(args.snf_k),
                    "snf_mu": float(args.snf_mu),
                    "standardize_rna": bool(args.standardize_rna),
                    "feature_zscore": bool(args.feature_zscore),
                },
                ckpt_path,
            )

    # (7) test 集成：平均概率 -> 指标
    probs_sum = None
    for m in fold_models:
        probs = predict_proba(m, features, adj)
        probs_sum = probs if probs_sum is None else (probs_sum + probs)
    avg_probs = probs_sum / float(len(fold_models))

    test_probs = avg_probs[test_idx]
    test_pred = np.argmax(test_probs, axis=1)
    test_true = labels_np[test_idx]

    macro_f1 = float(f1_score(test_true, test_pred, average="macro"))
    bacc = float(balanced_accuracy_score(test_true, test_pred))
    return macro_f1, bacc


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run MoGCN with Path-MoE-aligned subtype pipeline")
    p.add_argument("--base_dir", type=str, default="/data/zliu/Path_MoE/data/")
    p.add_argument("--cancer", type=str, default="BRCA")
    p.add_argument("--device", type=str, default="auto")

    # protocol
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--test_size", type=float, default=0.2)

    # alignment toggles
    p.add_argument("--standardize_rna", action="store_true", help="严格复用 run_pathmoe_subtype.py：dev 统计 StandardScaler 到 RNA")
    p.add_argument("--feature_zscore", action="store_true", help="可选：对 x_full 用 dev 统计做 z-score（更严格）")

    # MoGCN/SNF graph
    p.add_argument("--threshold", type=float, default=0.005)
    p.add_argument("--snf_metric", type=str, default="sqeuclidean")
    p.add_argument("--snf_k", type=int, default=20)
    p.add_argument("--snf_mu", type=float, default=0.5)

    # GCN hyperparams
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--optuna_trials", type=int, default=0, help=">0 时启用 nested tuning（仅 fold0）")

    # output
    p.add_argument("--save_models", action="store_true")
    p.add_argument(
        "--save_results_json",
        type=str,
        default="/data/zliu/Path_MoE/results_subtype/MoGCN_BRCA_summary.json",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    device = get_device(args.device)

    # 和 run_pathmoe_subtype.py 的默认行为保持一致：如果不传 flag，就不做 RNA scaler
    # 但“严格对齐”通常建议开启 --standardize_rna

    ds = SubtypeDataset(args.cancer, args.base_dir)

    seed_f1s: List[float] = []
    seed_baccs: List[float] = []

    for seed in range(int(args.seeds)):
        try:
            f1, bacc = evaluate_one_seed(seed=seed, ds=ds, args=args, device=device)
            seed_f1s.append(float(f1))
            seed_baccs.append(float(bacc))
            print(f"🏁 Seed {seed} | Balanced Acc: {bacc:.4f} | Macro F1: {f1:.4f}")
        except Exception:
            traceback.print_exc()
            continue

    out = Summary(
        cancer=str(args.cancer),
        seeds=int(args.seeds),
        mean_test_bacc=float(np.mean(seed_baccs)) if len(seed_baccs) else float("nan"),
        std_test_bacc=float(np.std(seed_baccs)) if len(seed_baccs) else float("nan"),
        mean_test_f1=float(np.mean(seed_f1s)) if len(seed_f1s) else float("nan"),
        std_test_f1=float(np.std(seed_f1s)) if len(seed_f1s) else float("nan"),
        seed_test_baccs=[float(x) for x in seed_baccs],
        seed_test_f1s=[float(x) for x in seed_f1s],
        config={
            "threshold": float(args.threshold),
            "snf_metric": str(args.snf_metric),
            "snf_k": int(args.snf_k),
            "snf_mu": float(args.snf_mu),
            "epochs": int(args.epochs),
            "hidden": int(args.hidden),
            "dropout": float(args.dropout),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "optuna_trials": int(args.optuna_trials),
            "standardize_rna": bool(args.standardize_rna),
            "feature_zscore": bool(args.feature_zscore),
            "feature_space": "concat([RNA,CNV,MET])",
            "adjacency": "SNF->threshold->exist(0/1)->row_norm(D^{-1}A)",
        },
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.save_results_json)), exist_ok=True)
    with open(args.save_results_json, "w") as f:
        json.dump(out.__dict__, f, indent=4)
    print(f"💾 已保存 JSON: {args.save_results_json}")


if __name__ == "__main__":
    main()

