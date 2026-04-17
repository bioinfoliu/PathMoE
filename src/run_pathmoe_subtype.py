import argparse
import copy
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import scipy.stats as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from dataset_subtype import SubtypeDataset
from model_moe_pm50 import TopKPathMoE
from utils import create_pathway_mask

def get_omics_status(batch) -> Tuple[List[int], List[int], List[int], List[int]]:
    """获取批次中各组学的存在状态 / Get omics presence status in batch"""
    x_rna, x_cnv, x_met = batch["x_rna"], batch["x_cnv"], batch["x_met"]
    rna_sum = x_rna.detach().cpu().abs().sum(dim=1)
    cnv_sum = x_cnv.detach().cpu().abs().sum(dim=1)
    met_sum = x_met.detach().cpu().abs().sum(dim=1)
    
    has_rna = (rna_sum > 0).to(torch.int64).tolist()
    has_cnv = (cnv_sum > 0).to(torch.int64).tolist()
    has_met = (met_sum > 0).to(torch.int64).tolist()
    
    counts = [int(h_r + h_c + h_m) for h_r, h_c, h_m in zip(has_rna, has_cnv, has_met)]
    return has_rna, has_cnv, has_met, counts

def get_device(device_str: str) -> torch.device:
    """智能获取设备 / Get device intelligently"""
    if device_str == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(): 
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)

def load_json_config(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path): return {}
    with open(path, "r") as f: return json.load(f)

def apply_config_overrides(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    for k, v in cfg.items():
        if hasattr(args, k): setattr(args, k, v)
    return args

def build_model(mask_matrix: torch.Tensor, num_classes: int, args: argparse.Namespace, device: torch.device) -> TopKPathMoE:
    model = TopKPathMoE(
        gene_mask=mask_matrix,
        num_classes=num_classes,
        num_omics=args.num_omics,
        top_k=args.top_k,
        gate_hidden_dim=args.gate_hidden_dim,
        expert_hidden_dim=args.expert_hidden_dim,
        expert_out_dim=args.expert_out_dim,
        expert_dropout=args.expert_dropout,
        expert_use_bn=not args.no_expert_bn,
        cls_hidden_dim=args.cls_hidden_dim,
        use_softmax=args.use_softmax,
        use_tri_gating=args.use_tri_gating,
        noise_std=args.noise_std 
    ).to(device)
    return model

# ==========================================
# 2. 核心训练逻辑 / Core Training Logic
# ==========================================

def train_one_model(
    ds: SubtypeDataset, train_idx: np.ndarray, val_idx: np.ndarray,
    mask_matrix: torch.Tensor, args: argparse.Namespace, device: torch.device,
) -> Tuple[TopKPathMoE, float]:
    """训练单模型 (含早停)"""
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False)

    model = build_model(mask_matrix=mask_matrix, num_classes=ds.num_classes, args=args, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    class_weights = ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_f1, best_state = 0.0, None
    patience, counter = 10, 0

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            x_rna, x_cnv, x_met = batch["x_rna"].to(device), batch["x_cnv"].to(device), batch["x_met"].to(device)
            y = batch["y"].to(device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x_rna, x_cnv, x_met)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                x_rna, x_cnv, x_met = batch["x_rna"].to(device), batch["x_cnv"].to(device), batch["x_met"].to(device)
                logits, _ = model(x_rna, x_cnv, x_met)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
                targets.extend(batch["y"].numpy().tolist())

        val_f1 = float(f1_score(targets, preds, average="macro"))
        if val_f1 > best_f1:
            best_f1, best_state, counter = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            counter += 1
        if counter >= patience: break

    if best_state is not None: model.load_state_dict(best_state)
    return model, best_f1

# 🚀 核心修改：5-Fold 集成评估逻辑
def evaluate_ensemble(
    models: List[TopKPathMoE], test_loader: DataLoader, test_y_true: np.ndarray, 
    pathway_names: List[str], device: torch.device
) -> Tuple[float, float, float, pd.DataFrame, pd.DataFrame]:
    """最终模型评估 (5-Fold 平均) / Final Ensemble Evaluation"""
    for m in models: m.eval()
    
    all_probs, all_preds, all_status = [], [], []
    gating_data = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_rna, x_cnv, x_met = batch["x_rna"].to(device), batch["x_cnv"].to(device), batch["x_met"].to(device)
            has_rna, has_cnv, has_met, counts = get_omics_status(batch)
            ids = batch["id"]
            
            batch_probs, batch_gates = [], []
            for m in models:
                logits, gate_weights = m(x_rna, x_cnv, x_met)
                probs = F.softmax(logits, dim=1)
                batch_probs.append(probs)
                batch_gates.append(gate_weights)
            
            # 取 5 个模型的概率和门控平均值
            avg_probs = torch.mean(torch.stack(batch_probs), dim=0).cpu().numpy()
            avg_gates = torch.mean(torch.stack(batch_gates), dim=0).cpu().numpy()
            preds = np.argmax(avg_probs, axis=1)
            
            all_probs.extend(avg_probs)
            all_preds.extend(preds)
            
            for i, s_id in enumerate(ids):
                # 记录预测状态
                all_status.append({
                    "sample_id": s_id, "has_RNA": has_rna[i],
                    "has_CNV": has_cnv[i], "has_MET": has_met[i], "omics_count": counts[i]
                })
                # 记录平均门控权重
                rec = {"sample_id": s_id}
                rec.update({p: float(avg_gates[i, j]) for j, p in enumerate(pathway_names)})
                gating_data.append(rec)

    all_preds = np.array(all_preds)
    acc = float(accuracy_score(test_y_true, all_preds))
    b_acc = float(balanced_accuracy_score(test_y_true, all_preds))
    macro_f1 = float(f1_score(test_y_true, all_preds, average="macro"))
    
    status_df = pd.DataFrame(all_status)
    status_df["pred_label"] = all_preds  # 🚀 添加预测标签到 DataFrame 中
    gating_df = pd.DataFrame(gating_data)
    
    return acc, b_acc, macro_f1, status_df, gating_df

# ==========================================
# 3. 主实验逻辑 / Main Experiment Logic
# ==========================================

def run_nested_experiment(args: argparse.Namespace):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING) # 减少刷屏
    device = get_device(args.device)
    
    # 🚀 强制锁定绝对路径
    ABS_ROOT = "/data/zliu/Path_MoE"
    pred_dir = os.path.join(ABS_ROOT, "predictions_subtype")
    gate_dir = os.path.join(ABS_ROOT, "gating_subtype")
    ckpt_dir = os.path.join(ABS_ROOT, "checkpoints_subtype")
    for d in [pred_dir, gate_dir, ckpt_dir]: os.makedirs(d, exist_ok=True)

    print(f"🚀 Starting Subtype Classification (5-Fold Ensemble): {args.cancer}")

    ds = SubtypeDataset(args.cancer, args.base_dir)
    labels = ds.y
    num_classes = ds.num_classes
    
    gene_df = pd.read_csv(os.path.join(args.base_dir, args.cancer, "filtered_subtype_data", f"TCGA-{args.cancer}.hallmark_tpm_filtered.csv"), index_col=0)
    gene_list = gene_df.columns.tolist()
    mask_matrix, pathway_names = create_pathway_mask(args.gmt_file, gene_list)
    mask_matrix = mask_matrix.to(device)
    
    f1_list, bacc_list = [], []

    for seed in range(args.seeds):
        try:
            print(f"\n🌱 [Seed {seed+1}/{args.seeds}]")
            torch.manual_seed(seed); np.random.seed(seed)
            
            # Outer Split (80/20)
            sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed)
            dev_idx, test_idx = next(sss_outer.split(np.zeros(len(labels)), labels))
            
            # 标准化
            current_ds = copy.copy(ds)
            scaler = StandardScaler()
            scaler.fit(ds.x_rna[dev_idx].numpy())
            current_ds.x_rna = torch.tensor(scaler.transform(ds.x_rna.numpy()), dtype=torch.float32)

            # 🚀 Inner Split (5-Fold CV on Dev set)
            dev_labels = labels[dev_idx]
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            folds = list(skf.split(np.zeros(len(dev_labels)), dev_labels))

            # 🚀 Optuna 调参 (仅使用 Fold 0 提升效率)
            print(f"   🔍 Tuning hyperparams on Fold 1...")
            f0_tr_idx, f0_val_idx = dev_idx[folds[0][0]], dev_idx[folds[0][1]]
            
            def objective(trial):
                t_args = copy.deepcopy(args)
                t_args.top_k = trial.suggest_int("top_k", 1, 5)
                t_args.lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
                _, val_f1 = train_one_model(current_ds, f0_tr_idx, f0_val_idx, mask_matrix, t_args, device)
                return val_f1

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=args.optuna_trials)
            print(f"   🏆 Best Params for Seed {seed}: {study.best_params}")
            
            best_args = copy.deepcopy(args)
            for k, v in study.best_params.items(): setattr(best_args, k, v)
            
            # 🚀 训练 5 个 Fold 的 Ensemble 模型
            ensemble_models = []
            for fold_idx, (tr_sub, val_sub) in enumerate(folds):
                print(f"   ⚙️ Training Ensemble Fold {fold_idx+1}/5...")
                f_tr_idx, f_val_idx = dev_idx[tr_sub], dev_idx[val_sub]
                model, _ = train_one_model(current_ds, f_tr_idx, f_val_idx, mask_matrix, best_args, device)
                ensemble_models.append(model)
                
                # 保存每个 Fold 的模型 (可选)
                if args.save_models:
                    m_path = os.path.join(ckpt_dir, f"{args.cancer}_s{seed}_fold{fold_idx}.pth")
                    torch.save(model.state_dict(), m_path)
            
            test_loader = DataLoader(Subset(current_ds, test_idx), batch_size=args.batch_size)
            acc, b_acc, f1, status_df, gating_df = evaluate_ensemble(
                ensemble_models, test_loader, labels[test_idx], pathway_names, device
            )
            
            # 保存预测和门控
            status_df["true_label"] = labels[test_idx]
            status_df.to_csv(os.path.join(pred_dir, f"{args.cancer}_seed{seed}_detailed.csv"), index=False)
            if args.save_gating:
                gating_df.to_csv(os.path.join(gate_dir, f"{args.cancer}_s{seed}_gating.csv"), index=False)
            
            f1_list.append(f1); bacc_list.append(b_acc)
            print(f"   🏁 Seed {seed} | Balanced Acc: {b_acc:.4f} | Macro F1: {f1:.4f}")

        except Exception: traceback.print_exc(); continue

    print(f"\n🏆 Final 20-Seed Ensemble Result:")
    print(f"   ➤ Mean Balanced Acc: {np.mean(bacc_list):.4f} ± {np.std(bacc_list):.4f}")
    print(f"   ➤ Mean Macro F1    : {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")


    if args.save_results_json:
        result_dict = {
            "cancer": args.cancer,
            "seeds": args.seeds,
            "mean_test_bacc": float(np.mean(bacc_list)),
            "std_test_bacc": float(np.std(bacc_list)),
            "mean_test_f1": float(np.mean(f1_list)),
            "std_test_f1": float(np.std(f1_list)),
            "seed_test_baccs": [float(x) for x in bacc_list],
            "seed_test_f1s": [float(x) for x in f1_list]
        }
        
        # 确保目录存在，防止 FileNotFoundError
        os.makedirs(os.path.dirname(os.path.abspath(args.save_results_json)), exist_ok=True)
        
        # 写入 JSON
        with open(args.save_results_json, "w") as f:
            json.dump(result_dict, f, indent=4)
        print(f"💾 成功！最终结果已保存至: {args.save_results_json}")

# ==========================================
# 4. 参数解析与入口 / Argparser & Entry
# ==========================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full Subtype Fast Path-MoE")
    p.add_argument("--base_dir", type=str, default="/data/zliu/Path_MoE/data/")
    p.add_argument("--gmt_file", type=str, default="/data/zliu/Path_MoE/data/h.all.v2023.1.Hs.symbols.gmt")
    p.add_argument("--cancer", type=str, default="BRCA")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--optuna_trials", type=int, default=30)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--num_omics", type=int, default=3)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--use_tri_gating", action="store_true", help="Use concatenated RNA+CNV+MET for the router")
    p.add_argument("--noise_std", type=float, default=0.2) 
    p.add_argument("--gate_hidden_dim", type=int, default=128)
    p.add_argument("--expert_hidden_dim", type=int, default=64)
    p.add_argument("--expert_out_dim", type=int, default=16)
    p.add_argument("--num_classes", type=int, default=5)
    p.add_argument("--expert_dropout", type=float, default=0.3)
    p.add_argument("--no_expert_bn", action="store_true")
    p.add_argument("--cls_hidden_dim", type=int, default=8)
    p.add_argument("--save_gating", action="store_true")
    p.add_argument("--gating_out_csv", type=str, default="")
    p.add_argument("--save_results_json", type=str, default="")
    p.add_argument("--save_models", action="store_true")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--use_softmax", action="store_true")
    return p

if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()

    cfg = load_json_config(args.config)
    args = apply_config_overrides(args, cfg)

    if args.gating_out_csv == "": args.gating_out_csv = None
    if args.save_results_json == "": args.save_results_json = None

    try:
        run_nested_experiment(args)
    except Exception as e:
        print(f"❌ Critical Failure: {e}")
        traceback.print_exc()