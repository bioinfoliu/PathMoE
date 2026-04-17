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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

# 导入自定义模块
from dataset_subtype import SubtypeDataset
from model_moe_pm50 import TopKPathMoE
from utils import create_pathway_mask

# ==========================================
# 🌟 核心：组学消融逻辑 (Zero-masking)
# ==========================================
def apply_omics_combination(x_rna, x_cnv, x_met, comb: str):
    """根据选择的组学组合，将不需要的组学张量强制置零"""
    if comb == "all":
        return x_rna, x_cnv, x_met
        
    if "rna" not in comb:
        x_rna = torch.zeros_like(x_rna)
    if "cnv" not in comb:
        x_cnv = torch.zeros_like(x_cnv)
    if "met" not in comb:
        x_met = torch.zeros_like(x_met)
        
    return x_rna, x_cnv, x_met

def get_omics_status(batch) -> Tuple[List[int], List[int], List[int], List[int]]:
    x_rna = batch["x_rna"]
    x_cnv = batch["x_cnv"]
    x_met = batch["x_met"]

    rna_sum = x_rna.detach().cpu().abs().sum(dim=1)
    cnv_sum = x_cnv.detach().cpu().abs().sum(dim=1)
    met_sum = x_met.detach().cpu().abs().sum(dim=1)

    has_rna = (rna_sum > 0).to(torch.int64).tolist()
    has_cnv = (cnv_sum > 0).to(torch.int64).tolist()
    has_met = (met_sum > 0).to(torch.int64).tolist()

    counts = [int(h_r + h_c + h_m) for h_r, h_c, h_m in zip(has_rna, has_cnv, has_met)]
    return has_rna, has_cnv, has_met, counts

def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)

def load_json_config(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

def apply_config_overrides(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)
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
        noise_std=args.noise_std
    ).to(device)
    return model

# ==========================================
# 2. 核心训练逻辑 / Core Training Logic
# ==========================================

def train_one_model(
    ds: SubtypeDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    mask_matrix: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[TopKPathMoE, float]:
    """
    训练单模型 (含组学消融和早停)
    """
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False)

    model = build_model(mask_matrix=mask_matrix, num_classes=ds.num_classes, args=args, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    class_weights = ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_f1 = 0.0
    best_state = None
    patience = 10
    counter = 0

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            # 🌟 组学消融过滤
            x_r, x_c, x_m = apply_omics_combination(batch["x_rna"].to(device), batch["x_cnv"].to(device), batch["x_met"].to(device), args.omics_comb)
            y = batch["y"].to(device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x_r, x_c, x_m)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # 验证 F1
        model.eval()
        preds: List[int] = []
        targets: List[int] = []
        with torch.no_grad():
            for batch in val_loader:
                # 🌟 组学消融过滤
                x_r, x_c, x_m = apply_omics_combination(batch["x_rna"].to(device), batch["x_cnv"].to(device), batch["x_met"].to(device), args.omics_comb)
                logits, _ = model(x_r, x_c, x_m)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
                targets.extend(batch["y"].numpy().tolist())

        try:
            val_f1 = float(f1_score(targets, preds, average="macro"))
        except Exception:
            val_f1 = 0.0

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            
        if counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1

def evaluate_ensemble(
    models: List[TopKPathMoE],
    test_loader: DataLoader,
    test_y_true: np.ndarray,
    device: torch.device,
    omics_comb: str,
    pathway_names: List[str]
) -> Tuple[float, float, float, pd.DataFrame, pd.DataFrame]:
    """
    使用 5-Fold 模型集成进行预测并提取平均门控权重 (包含组学消融)
    """
    for m in models:
        m.eval()
        
    all_status = []
    gating_data = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 🌟 组学消融过滤
            x_r, x_c, x_m = apply_omics_combination(batch["x_rna"].to(device), batch["x_cnv"].to(device), batch["x_met"].to(device), omics_comb)
            ids = batch["id"]
            has_rna, has_cnv, has_met, counts = get_omics_status(batch)
            
            batch_probs = []
            batch_gates = []
            
            for model in models:
                logits, gate_weights = model(x_r, x_c, x_m)
                probs = F.softmax(logits, dim=1)
                batch_probs.append(probs)
                batch_gates.append(gate_weights)
            
            # 对 5 个模型的概率和权重取平均
            avg_probs = torch.mean(torch.stack(batch_probs), dim=0).cpu().numpy()
            avg_gates = torch.mean(torch.stack(batch_gates), dim=0).cpu().numpy()
            preds = np.argmax(avg_probs, axis=1)
            
            all_probs.extend(avg_probs)
            all_preds.extend(preds)
            
            for i, s_id in enumerate(ids):
                # 记录预测状态
                all_status.append({
                    "sample_id": s_id,
                    "has_RNA": has_rna[i],
                    "has_CNV": has_cnv[i],
                    "has_MET": has_met[i],
                    "omics_count": counts[i]
                })
                # 记录门控权重
                rec = {"sample_id": s_id}
                for j, p_name in enumerate(pathway_names):
                    rec[p_name] = float(avg_gates[i, j])
                gating_data.append(rec)

    all_preds = np.array(all_preds)
    
    try:
        acc = float(accuracy_score(test_y_true, all_preds))
        b_acc = float(balanced_accuracy_score(test_y_true, all_preds))
        macro_f1 = float(f1_score(test_y_true, all_preds, average="macro"))
    except Exception:
        acc, b_acc, macro_f1 = 0.0, 0.0, 0.0
        
    status_df = pd.DataFrame(all_status)
    # 🌟 修复: 添加预测的 pred_label
    status_df["pred_label"] = all_preds
    
    gating_df = pd.DataFrame(gating_data)
    
    return acc, b_acc, macro_f1, status_df, gating_df

# ==========================================
# 3. 主实验逻辑 / Main Experiment Logic
# ==========================================

def run_nested_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING) 
    except Exception as e:
        raise RuntimeError("请先安装 optuna: pip install optuna")

    device = get_device(args.device)
    print(f"🚀 Starting Nested Subtype Ablation: {args.cancer} | Comb: {args.omics_comb} | Device={device}")
    print(f"⚙️  Settings: {args.seeds} Seeds x {args.optuna_trials} Tuning Trials | Noise Std: {args.noise_std}")

    # 1. 加载数据
    ds = SubtypeDataset(args.cancer, args.base_dir)
    labels = ds.y
    num_classes = ds.num_classes
    
    gene_df = pd.read_csv(os.path.join(args.base_dir, args.cancer, "filtered_subtype_data", f"TCGA-{args.cancer}.hallmark_tpm_filtered.csv"), index_col=0)
    gene_list = gene_df.columns.tolist()
    mask_matrix, pathway_names = create_pathway_mask(args.gmt_file, gene_list)
    mask_matrix = mask_matrix.to(device)
    
    # 结果容器
    f1_list, bacc_list = [], []
    all_best_params = []
    
    # 🌟 独立创建生存期消融专属目录 (使用绝对路径防护)
    ABS_ROOT = "/data/zliu/Path_MoE"
    res_dir = os.path.join(ABS_ROOT, "results_moi_ablation_subtype")
    pred_dir = os.path.join(ABS_ROOT, "predictions_moi_ablation_subtype")
    gate_dir = os.path.join(ABS_ROOT, "gating_moi_ablation_subtype")
    ckpt_dir = os.path.join(ABS_ROOT, "checkpoints_moi_ablation_subtype")
    
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    if args.save_gating: os.makedirs(gate_dir, exist_ok=True)
    if hasattr(args, 'save_models') and args.save_models: os.makedirs(ckpt_dir, exist_ok=True)

    # 确定 JSON 默认保存路径
    default_json_path = os.path.join(res_dir, f"ablation_{args.cancer}_{args.omics_comb}_summary.json")
    actual_json_path = args.save_results_json if args.save_results_json else default_json_path

    # ----------------------------------------------------
    #  Outer Loop: Independent Splits
    # ----------------------------------------------------
    for seed in range(args.seeds):
        try:
            print(f"\n" + "="*60)
            print(f"🌱 [Outer Loop] Seed {seed+1}/{args.seeds} (Random State: {seed})")
            print(f"="*60)
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # 1. Outer Split: 80% Train/Val (Dev), 20% Test
            sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed)
            dev_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))
            
            # 🌟 泄漏防护 (Leakage-Free Scaling)
            current_ds = copy.copy(ds)
            scaler = StandardScaler()
            scaler.fit(ds.x_rna[dev_idx].numpy())
            current_ds.x_rna = torch.tensor(scaler.transform(ds.x_rna.numpy()), dtype=torch.float32)
            
            # 准备 Test Data
            test_loader = DataLoader(Subset(current_ds, test_idx), batch_size=args.batch_size, shuffle=False)
            test_y_true = labels[test_idx]
            
            # 获取 Dev 数据的索引和标签
            dev_indices = np.array(range(len(ds)))[dev_idx]
            dev_labels = labels[dev_idx]

            # ------------------------------------------------
            #  Inner Loop: Optuna Tuning (调 Fold 0)
            # ------------------------------------------------
            print(f"   🔍 Tuning hyperparams on Fold 1...")
            
            skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=seed)
            folds = list(skf.split(dev_indices, dev_labels))
            f0_tr_idx, f0_val_idx = dev_indices[folds[0][0]], dev_indices[folds[0][1]]

            def objective(trial: "optuna.Trial") -> float:
                trial_args = copy.deepcopy(args)
                trial_args.save_models = False 
                
                trial_args.top_k = trial.suggest_int("top_k", 1, 5) 
                trial_args.gate_hidden_dim = trial.suggest_int("gate_hidden_dim", 32, 128)   
                trial_args.expert_hidden_dim = trial.suggest_int("expert_hidden_dim", 16, 64)
                trial_args.expert_out_dim = trial.suggest_int("expert_out_dim", 16, 64)
                trial_args.expert_dropout = trial.suggest_float("expert_dropout", 0.3, 0.6)
                trial_args.cls_hidden_dim = trial.suggest_int("cls_hidden_dim", 16, 32)
                trial_args.lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
                trial_args.weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

                _, val_f1 = train_one_model(
                    ds=current_ds, train_idx=f0_tr_idx, val_idx=f0_val_idx,
                    mask_matrix=mask_matrix, args=trial_args, device=device
                )
                return val_f1

            # 运行 Optuna
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=args.optuna_trials)
            
            best_params = study.best_params
            print(f"   🏆 Best Params for Seed {seed}: {best_params}")
            all_best_params.append({"seed": seed, "params": best_params, "val_f1": study.best_value})

            # ------------------------------------------------
            #  Final Training: 5-Fold Ensemble Retraining
            # ------------------------------------------------
            print(f"   🔄 Retraining 5-Fold Ensemble with Best Params...")
            
            final_run_args = copy.deepcopy(args)
            for k, v in best_params.items():
                setattr(final_run_args, k, v)
            
            final_fold_models = []
            
            for fold, (train_sub_idx, val_sub_idx) in enumerate(folds):
                train_idx = dev_indices[train_sub_idx]
                val_idx = dev_indices[val_sub_idx]

                model, _ = train_one_model(
                    ds=current_ds, train_idx=train_idx, val_idx=val_idx,
                    mask_matrix=mask_matrix, args=final_run_args, device=device
                )
                final_fold_models.append(model)
                
                # 保存最终模型
                if hasattr(args, 'save_models') and args.save_models:
                    ckpt_path = os.path.join(ckpt_dir, f"{args.cancer}_s{seed}_{args.omics_comb}_fold{fold+1}.pth")
                    torch.save(model.state_dict(), ckpt_path)

            # ------------------------------------------------
            #  Final Evaluation: Test on the Locked 20%
            # ------------------------------------------------
            print(f"   🧪 Evaluating Ensemble on Locked Test Set (20%)...")
            test_acc, test_bacc, test_f1, status_df, gating_df = evaluate_ensemble(
                models=final_fold_models, 
                test_loader=test_loader, 
                test_y_true=test_y_true, 
                device=device,
                omics_comb=args.omics_comb,
                pathway_names=pathway_names
            )

            # 🌟 保存预测详情 (已包含 pred_label 和 true_label)
            status_df["true_label"] = test_y_true
            pred_out_path = os.path.join(pred_dir, f"{args.cancer}_s{seed}_{args.omics_comb}_detailed.csv")
            status_df.to_csv(pred_out_path, index=False)

            if args.save_gating:
                gating_out_path = os.path.join(gate_dir, f"{args.cancer}_s{seed}_{args.omics_comb}_gating.csv")
                gating_df.to_csv(gating_out_path, index=False)

            print(f"   🏁 Seed {seed} | Balanced Acc: {test_bacc:.4f} | Macro F1: {test_f1:.4f}")
            f1_list.append(test_f1)
            bacc_list.append(test_bacc)

            # 🛡️ 实时更新 JSON
            _temp_result = {
                "cancer": args.cancer,
                "comb": args.omics_comb,
                "completed_seeds": seed + 1,
                "mean_test_f1": float(np.mean(f1_list)),
                "std_test_f1": float(np.std(f1_list)),
                "mean_test_bacc": float(np.mean(bacc_list)),
                "std_test_bacc": float(np.std(bacc_list)),
                "seed_test_f1s": [float(x) for x in f1_list],
                "seed_test_baccs": [float(x) for x in bacc_list],
                "best_params_history": all_best_params
            }
            with open(actual_json_path, "w") as f:
                json.dump(_temp_result, f, indent=2)

        except Exception as e:
            print(f"❌ Error in Seed {seed}: {str(e)}")
            traceback.print_exc()
            continue

    # ================= 汇总统计 =================
    if not f1_list:
        return {"error": "No results generated"}

    mean_f1 = np.mean(f1_list)
    std_f1 = np.std(f1_list)
    mean_bacc = np.mean(bacc_list)
    std_bacc = np.std(bacc_list)

    result = {
        "cancer": args.cancer,
        "comb": args.omics_comb,
        "seeds": args.seeds,
        "mean_test_f1": float(mean_f1),
        "std_test_f1": float(std_f1),
        "mean_test_bacc": float(mean_bacc),
        "std_test_bacc": float(std_bacc),
        "seed_test_f1s": [float(x) for x in f1_list],
        "seed_test_baccs": [float(x) for x in bacc_list],
        "best_params_history": all_best_params
    }

    with open(actual_json_path, "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"\n🏆 Final 20-Seed Ensemble Result for [{args.omics_comb}]:")
    print(f"   ➤ Mean Balanced Acc: {mean_bacc:.4f} ± {std_bacc:.4f}")
    print(f"   ➤ Mean Macro F1    : {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"💾 Final results saved to {actual_json_path}")
        
    return result

# ==========================================
# 4. 参数解析与入口 / Argparser & Entry
# ==========================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Nested Subtype Analysis with Path-MoE & Omics Ablation")
    p.add_argument("--base_dir", type=str, default="/data/zliu/Path_MoE/data/")
    p.add_argument("--gmt_file", type=str, default="/data/zliu/Path_MoE/data/h.all.v2023.1.Hs.symbols.gmt")
    p.add_argument("--cancer", type=str, default="BRCA")
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--seeds", type=int, default=20, help="Outer loop seeds (Total Experiments)")
    p.add_argument("--optuna_trials", type=int, default=30, help="Inner loop tuning trials")
    
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-3)

    p.add_argument("--num_omics", type=int, default=3)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--gate_hidden_dim", type=int, default=128)
    p.add_argument("--expert_hidden_dim", type=int, default=64)
    p.add_argument("--expert_out_dim", type=int, default=16)
    p.add_argument("--expert_dropout", type=float, default=0.3)
    p.add_argument("--no_expert_bn", action="store_true")
    p.add_argument("--cls_hidden_dim", type=int, default=8)

    # 🌟 引入核心的噪声和类别参数
    p.add_argument("--noise_std", type=float, default=0.2, help="Noisy Gating parameter")
    p.add_argument("--num_classes", type=int, default=5, help="PAM50 has 5 classes")

    # 🌟 引入组学控制参数
    p.add_argument("--omics_comb", type=str, default="all", help="Choices: rna, cnv, met, rna_cnv, rna_met, cnv_met, all")

    p.add_argument("--save_gating", action="store_true")
    p.add_argument("--gating_out_csv", type=str, default="")
    p.add_argument("--save_results_json", type=str, default="")
    p.add_argument("--save_models", action="store_true")
    p.add_argument("--config", type=str, default="")

    p.add_argument("--use_softmax", action="store_true", help="Use Softmax instead of Sigmoid for gating ablation")
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