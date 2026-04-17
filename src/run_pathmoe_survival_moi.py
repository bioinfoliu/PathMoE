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
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

# 导入自定义模块
from dataset_survival import SurvivalDataset
from model_moe import TopKPathMoE
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
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，但指定了 --device cuda")
    if device_str == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS 不可用，但指定了 --device mps")
    return torch.device(device_str)

def load_json_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r") as f:
        return json.load(f)

def apply_config_overrides(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args

def build_model(mask_matrix: torch.Tensor, args: argparse.Namespace, device: torch.device) -> TopKPathMoE:
    model = TopKPathMoE(
        gene_mask=mask_matrix,
        num_classes=args.num_classes, # 🚀 修复点：传递类别参数
        num_omics=args.num_omics,
        top_k=args.top_k,
        gate_hidden_dim=args.gate_hidden_dim,
        expert_hidden_dim=args.expert_hidden_dim,
        expert_out_dim=args.expert_out_dim,
        expert_dropout=args.expert_dropout,
        expert_use_bn=not args.no_expert_bn,
        cls_hidden_dim=args.cls_hidden_dim,
        use_softmax=args.use_softmax,
        noise_std=args.noise_std  # 🚀 修复点：将噪声参数传递给底层模型
    ).to(device)
    return model

def train_one_fold(
    ds: SurvivalDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    mask_matrix: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[TopKPathMoE, float]:
    """
    训练单个Fold的模型 (加入组学消融 和 Early Stopping)
    """
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False)

    model = build_model(mask_matrix=mask_matrix, args=args, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_state = None
    
    # === Early Stopping 设置 ===
    patience = 10
    counter = 0
    # ==========================

    for _epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            # 🌟 组学消融过滤
            x_r, x_c, x_m = apply_omics_combination(batch["x_rna"].to(device), batch["x_cnv"].to(device), batch["x_met"].to(device), args.omics_comb)
            y = batch["y"].to(device).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x_r, x_c, x_m)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # 验证 AUC
        model.eval()
        preds: List[float] = []
        targets: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                # 🌟 组学消融过滤
                x_r, x_c, x_m = apply_omics_combination(batch["x_rna"].to(device), batch["x_cnv"].to(device), batch["x_met"].to(device), args.omics_comb)
                logits, _ = model(x_r, x_c, x_m)
                preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten().tolist())
                targets.extend(batch["y"].detach().cpu().numpy().flatten().tolist())

        try:
            auc = float(roc_auc_score(targets, preds))
        except Exception:
            auc = 0.5

        # === Early Stopping 核心逻辑 ===
        if auc > best_auc:
            best_auc = auc
            best_state = copy.deepcopy(model.state_dict())
            counter = 0  # 性能提升，重置计数器
        else:
            counter += 1  # 性能未提升，计数器加1
            
        if counter >= patience:
            break
        # ==============================

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def evaluate_ensemble(
    fold_models: List[TopKPathMoE],
    test_loader: DataLoader,
    test_y_true: np.ndarray,
    device: torch.device,
    omics_comb: str,
    pathway_names: List[str]
) -> Tuple[float, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    使用 5-Fold 模型集成进行预测并提取平均门控权重 (合并为单次遍历)
    """
    for m in fold_models:
        m.eval()
        
    all_status = []
    gating_data = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 🌟 组学消融过滤
            x_r, x_c, x_m = apply_omics_combination(batch["x_rna"].to(device), batch["x_cnv"].to(device), batch["x_met"].to(device), omics_comb)
            ids = batch["id"]
            has_rna, has_cnv, has_met, counts = get_omics_status(batch)
            
            batch_probs = []
            batch_gates = []
            
            for model in fold_models:
                logits, gate_weights = model(x_r, x_c, x_m)
                probs = torch.sigmoid(logits).flatten()
                batch_probs.append(probs)
                batch_gates.append(gate_weights)
            
            # 对 5 个模型的概率和权重取平均
            avg_probs = torch.mean(torch.stack(batch_probs), dim=0).cpu().numpy()
            avg_gates = torch.mean(torch.stack(batch_gates), dim=0).cpu().numpy()
            
            all_probs.extend(avg_probs)
            
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

    avg_probs_array = np.array(all_probs, dtype=np.float64)
    try:
        auc = float(roc_auc_score(test_y_true, avg_probs_array))
    except Exception:
        auc = 0.5
        
    status_df = pd.DataFrame(all_status)
    gating_df = pd.DataFrame(gating_data)
    
    return auc, avg_probs_array, status_df, gating_df


# ==========================================
#         Core Logic: Nested Experiment
# ==========================================

def run_nested_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING) # 减少刷屏
    except Exception as e:
        raise RuntimeError("请先安装 optuna: pip install optuna")

    device = get_device(args.device)
    print(f"🚀 Starting Nested Survival Ablation: {args.cancer} | Comb: {args.omics_comb} | Device={device}")
    print(f"⚙️  Settings: {args.seeds} Seeds x {args.optuna_trials} Tuning Trials | Noise Std: {args.noise_std}")

    # 1. 加载数据
    ds = SurvivalDataset(args.cancer, args.base_dir)
    labels = ds.get_labels()
    
    mask_matrix, pathway_names = create_pathway_mask(args.gmt_file, ds.gene_list)
    mask_matrix = mask_matrix.to(device)
    
    # 结果容器
    final_test_results = []
    all_best_params = []
    
    # 🌟 独立创建生存期消融专属目录 (使用绝对路径防护)
    ABS_ROOT = "/data/zliu/Path_MoE"
    res_dir = os.path.join(ABS_ROOT, "results_survival_ablation")
    pred_dir = os.path.join(ABS_ROOT, "predictions_survival_ablation")
    gate_dir = os.path.join(ABS_ROOT, "gating_survival_ablation")
    ckpt_dir = os.path.join(ABS_ROOT, "checkpoints_survival_ablation")
    
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    if args.save_gating: os.makedirs(gate_dir, exist_ok=True)
    if hasattr(args, 'save_models') and args.save_models: os.makedirs(ckpt_dir, exist_ok=True)

    # 确定 JSON 默认保存路径（包含组合名）
    default_json_path = os.path.join(res_dir, f"ablation_{args.cancer}_{args.omics_comb}.json")
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
            
            # 准备 Test Data
            test_loader = DataLoader(Subset(ds, test_idx), batch_size=args.batch_size, shuffle=False)
            test_y_true = labels[test_idx]
            
            # 获取 Dev 数据的索引和标签
            dev_indices = np.array(range(len(ds)))[dev_idx]
            dev_labels = labels[dev_idx]

            # ------------------------------------------------
            #  Inner Loop: Optuna Tuning (🚀 优化加速：仅调 Fold 0)
            # ------------------------------------------------
            print(f"   🔍 Tuning hyperparams on Fold 1...")
            
            skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=seed)
            folds = list(skf.split(dev_indices, dev_labels))
            f0_tr_idx, f0_val_idx = dev_indices[folds[0][0]], dev_indices[folds[0][1]]

            def objective(trial: "optuna.Trial") -> float:
                # 采样超参数
                trial_args = copy.deepcopy(args)
                trial_args.save_models = False # 调参时不存模型
                
                trial_args.top_k = trial.suggest_int("top_k", 1, 3) 
                trial_args.gate_hidden_dim = trial.suggest_int("gate_hidden_dim", 32, 128)   
                trial_args.expert_hidden_dim = trial.suggest_int("expert_hidden_dim", 16, 64)
                trial_args.expert_out_dim = trial.suggest_int("expert_out_dim", 16, 64)
                trial_args.expert_dropout = trial.suggest_float("expert_dropout", 0.3, 0.6)
                trial_args.cls_hidden_dim = trial.suggest_int("cls_hidden_dim", 16, 32)
                trial_args.lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
                trial_args.weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

                _, val_auc = train_one_fold(
                    ds=ds, train_idx=f0_tr_idx, val_idx=f0_val_idx,
                    mask_matrix=mask_matrix, args=trial_args, device=device
                )
                return val_auc

            # 运行 Optuna
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=args.optuna_trials)
            
            best_params = study.best_params
            print(f"   🏆 Best Params for Seed {seed}: {best_params}")
            all_best_params.append({"seed": seed, "params": best_params, "val_auc": study.best_value})

            # ------------------------------------------------
            #  Final Training: 5-Fold Ensemble Retraining
            # ------------------------------------------------
            print(f"   🔄 Retraining 5-Fold Ensemble with Best Params...")
            
            # 构建最终参数
            final_run_args = copy.deepcopy(args)
            for k, v in best_params.items():
                setattr(final_run_args, k, v)
            
            final_fold_models = []
            
            for fold, (train_sub_idx, val_sub_idx) in enumerate(folds):
                train_idx = dev_indices[train_sub_idx]
                val_idx = dev_indices[val_sub_idx]

                model, _ = train_one_fold(
                    ds=ds, train_idx=train_idx, val_idx=val_idx,
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
            test_auc, y_pred_scores, status_df, gating_df = evaluate_ensemble(
                fold_models=final_fold_models, 
                test_loader=test_loader, 
                test_y_true=test_y_true, 
                device=device,
                omics_comb=args.omics_comb,
                pathway_names=pathway_names
            )

            # 🌟 保存预测详情
            status_df["true_label"] = test_y_true
            status_df["pred_score"] = y_pred_scores
            pred_out_path = os.path.join(pred_dir, f"{args.cancer}_s{seed}_{args.omics_comb}_detailed.csv")
            status_df.to_csv(pred_out_path, index=False)

            if args.save_gating:
                gating_out_path = os.path.join(gate_dir, f"{args.cancer}_s{seed}_{args.omics_comb}_gating.csv")
                gating_df.to_csv(gating_out_path, index=False)

            print(f"   🏁 Seed {seed} Final Test AUC: {test_auc:.4f}")
            final_test_results.append(test_auc)

            # 🛡️ 实时更新 JSON
            _temp_mean = np.mean(final_test_results)
            _temp_std = np.std(final_test_results)
            _temp_result = {
                "cancer": args.cancer,
                "comb": args.omics_comb,
                "completed_seeds": seed + 1,
                "mean_test_auc": float(_temp_mean),
                "std_test_auc": float(_temp_std),
                "seed_test_aucs": [float(x) for x in final_test_results],
                "best_params_history": all_best_params
            }
            with open(actual_json_path, "w") as f:
                json.dump(_temp_result, f, indent=2)

        except Exception as e:
            print(f"❌ Error in Seed {seed}: {str(e)}")
            traceback.print_exc()
            continue

    # ================= 汇总统计 =================
    if not final_test_results:
        return {"error": "No results generated"}

    mean_test_auc = np.mean(final_test_results)
    std_test_auc = np.std(final_test_results)

    if len(final_test_results) > 1:
        ci_bound = st.t.interval(0.95, len(final_test_results)-1, loc=mean_test_auc, scale=st.sem(final_test_results))
        ci_lower, ci_upper = ci_bound
    else:
        ci_lower, ci_upper = mean_test_auc, mean_test_auc

    result = {
        "cancer": args.cancer,
        "comb": args.omics_comb,
        "seeds": args.seeds,
        "mean_test_auc": float(mean_test_auc),
        "std_test_auc": float(std_test_auc),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "seed_test_aucs": [float(x) for x in final_test_results],
        "best_params_history": all_best_params
    }

    with open(actual_json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"💾 Final results saved to {actual_json_path}")
        
    return result


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Nested Survival Analysis with Path-MoE & Omics Ablation")
    p.add_argument("--base_dir", type=str, default="/data/zliu/Path_MoE/data/")
    p.add_argument("--gmt_file", type=str, default="/data/zliu/Path_MoE/data/h.all.v2023.1.Hs.symbols.gmt")
    p.add_argument("--cancer", type=str, default="BRCA")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])

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

    # 🌟 引入核心的噪声和类别参数 (修复点)
    p.add_argument("--noise_std", type=float, default=0.2, help="Noisy Gating parameter")
    p.add_argument("--num_classes", type=int, default=1, help="Must be 1 for survival analysis")

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

    run_nested_experiment(args)