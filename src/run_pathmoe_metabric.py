import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import ast
import re
import copy
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

# 导入你自己的模型和工具
from model_moe_pm50 import TopKPathMoE
from utils import create_pathway_mask

def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def load_metabric_data(data_dir: str):
    print("📥 Loading preprocessed METABRIC data (from .npy)...")
    
    x_rna_np = np.load(os.path.join(data_dir, "mrna.npy"))
    x_cnv_np = np.load(os.path.join(data_dir, "cnv.npy"))
    x_met_np = np.load(os.path.join(data_dir, "met.npy"))
    y_np = np.load(os.path.join(data_dir, "y.npy"))
    
    gene_list_df = pd.read_csv(os.path.join(data_dir, "tcga_gene_list.csv"), header=0)
    gene_list = gene_list_df.iloc[:, 0].tolist() 
    
    sample_ids_df = pd.read_csv(os.path.join(data_dir, "sample_ids.csv"), header=0)
    sample_ids = sample_ids_df.iloc[:, 0].tolist()
    
    # 转换为 Tensor
    tensor_rna = torch.tensor(x_rna_np, dtype=torch.float32)
    tensor_cnv = torch.tensor(x_cnv_np, dtype=torch.float32)
    tensor_met = torch.tensor(x_met_np, dtype=torch.float32)
    tensor_y = torch.tensor(y_np, dtype=torch.long)

    # 保持与训练时一致的标准化
    from sklearn.preprocessing import StandardScaler
    scaler_rna = StandardScaler()
    tensor_rna = torch.tensor(scaler_rna.fit_transform(tensor_rna.numpy()), dtype=torch.float32)
    
    dataset = TensorDataset(tensor_rna, tensor_cnv, tensor_met, tensor_y)
    
    def collate_fn(batch):
        r, c, m, y = zip(*batch)
        return {
            "x_rna": torch.stack(r),
            "x_cnv": torch.stack(c),
            "x_met": torch.stack(m),
            "y": torch.stack(y),
            "id": sample_ids[:len(batch)] 
        }
        
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    return loader, y_np, gene_list, sample_ids

def build_model(mask_matrix: torch.Tensor, args: argparse.Namespace, device: torch.device) -> TopKPathMoE:
    model = TopKPathMoE(
        gene_mask=mask_matrix,
        num_classes=args.num_classes,
        num_omics=args.num_omics,
        top_k=args.top_k,
        gate_hidden_dim=args.gate_hidden_dim,
        expert_hidden_dim=args.expert_hidden_dim,
        expert_out_dim=args.expert_out_dim,
        expert_dropout=args.expert_dropout,
        expert_use_bn=not args.no_expert_bn,
        cls_hidden_dim=args.cls_hidden_dim,
        use_softmax=args.use_softmax,
        noise_std=0.0 
    ).to(device)
    return model

def load_seed_args(seed: int, base_args: argparse.Namespace, log_path: str) -> argparse.Namespace:
    """
    🌟 根据 Seed 从 log 文件中动态正则匹配最佳参数
    """
    seed_args = copy.deepcopy(base_args)
    
    if not os.path.exists(log_path):
        print(f"   ⚠️ Warning: Log file not found at {log_path}. Size mismatch likely!")
        return seed_args

    # 正则表达式寻找格式如: "Best Params for Seed 0: {'top_k': 5, 'lr': 0.0003...}"
    target_pattern = re.compile(rf"Best Params for Seed {seed}:\s*({{.*?}})")
    
    best_params = None
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = target_pattern.search(line)
            if match:
                dict_str = match.group(1)
                try:
                    # 安全地将字符串转化为 Python 字典
                    best_params = ast.literal_eval(dict_str)
                except Exception as e:
                    print(f"   ❌ Error parsing dict string for Seed {seed}: {e}")
                break # 找到后就可以跳出文件读取了
    
    if best_params:
        # 将解析出来的参数覆盖到 seed_args 中
        for k, v in best_params.items():
            if hasattr(seed_args, k):
                setattr(seed_args, k, v)
        print(f"   🔧 Parsed log for Seed {seed}: Updated {list(best_params.keys())}")
    else:
        print(f"   ⚠️ Warning: Could not find best params for Seed {seed} in log. Using defaults.")
        
    return seed_args

def run_external_validation(args):
    device = get_device(args.device)
    print(f"🚀 Starting METABRIC External Validation | Device: {device}")
    
    # 1. 加载数据
    metabric_dir = "/data/zliu/Path_MoE/data/brca_metabric/processed"
    test_loader, y_true, gene_list, sample_ids = load_metabric_data(metabric_dir)
    
    print(f"   ➤ Test Samples: {len(y_true)}")
    print(f"   ➤ Gene Dimension: {len(gene_list)}")
    
    # 2. 生成完全一致的 Mask Matrix
    mask_matrix, pathway_names = create_pathway_mask(args.gmt_file, gene_list)
    mask_matrix = mask_matrix.to(device)
    
    seed_f1s = []
    seed_baccs = []
    
    ABS_ROOT = "/data/zliu/Path_MoE"
    log_file_path = os.path.join(ABS_ROOT, "logs_subtype/BRCA_subtype_run.log") # 🌟 指定日志路径
    
    os.makedirs(os.path.join(ABS_ROOT, "results_metabric"), exist_ok=True)
    
    gate_dir = ""
    if args.save_gating:
        gate_dir = os.path.join(ABS_ROOT, "gating_metabric")
        os.makedirs(gate_dir, exist_ok=True)

    for seed in range(args.seeds):
        print(f"\n🌱 Evaluating Ensemble Models from Seed {seed}...")
        
        # 🌟 传入日志文件的路径进行解析获取专属超参数
        seed_args = load_seed_args(seed, args, log_file_path)
        
        ensemble_models = []
        for fold in range(5): 
            ckpt_path = os.path.join(ABS_ROOT, f"checkpoints_subtype/BRCA_s{seed}_fold{fold}.pth") 
            if not os.path.exists(ckpt_path):
                print(f"   ⚠️ Checkpoint not found: {ckpt_path}")
                continue
                
            # 🌟 使用解析到的 seed_args 构架专属模型
            model = build_model(mask_matrix, seed_args, device)
            
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
            except RuntimeError as e:
                print(f"   ❌ Size mismatch error loading {ckpt_path}. Error: {e}")
                continue
                
            model.eval()

            # ========================================================
            # 🚀 补丁 1：强制开启 BN 层的 Test-Time Adaptation
            # ========================================================
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm1d):
                    m.train()
            # ========================================================

            ensemble_models.append(model)
            
        if len(ensemble_models) == 0:
            print(f"   ❌ No models loaded for Seed {seed}. Skipping...")
            continue
            
        print(f"   ✅ Successfully loaded {len(ensemble_models)} fold models.")
        
        # 3. 对当前 Seed 进行集成预测
        seed_preds_probs = []
        gating_data = [] # 🌟 用于存储当前 Seed 的门控权重
        
        with torch.no_grad():
            for batch in test_loader:
                x_r, x_c, x_m = batch["x_rna"].to(device), batch["x_cnv"].to(device), batch["x_met"].to(device)
                ids = batch["id"] 
                
                batch_probs = []
                batch_gates = [] 
                
                for m in ensemble_models:
                    logits, gate_weights = m(x_r, x_c, x_m)
                    probs = F.softmax(logits, dim=1)
                    batch_probs.append(probs)
                    batch_gates.append(gate_weights)
                    
                # 当前 batch 所有模型的平均概率和门控权重
                avg_batch_probs = torch.mean(torch.stack(batch_probs), dim=0)
                avg_batch_gates = torch.mean(torch.stack(batch_gates), dim=0).cpu().numpy()
                
                seed_preds_probs.append(avg_batch_probs.cpu().numpy())
                
                # 🌟 将门控权重格式化并记录
                for i, s_id in enumerate(ids):
                    rec = {"sample_id": s_id}
                    for j, p_name in enumerate(pathway_names):
                        rec[p_name] = float(avg_batch_gates[i, j])
                    gating_data.append(rec)
                
        seed_preds_probs = np.concatenate(seed_preds_probs, axis=0)
        final_preds = np.argmax(seed_preds_probs, axis=1)
        
        # 🌟 保存 Gating 为 CSV
        if args.save_gating:
            gating_df = pd.DataFrame(gating_data)
            out_csv_path = os.path.join(gate_dir, f"METABRIC_s{seed}_gating.csv")
            gating_df.to_csv(out_csv_path, index=False)
        
        # 4. 计算指标
        acc = accuracy_score(y_true, final_preds)
        bacc = balanced_accuracy_score(y_true, final_preds)
        macro_f1 = f1_score(y_true, final_preds, average="macro")
        
        print(f"   🏁 Seed {seed} Validation -> Macro F1: {macro_f1:.4f} | Balanced Acc: {bacc:.4f}")
        seed_f1s.append(macro_f1)
        seed_baccs.append(bacc)
        
    print("\n========================================================")
    print(f"🏆 Final METABRIC External Validation Result (Across {len(seed_f1s)} Seeds)")
    if len(seed_f1s) > 0:
        print(f"   ➤ Mean Macro F1    : {np.mean(seed_f1s):.4f} ± {np.std(seed_f1s):.4f}")
        print(f"   ➤ Mean Balanced Acc: {np.mean(seed_baccs):.4f} ± {np.std(seed_baccs):.4f}")
        if args.save_gating:
            print(f"   💾 Gating weights saved to: {gate_dir}")
    else:
        print("   ❌ No results generated. Please check checkpoint paths.")
    print("========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gmt_file", type=str, default="/data/zliu/Path_MoE/data/h.all.v2023.1.Hs.symbols.gmt")
    parser.add_argument("--device", type=str, default="cuda:0") 
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--num_omics", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=5) 
    
    # 这里的参数变为了 Fallback Default，实际会被 load_seed_args 从 log 文件覆盖
    parser.add_argument("--top_k", type=int, default=3) 
    parser.add_argument("--gate_hidden_dim", type=int, default=128)
    parser.add_argument("--expert_hidden_dim", type=int, default=64)
    parser.add_argument("--expert_out_dim", type=int, default=16)
    parser.add_argument("--expert_dropout", type=float, default=0.3)
    parser.add_argument("--no_expert_bn", action="store_true")
    parser.add_argument("--cls_hidden_dim", type=int, default=8)
    parser.add_argument("--use_softmax", action="store_true")
    
    # 🌟 必须加上这个才能保存门控权重
    parser.add_argument("--save_gating", action="store_true", help="Save extracted gating weights to CSV")
    
    args = parser.parse_args()
    run_external_validation(args)