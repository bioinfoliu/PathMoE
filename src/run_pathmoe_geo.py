import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from model_moe_pm50 import TopKPathMoE
from utils import create_pathway_mask

def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def load_external_data(data_dir: str):
    """
    加载外部验证数据（如 GSE25066 或 METABRIC）
    """
    print(f"📥 Loading preprocessed validation data from: {data_dir}")
    
    # 读取预处理好的 npy 矩阵
    # 对于 GSE25066，cnv.npy 和 met.npy 是全零矩阵
    x_rna_np = np.load(os.path.join(data_dir, "mrna.npy"))
    x_cnv_np = np.load(os.path.join(data_dir, "cnv.npy"))
    x_met_np = np.load(os.path.join(data_dir, "met.npy"))
    y_np = np.load(os.path.join(data_dir, "y.npy"))
    
    # 读取基因列表和样本 ID
    gene_list_df = pd.read_csv(os.path.join(data_dir, "tcga_gene_list.csv"))
    gene_list = gene_list_df.iloc[:, 0].tolist() 
    
    sample_ids_df = pd.read_csv(os.path.join(data_dir, "sample_ids.csv"))
    sample_ids = sample_ids_df.iloc[:, 0].tolist()
    
    # 转换为 Tensor
    tensor_rna = torch.tensor(x_rna_np, dtype=torch.float32)
    tensor_cnv = torch.tensor(x_cnv_np, dtype=torch.float32)
    tensor_met = torch.tensor(x_met_np, dtype=torch.float32)
    tensor_y = torch.tensor(y_np, dtype=torch.long)


    scaler_rna = StandardScaler()
    rna_scaled = scaler_rna.fit_transform(tensor_rna.numpy())
    tensor_rna = torch.tensor(rna_scaled, dtype=torch.float32)
    
    dataset = TensorDataset(tensor_rna, tensor_cnv, tensor_met, tensor_y)
    
    def collate_fn(batch):
        r, c, m, y = zip(*batch)
        return {
            "x_rna": torch.stack(r),
            "x_cnv": torch.stack(c),
            "x_met": torch.stack(m),
            "y": torch.stack(y)
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

def run_external_validation(args):
    device = get_device(args.device)
    
    # 自动识别数据集名称用于打印和保存
    dataset_name = os.path.basename(os.path.dirname(args.data_dir))
    print(f"🚀 Starting External Validation on {dataset_name} | Device: {device}")
    
    # 1. 加载数据
    test_loader, y_true, gene_list, sample_ids = load_external_data(args.data_dir)
    
    print(f"   ➤ Test Samples: {len(y_true)}")
    print(f"   ➤ Gene Dimension: {len(gene_list)}")
    
    # 2. 生成 Mask Matrix
    mask_matrix, pathway_names = create_pathway_mask(args.gmt_file, gene_list)
    mask_matrix = mask_matrix.to(device)
    
    seed_f1s = []
    seed_baccs = []
    
    ABS_ROOT = "/data/zliu/Path_MoE"
    output_subdir = f"results_{dataset_name.lower()}"
    os.makedirs(os.path.join(ABS_ROOT, output_subdir), exist_ok=True)

    for seed in range(args.seeds):
        print(f"\n🌱 Evaluating Ensemble Models from Seed {seed}...")
        
        ensemble_models = []
        for fold in range(5): 
            ckpt_path = os.path.join(ABS_ROOT, f"checkpoints_pm50/BRCA_s{seed}_fold{fold}.pth") 
            if not os.path.exists(ckpt_path):
                continue
                
            model = build_model(mask_matrix, args, device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
            model.eval()

            # ========================================================
            # 🚀 破局补丁 1：BN Test-Time Adaptation
            # 允许 BN 层根据当前外部验证集的分布更新均值和方差
            # ========================================================
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm1d):
                    m.train()
            # ========================================================

            ensemble_models.append(model)
            
        if len(ensemble_models) == 0:
            print(f"   ❌ No models loaded for Seed {seed}. Skipping...")
            continue
            
        print(f"   ✅ Loaded {len(ensemble_models)} fold models.")
        
        # 3. 集成预测
        seed_preds_probs = []
        with torch.no_grad():
            for batch in test_loader:
                x_r, x_c, x_m = batch["x_rna"].to(device), batch["x_cnv"].to(device), batch["x_met"].to(device)
                
                # ========================================================
                # 🚀 破局补丁 2：由于 GSE25066 是伪多组学，强制置零缺失模态
                # 确保模型只通过 RNA 通道提取有效生物学信息
                # ========================================================
                x_c = torch.zeros_like(x_c)
                x_m = torch.zeros_like(x_m)
                # ========================================================

                batch_probs = []
                for m in ensemble_models:
                    logits, _ = m(x_r, x_c, x_m)
                    probs = F.softmax(logits, dim=1)
                    batch_probs.append(probs)
                    
                avg_batch_probs = torch.mean(torch.stack(batch_probs), dim=0)
                seed_preds_probs.append(avg_batch_probs.cpu().numpy())
                
        seed_preds_probs = np.concatenate(seed_preds_probs, axis=0)
        final_preds = np.argmax(seed_preds_probs, axis=1)
        
        # 4. 指标计算
        acc = accuracy_score(y_true, final_preds)
        bacc = balanced_accuracy_score(y_true, final_preds)
        macro_f1 = f1_score(y_true, final_preds, average="macro")
        
        print(f"   🏁 Seed {seed} -> Macro F1: {macro_f1:.4f} | Balanced Acc: {bacc:.4f}")
        seed_f1s.append(macro_f1)
        seed_baccs.append(bacc)
        
    print("\n" + "="*60)
    print(f"🏆 Final {dataset_name} External Validation Result ({len(seed_f1s)} Seeds)")
    if len(seed_f1s) > 0:
        print(f"   ➤ Mean Macro F1    : {np.mean(seed_f1s):.4f} ± {np.std(seed_f1s):.4f}")
        print(f"   ➤ Mean Balanced Acc: {np.mean(seed_baccs):.4f} ± {np.std(seed_baccs):.4f}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/zliu/Path_MoE/data/GSE25066/processed")
    parser.add_argument("--gmt_file", type=str, default="/data/zliu/Path_MoE/data/h.all.v2023.1.Hs.symbols.gmt")
    parser.add_argument("--device", type=str, default="cuda:0") 
    parser.add_argument("--seeds", type=int, default=20) # 外部验证可以先跑5个seed看看效果
    parser.add_argument("--num_omics", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=5) 
    parser.add_argument("--top_k", type=int, default=3) 
    parser.add_argument("--gate_hidden_dim", type=int, default=128)
    parser.add_argument("--expert_hidden_dim", type=int, default=64)
    parser.add_argument("--expert_out_dim", type=int, default=16)
    parser.add_argument("--expert_dropout", type=float, default=0.3)
    parser.add_argument("--no_expert_bn", action="store_true")
    parser.add_argument("--cls_hidden_dim", type=int, default=8)
    parser.add_argument("--use_softmax", action="store_true")
    
    args = parser.parse_args()
    run_external_validation(args)