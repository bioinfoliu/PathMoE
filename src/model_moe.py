import torch
import torch.nn as nn
import torch.nn.functional as F

class PathwayExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, out_dim=16, dropout=0.3, use_bn=True):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim)]
        if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
        layers += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class TopKPathMoE(nn.Module):
    def __init__(
        self,
        gene_mask,
        num_classes=1, # 1 for Survival (BCE), >1 for Subtype (CE)
        num_omics=3,
        top_k=3,
        gate_hidden_dim=128,
        expert_hidden_dim=64,
        expert_out_dim=16,
        expert_dropout=0.3,
        expert_use_bn=True,
        cls_hidden_dim=8,
        use_softmax=False, 
        noise_std=0.5  # 🚀 关键：噪声强度
    ):
        super().__init__()
        self.num_genes, self.num_pathways = gene_mask.shape
        self.num_omics = num_omics
        self.use_softmax = use_softmax 
        self.top_k = top_k
        self.num_classes = num_classes
        self.noise_std = noise_std

        # 注册 Mask
        full_mask = torch.cat([gene_mask] * num_omics, dim=0) 
        self.register_buffer("pathway_mask", full_mask)

        # Gating Network
        self.gate = nn.Sequential(
            nn.Linear(self.num_genes, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, self.num_pathways),
        )

        # Experts
        input_dim = self.num_genes * num_omics
        self.experts = nn.ModuleList([
            PathwayExpert(input_dim, expert_hidden_dim, expert_out_dim, expert_dropout, expert_use_bn)
            for _ in range(self.num_pathways)
        ])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(expert_out_dim, cls_hidden_dim),
            nn.ReLU(),
            nn.Linear(cls_hidden_dim, self.num_classes), 
        )

    def forward(self, x_rna, x_cnv, x_met):
        # --- A. Noisy Gating ---
        gate_logits = self.gate(x_rna)

        if self.training:
            # 🚀 在 Logits 上加入噪声，打破局部最优，强制探索
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise

        topk_logits, topk_indices = torch.topk(gate_logits, k=self.top_k, dim=1)

        if self.use_softmax:
            topk_weights = F.softmax(topk_logits, dim=1)
        else:
            topk_weights = torch.sigmoid(topk_logits)

        gate_weights = torch.zeros_like(gate_logits)
        gate_weights.scatter_(1, topk_indices, topk_weights)

        # --- B. Expert Forward ---
        x_full = torch.cat([x_rna, x_cnv, x_met], dim=1) 
        expert_outputs = []
        for i in range(self.num_pathways):
            x_masked = x_full * self.pathway_mask[:, i]
            expert_outputs.append(self.experts[i](x_masked))

        expert_features = torch.stack(expert_outputs, dim=1) 

        # --- C. Weighted Fusion ---
        weighted_feat = torch.sum(expert_features * gate_weights.unsqueeze(-1), dim=1)

        # --- D. Prediction ---
        logits = self.classifier(weighted_feat)
        return logits, gate_weights