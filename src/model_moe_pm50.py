import torch
import torch.nn as nn
import torch.nn.functional as F

class PathwayExpert(nn.Module):
    """
    单个通路专家。
    输入已经被 Mask 处理过，只包含该通路相关的 RNA/CNV/MET 信息。
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 16,
        dropout: float = 0.3,
        use_bn: bool = True,
    ):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers += [
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TopKPathMoE(nn.Module):
    """
    核心模型：通路级 Mixture-of-Experts (带 Noisy Gating)
    用于亚型分类或生存分析。
    """
    def __init__(
        self,
        gene_mask: torch.Tensor,
        num_classes: int,
        num_omics: int = 3,
        top_k: int = 3,
        gate_hidden_dim: int = 128,
        expert_hidden_dim: int = 64,
        expert_out_dim: int = 16,
        expert_dropout: float = 0.3,
        expert_use_bn: bool = True,
        cls_hidden_dim: int = 8,
        use_softmax: bool = False,
        noise_std: float = 0.5,
        use_tri_gating: bool = False  # 🚀 新增：Tri-omics Router 开关
    ):
        super().__init__()
        self.num_genes, self.num_pathways = gene_mask.shape
        self.num_omics = num_omics
        self.use_softmax = use_softmax
        self.top_k = top_k
        self.expert_out_dim = expert_out_dim
        self.num_classes = num_classes
        self.noise_std = noise_std
        self.use_tri_gating = use_tri_gating # 🚀 保存开关状态

        # 1. 注册 Pathway Mask [Genes * num_omics, num_pathways]
        full_mask = torch.cat([gene_mask] * num_omics, dim=0)
        self.register_buffer("pathway_mask", full_mask)

        # 2. Gating Network (Router)
        # 🚀 动态计算输入维度：RNA-only 或 Tri-omics
        gate_in_dim = self.num_genes * num_omics if use_tri_gating else self.num_genes
        
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, self.num_pathways),
        )

        # 3. 专家网络：每个 Pathway 一个 Expert
        input_dim = self.num_genes * num_omics
        self.experts = nn.ModuleList(
            [
                PathwayExpert(
                    input_dim=input_dim,
                    hidden_dim=expert_hidden_dim,
                    out_dim=expert_out_dim,
                    dropout=expert_dropout,
                    use_bn=expert_use_bn,
                )
                for _ in range(self.num_pathways)
            ]
        )

        # 4. 最终分类头
        self.classifier = nn.Sequential(
            nn.Linear(expert_out_dim, cls_hidden_dim),
            nn.ReLU(),
            nn.Linear(cls_hidden_dim, self.num_classes),
        )

    def forward(
        self,
        x_rna: torch.Tensor,
        x_cnv: torch.Tensor,
        x_met: torch.Tensor,
    ):
        # --- A. Gating Network (Noisy Top-K) ---
        # 🚀 根据开关准备 Router 的输入
        if self.use_tri_gating:
            gate_input = torch.cat([x_rna, x_cnv, x_met], dim=1)
        else:
            gate_input = x_rna
            
        gate_logits = self.gate(gate_input)

        # 🚀 仅在训练模式下注入噪声，强迫模型探索
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise

        topk_logits, topk_indices = torch.topk(gate_logits, k=self.top_k, dim=1)

        if self.use_softmax:
            topk_weights = F.softmax(topk_logits, dim=1)
        else:
            topk_weights = torch.sigmoid(topk_logits)

        # 构建稀疏权重矩阵 [Batch, num_pathways]
        gate_weights = torch.zeros_like(gate_logits)
        gate_weights.scatter_(1, topk_indices, topk_weights)

        # --- B. Expert Forward (带 Masking) ---
        x_full = torch.cat([x_rna, x_cnv, x_met], dim=1)

        expert_outputs = []
        for i in range(self.num_pathways):
            mask_i = self.pathway_mask[:, i]
            x_masked = x_full * mask_i
            out = self.experts[i](x_masked)
            expert_outputs.append(out)

        # [Batch, num_pathways, expert_out_dim]
        expert_features = torch.stack(expert_outputs, dim=1)

        # --- C. 加权融合 ---
        weighted_feat = torch.sum(
            expert_features * gate_weights.unsqueeze(-1), dim=1
        )

        # --- D. 预测 ---
        logits = self.classifier(weighted_feat)
        return logits, gate_weights