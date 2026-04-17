# MoPE-MOI
**Pathway-Guided Mixture-of-Experts for Multi-omics Integration**

MoPE-MOI is a deep learning framework designed for multi-omics integration in cancer research. It utilizes a biologically informed, sparse Mixture-of-Experts (MoE) architecture to provide state-of-the-art predictive performance while maintaining strict biological interpretability and patient-level mechanistic traceability.

## ✨ Key Features
* **Pathway-Guided Experts:** Employs 50 MSigDB Hallmark pathways as structural priors to prevent "black-box" predictions.
* **Asymmetric Routing:** Uses a parsimonious RNA-only global gating network by default to filter dimensionality-induced noise, while local experts process full multi-omics tensors.
* **Stochastic Routing (Noisy Gating):** Injects tunable Gaussian noise during routing to prevent expert collapse and preserve inter-patient tumor heterogeneity.
* **Non-Competitive Activation:** Replaces traditional zero-sum Softmax with independent Sigmoid activation, faithfully capturing concurrent hyperactivation of oncogenic hallmarks.
* **Robust Evaluation:** Built-in nested 5-Fold cross-validation ensemble with automated hyperparameter tuning via Optuna.

## ⚙️ Environment Setup
Create a virtual environment and install the required dependencies:
```bash
conda create -n pathmoe python=3.9 -y
conda activate pathmoe
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scipy scikit-learn optuna
```

## 📂 Data Preparation
All multi-omics datasets (RNA, CNV, Methylation) should be processed and placed in the designated data directory. The default path configured in the scripts is `/data/zliu/Path_MoE/data/`.

Ensure you have the MSigDB Hallmark gene set downloaded:
* `h.all.v2023.1.Hs.symbols.gmt`

## 🚀 Quick Start: Subtype Classification

The `run_pathmoe_subtype.py` script handles the complete pipeline, including Optuna hyperparameter searching and 20-seed ensemble evaluation.

**Standard Run (Asymmetric RNA-only Router + Noisy Gating + Sigmoid):**
```bash
python run_pathmoe_subtype.py \
    --cancer BRCA \
    --seeds 20 \
    --noise_std 0.2 \
    --save_gating \
    --save_results_json ./results/BRCA_standard.json
```

**Ablation Run 1: Symmetric Tri-omics Router:**
```bash
python run_pathmoe_subtype.py \
    --cancer BRCA \
    --seeds 20 \
    --use_tri_gating \
    --noise_std 0.2 \
    --save_results_json ./results/BRCA_triomics.json
```

**Ablation Run 2: Traditional Softmax Activation (Competitive):**
```bash
python run_pathmoe_subtype.py \
    --cancer BRCA \
    --seeds 20 \
    --use_softmax \
    --noise_std 0.2 \
    --save_results_json ./results/BRCA_softmax.json
```

## 📁 Repository Structure
```text
PathMoE/
├── model_moe_pm50.py         # Core TopKPathMoE architecture (Gating & Experts)
├── run_pathmoe_subtype.py    # Main training/evaluation script (Subtyping)
├── dataset_subtype.py        # PyTorch Dataset for multi-omics loading
├── utils.py                  # Utility functions (Mask generation, etc.)
└── README.md
```

## 📊 Outputs
Running the pipeline will automatically generate three directories:
* `predictions_subtype/`: Contains detailed per-sample predictions.
* `gating_subtype/`: Contains the sparse routing weights for biological interpretation.
* `checkpoints_subtype/`: Stores model weights for each fold.
