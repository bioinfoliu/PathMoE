# 💻 Source Code (`src`)

This directory contains the core implementation of the MoPE-MOI architecture, including model definitions, data loaders, training loops, and execution scripts for both cancer subtyping and survival prediction tasks.

## 📂 File Manifest

### 🧠 Model Architecture
* `model_moe.py`: Base implementation of the Mixture-of-Experts (MoE) network.
* `model_moe_pm50.py`: Extended TopK PathMoE architecture incorporating specific pathway priors and gating mechanisms.

### 📊 Data Processing
* `dataset_subtype.py`: PyTorch `Dataset` and `DataLoader` for handling multi-omics inputs and clinical labels for the **Cancer Subtyping** task.
* `dataset_survival.py`: PyTorch `Dataset` and `DataLoader` tailored for handling censored survival data (time-to-event) for the **Survival Analysis** task.

### 🚀 Main Execution Scripts
* `run_pathmoe_subtype.py`: The main training and evaluation loop for the subtyping classification task. Handles model initialization, optimizer setup, and metric logging.
* `run_pathmoe_survival.py`: The main training and evaluation loop for the survival prediction task (using Cox-PH loss or equivalent).

### ⚙️ Shell Scripts (Batch Execution)
* `run_brca_subtype.sh`: Bash script to automate training specifically on the BRCA (Breast Cancer) cohort with predefined hyperparameters.
* `run_pan_cancer.sh`: Bash script to iterate the pipeline across multiple cancer types for pan-cancer analysis.

### 🛠️ Utilities
* `utils.py`: Contains auxiliary functions such as custom loss calculations, random seed fixing, and structural mask generation for pathway priors.

## 🏃 How to Run
To execute a basic subtyping experiment for BRCA, navigate to the root directory and run:
```bash
bash src/run_brca_subtype.sh
