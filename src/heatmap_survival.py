import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. Path Configuration
# ==========================================
BASE_DIR = "/data/zliu/Path_MoE"
# Adjust this to the actual path where your Survival Gating files are saved
GATING_DIR = "/data/zliu/gating" 

CANCERS = ["BLCA", "BRCA", "HNSC", "KIRC", "LGG", "LUAD", "LUSC", "PRAD", "STAD", "THCA"]
SEEDS = 20

print("🔍 Scanning Gating files for all cancer types...")

# ==========================================
# 2. Calculate Selection Frequency using the "Zero-Mask" Trick
# ==========================================
cancer_freq_dict = {}

for cancer in CANCERS:
    seed_freqs = []
    for seed in range(SEEDS):
        gating_csv = os.path.join(GATING_DIR, f"{cancer}_s{seed}_gating.csv")
        
        if not os.path.exists(gating_csv):
            continue
            
        gating_df = pd.read_csv(gating_csv)
        pathways = [col for col in gating_df.columns if col != "sample_id"]
        gating_mat = gating_df[pathways].values
        
        # 🚀 CORE MAGIC: The Top-K mechanism automatically sets unselected weights to 0.
        # We directly check for > 1e-4 to accurately restore which pathways the model 
        # selected at the time, perfectly bypassing the unknown K value!
        activation_mask = (gating_mat > 1e-4).astype(float)
        
        # Calculate the average selection frequency across all patients for this Seed
        seed_freq = activation_mask.mean(axis=0)
        seed_freqs.append(seed_freq)
        
    if seed_freqs:
        # Take the grand mean (Ensemble) of the frequencies across 20 Seeds
        cancer_ensemble_freq = np.mean(seed_freqs, axis=0)
        cancer_freq_dict[cancer] = cancer_ensemble_freq

# ==========================================
# 3. Ensemble & Panoramic Plotting
# ==========================================
print("📊 Generating Pan-Cancer Panoramic Heatmap...")

# Convert dictionary to DataFrame (Rows: Pathway, Columns: Cancer)
ensemble_freq_df = pd.DataFrame(cancer_freq_dict, index=pathways)

# Clean up pathway names: Remove "HALLMARK_" prefix and replace underscores with spaces
clean_pathways = []
for p in ensemble_freq_df.index:
    clean_name = p.replace("HALLMARK_", "").replace("_", " ").title()
    clean_pathways.append(clean_name)
ensemble_freq_df.index = clean_pathways

# Filter out "dead pathways" where the selection frequency is 0 across all cancer types
plot_df = ensemble_freq_df[ensemble_freq_df.sum(axis=1) > 0]

# --- Start Plotting ---
sns.set_theme(style="white")

cg = sns.clustermap(
    plot_df, 
    cmap="YlGnBu",           
    annot=False,             
    col_cluster=True,       
    row_cluster=True,        
    linewidths=.5, 
    figsize=(10, 11),                # Widen slightly to accommodate 10 cancer types
    dendrogram_ratio=(0.2, 0.05),    # Compress the top clustering tree to save space
    cbar_pos=(1.05, 0.3, 0.03, 0.4), # Place Colorbar on the right
    cbar_kws={'label': 'Ensemble Selection Frequency'}
)

# Adjust title and label fonts
cg.ax_heatmap.set_ylabel("Hallmark Pathways", fontsize=12, fontweight='bold')
cg.ax_heatmap.set_xlabel("Cancer Types", fontsize=12, fontweight='bold')
plt.setp(cg.ax_heatmap.get_yticklabels(), rotation=0, fontsize=10) 
plt.setp(cg.ax_heatmap.get_xticklabels(), fontsize=11, rotation=45, ha='right')

plt.suptitle("Pan-Cancer Survival: Pathway Selection Frequency", y=1.05, fontsize=15, fontweight='bold')

# Save high-resolution images
output_pdf = os.path.join(BASE_DIR, "PanCancer_survival_heatmap.pdf")
output_png = os.path.join(BASE_DIR, "PanCancer_survival_heatmap.png")
cg.savefig(output_pdf, bbox_inches='tight', dpi=300)
cg.savefig(output_png, bbox_inches='tight', dpi=300)

print(f"🎉 Success! Panoramic heatmap saved to:")
print(f"   -> {output_pdf}")
print(f"   -> {output_png}")

# ==========================================
# 4. Extract Core Story Insights
# ==========================================
global_mean = ensemble_freq_df.mean(axis=1)
global_top3 = global_mean.nlargest(3)

print("\n🌍 [Global Top 3] (Proves the model captures universal lethal mechanisms across cancers):")
for pathway, score in global_top3.items():
    print(f"  ➤ {pathway} (Mean Frequency: {score:.3f})")

print("\n🎯 [Cancer-Specific Top 1] (The most lethal unique feature for each cancer type):")
for cancer in ensemble_freq_df.columns:
    cancer_top1 = ensemble_freq_df[cancer].nlargest(1)
    
    for pathway, score in cancer_top1.items():
        g_mean = global_mean[pathway]
        enrichment = score / (g_mean + 1e-6) # Enrichment multiple relative to global
        
        # Mark truly specific pathways
        marker = "🔥 (Highly Specific)" if enrichment > 1.5 else "✅"
        
        print(f"  [{cancer}]: {marker} {pathway} (Current Freq: {score:.3f} | Global: {g_mean:.3f})")