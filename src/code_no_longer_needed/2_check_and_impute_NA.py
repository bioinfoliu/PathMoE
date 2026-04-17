import os
import pandas as pd


# 配置
BASE_DIR = "/data/zliu/Path_MoE/data"
CANCERS = ["BRCA", "HNSC", "LGG", "THCA", "PRAD", "LUAD", "BLCA", "STAD", "LUSC", "KIRC"]
OMICS = ["tpm", "cnv", "met"]

print(f"{'Cancer':<10} | {'RNA NaN':<8} | {'CNV NaN':<8} | {'MET NaN':<8} | {'Surv NaN':<8}")
print("-" * 60)

for cancer in CANCERS:
    data_dir = os.path.join(BASE_DIR, cancer, "filtered_data")
    stats = []
    
    # 检查生存数据
    f_surv = os.path.join(data_dir, f"TCGA-{cancer}.survival_filtered.csv")
    if os.path.exists(f_surv):
        df_s = pd.read_csv(f_surv)
        stats.append(df_s.isna().sum().sum())
    else:
        stats.append("N/A")

    # 检查三组学
    for omic in OMICS:
        f_path = os.path.join(data_dir, f"TCGA-{cancer}.hallmark_{omic}_filtered.csv")
        if os.path.exists(f_path):
            # 注意：组学数据很大，只检查 sum
            df_o = pd.read_csv(f_path, index_col=0)
            stats.insert(-1, df_o.isna().sum().sum())
        else:
            stats.insert(-1, "N/A")
            
    print(f"{cancer:<10} | {stats[0]:<8} | {stats[1]:<8} | {stats[2]:<8} | {stats[3]:<8}")

import os
import pandas as pd

# ================= 配置区 =================
BASE_DIR = "/data/zliu/Path_MoE/data"
CANCERS = ["BRCA", "HNSC", "LGG", "THCA", "PRAD", "LUAD", "BLCA", "STAD", "LUSC", "KIRC"]
OMICS = ["tpm", "cnv", "met"]

def fix_nans():
    print("🚀 Starting NaN cleanup for 10 cancer types...")
    
    for cancer in CANCERS:
        data_dir = os.path.join(BASE_DIR, cancer, "filtered_data")
        print(f"\nProcessing {cancer}...")
        
        for omic in OMICS:
            file_name = f"TCGA-{cancer}.hallmark_{omic}_filtered.csv"
            file_path = os.path.join(data_dir, file_name)
            
            if os.path.exists(file_path):
                # 1. 读取数据
                df = pd.read_csv(file_path, index_col=0)
                
                # 2. 检查是否有 NaN
                nan_count = df.isna().sum().sum()
                if nan_count > 0:
                    # 3. 核心操作：填充 0 并覆写原文件
                    df_filled = df.fillna(0)
                    df_filled.to_csv(file_path)
                    print(f"   ✅ {omic.upper()}: Fixed {nan_count} NaNs.")
                else:
                    print(f"   ⚪ {omic.upper()}: Clean (0 NaNs).")
            else:
                print(f"   ⚠️ {omic.upper()}: File not found.")

    print("\n✨ All done! Your datasets are now pure and ready for fitting.")

if __name__ == "__main__":
    fix_nans()


print(f"{'Cancer':<10} | {'RNA NaN':<8} | {'CNV NaN':<8} | {'MET NaN':<8} | {'Surv NaN':<8}")
print("-" * 60)
print("-after fix-")
print("-" * 60)

