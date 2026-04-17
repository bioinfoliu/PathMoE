import os
import pandas as pd

# Directly pointing to the folder in your screenshot
data_dir = "/data/zliu/Path_MoE/data/BRCA/filtered_subtype_data"

files_to_check = [
    "TCGA-BRCA.hallmark_tpm_filtered.csv",
    "TCGA-BRCA.hallmark_cnv_filtered.csv",
    "TCGA-BRCA.hallmark_met_filtered.csv",
    "TCGA-BRCA.subtype_filtered.csv"
]

print("-" * 50)
print(f"{'File Name':<40} | {'NaN Count'}")
print("-" * 50)

for file in files_to_check:
    file_path = os.path.join(data_dir, file)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0)
        nan_count = df.isna().sum().sum()
        
        if nan_count == 0:
            print(f"{file:<40} | ✅ 0")
        else:
            print(f"{file:<40} | ❌ {nan_count}")
    else:
        print(f"{file:<40} | ⚠️ File Not Found")
print("-" * 50)